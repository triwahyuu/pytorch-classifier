import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import models, transforms
from PIL import Image
from utils import prepare_model

import os

class Classifier(object):
    def __init__(self, pretrained_path, cuda=True):
        super(Classifier, self).__init__()

        pretrained = torch.load(pretrained_path)
        self.model, _, _ = prepare_model(pretrained['arch'], pretrained['num_classes'])
        self.model.load_state_dict(pretrained['model_state_dict'], strict=True)

        self.is_cuda = cuda
        if self.is_cuda:
            self.model = self.model.cuda()
            self.is_cuda = True
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __call__(self, input_imgs_or_path):
        input_imgs = []
        single_pred = False
        if isinstance(input_imgs_or_path, str):
            if os.path.isdir(input_imgs_or_path):
                input_imgs = [
                    os.path.join(input_imgs_or_path, f) for f in os.listdir(input_imgs_or_path) \
                        if os.path.isfile(os.path.join(input_imgs_or_path, f))
                ]
                if len(input_imgs) == 0:
                    raise RuntimeError("no files in directory")
            elif os.path.isfile(input_imgs_or_path):
                input_imgs = [input_imgs_or_path]
                single_pred = True
            else:
                raise RuntimeError("unrecognized input")
        elif isinstance(input_imgs_or_path, list):
            input_imgs = input_imgs_or_path
        else:
            raise RuntimeError("unrecognized input")

        result = []
        for img_path in input_imgs:
            img = self.preprocess(Image.open(img_path)).unsqueeze(0)
            if self.is_cuda:
                img = img.cuda()
            
            with torch.no_grad():
                pred = torch.sigmoid(self.model(img))
            _, cls_pred = pred.max(-1)
            result.append(cls_pred)

        if single_pred:
            return result[-1]
        else:
            return result


if __name__ == "__main__":
    import argparse

    data_dir = os.path.join(os.path.dirname(__file__), "dataset", "val")
    category = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    category.sort()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='result/resnet18/resnet18_best.pth', 
        help='pretrained model path')
    parser.add_argument('--input', default='', 
        help='input path for single image or folder containing images')
    args = parser.parse_args()

    classifier = Classifier(args.model)

    if args.input != '':
        out = classifier(args.input)
        if isinstance(out, list):
            for idx in out:
                print(category[idx])
        else:
            print(category[out])
    else:
        input_imgs = "data"
        out = classifier(input_imgs)
        print([category[i] for i in out])
