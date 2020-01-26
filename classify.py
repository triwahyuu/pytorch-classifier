import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from utils import prepare_model

import os

class Classifier(object):
    def __init__(self, pretrained_path, cuda=True):
        super(Classifier, self).__init__()

        pretrained = torch.load(pretrained_path)
        self.model, _, _ = prepare_model(pretrained['arch'], 2)
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
    
    def __call__(self, input_img):
        img = self.preprocess(Image.open(input_img)).unsqueeze(0)
        if self.is_cuda:
            img = img.cuda()
        
        with torch.no_grad():
            pred = torch.sigmoid(self.model(img))
        _, cls_pred = pred.max(-1)
        return cls_pred