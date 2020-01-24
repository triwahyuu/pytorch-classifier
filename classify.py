import torch
import torch.nn as nn
from torchvision import models

class Classifier():
    def __init__(self, pretrained_path):
        super(Classifier, self).__init__()

        pretrained = torch.load(pretrained_path)
        self.model = getattr(models, pretrained['arch'])(pretrained=True)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    def __call__(self, input_img):
        pred = self.model(input_img)
        _, cls_pred = pred.max(-1)
        return cls_pred