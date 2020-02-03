import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt


def visualise(inp, title=None):
    if type(inp) == torch.Tensor:
        img = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
    
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.show()


def prepare_model(arch, n_class, freeze_layer=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = getattr(models, arch)(pretrained=True)
    if freeze_layer:
        for param in model.parameters():
            param.requires_grad = False
    
    optimizer = None
    if "resnet" in arch:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, n_class)
        model.to(device)

        optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    elif "vgg" in arch:
        num_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, n_class),
        )
        model.to(device)

        optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

    ## loss function
    criterion = nn.CrossEntropyLoss().to(device)
    return model, criterion, optimizer


def is_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True
    except:
        return False
