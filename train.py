from __future__ import print_function, division

import multiprocessing
multiprocessing.set_start_method('spawn', True)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import tqdm

import os
import time
import argparse
import datetime


def prepare_dataloaders(datapath='dataset/', img_size=224, batch_size=4):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(int(img_size*1.43)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(datapath, x), data_transforms[x]) \
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                   shuffle=True, num_workers=4, pin_memory=True) for x in ['train', 'val']}

    return dataloaders


def prepare_model(arch, n_class, freeze_layer=True):
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
            nn.Linear(in_features),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        model.to(device)

        optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

    ## loss function
    criterion = nn.CrossEntropyLoss().to(device)
    return model, criterion, optimizer

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

def train_model(model, arch, dataloaders, criterion, optimizer, 
        scheduler=None, num_epochs=25, output_path='result/', start_epoch=0):

    best_acc = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "tb"), exist_ok=True)

    writer = SummaryWriter(os.path.join(output_path, "tb"))
    with open(os.path.join(output_path, "log.txt"), 'w') as f:
        metrics = "phase, epoch, loss, correct"
        f.write(metrics + '\n')

    for epoch in tqdm.trange(start_epoch, num_epochs, total=num_epochs, desc="epoch ", ncols=80):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm.tqdm(dataloaders[phase], total=len(dataloaders[phase]), 
                    desc=f" {phase}", ncols=80, leave=False):

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backpropagation
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                loss_data = loss.item() * inputs.size(0)
                running_loss += loss_data
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            writer.add_scalar(f"loss/{phase}", epoch_loss, epoch)
            writer.add_scalar(f"accuracy/{phase}", epoch_acc, epoch)

            with open(os.path.join(output_path, "log.txt"), 'a') as f:
                metrics = "{}, {}, {:.10f}, {:.10f}".format(phase, epoch, 
                    epoch_loss, epoch_acc)
                f.write(metrics + '\n')

            # deep copy the model
            if phase == 'val':
                torch.save({
                    'arch': arch, 'epoch': epoch,
                    'optim_state_dict': optimizer.state_dict(),
                    'model_state_dict': model.state_dict(),
                    'best_acc': best_acc,
                }, os.path.join(output_path, 'checkpoint.pth'))

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save({
                        'arch': arch, 'epoch': epoch,
                        'optim_state_dict': optimizer.state_dict(),
                        'model_state_dict': model.state_dict(),
                        'best_acc': best_acc,
                    }, os.path.join(output_path, f'{arch}_best.pth'))

    print('Best Acc: {:4f}'.format(best_acc))


if __name__ == "__main__":
    model_choices = models.resnet.__all__[1:] + models.vgg.__all__[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--arch", metavar="arch", default="resnet101", choices=model_choices,
        help="model architecture: " + " | ".join(model_choices) +  "(default: resnet101)"
    )
    parser.add_argument(
        '--datapath', default='dataset/', help='suction grasp dataset path',
    )
    parser.add_argument(
        '--output-path', default='', help='training result path',
    )
    parser.add_argument("--resume", default="", type=str, help="checkpoint path to resume training")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
    args = parser.parse_args()

    print(f"Training {args.arch}")
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    ## dataset
    dataloaders = prepare_dataloaders(args.datapath, args.img_size, args.batch_size)
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
    class_names = dataloaders['train'].dataset.classes
    n_class = len(class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## models
    model, criterion, optimizer = prepare_model(args.arch, n_class)

    start_epoch = 0
    if args.resume != '':
        checkpoint = torch.load(args.resume)
        if checkpoint["arch"] != args.arch:
            raise ValueError
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_model(model, args.arch, dataloaders, criterion, optimizer, scheduler=scheduler, 
        num_epochs=args.epochs, output_path=os.path.join("result", args.arch, now), start_epoch=start_epoch)
