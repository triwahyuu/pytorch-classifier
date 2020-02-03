from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from utils import prepare_model, is_notebook

import os
import time
import argparse
import datetime
from tqdm.auto import trange, tqdm

available_models = models.resnet.__all__[1:] + models.vgg.__all__[1:]

def prepare_dataloaders(datapath='dataset/', img_size=224, batch_size=4):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomPerspective(distortion_scale=0.2),
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
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
                        shuffle=True, num_workers=4, pin_memory=True),
                    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=1,
                        shuffle=False, num_workers=1, pin_memory=True)}

    return dataloaders


def train_model(model, arch, dataloaders, criterion, optimizer, 
        scheduler=None, num_epochs=25, output_path='result/', start_epoch=0):

    best_acc = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
    
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "tb"), exist_ok=True)

    writer = SummaryWriter(os.path.join(output_path, "tb"))
    with open(os.path.join(output_path, "log.txt"), 'w') as f:
        metrics = "phase, epoch, loss, accuracy"
        f.write(metrics + '\n')
    
    if not os.path.exists(os.path.join(output_path.split("/")[0], "summary.txt")):
        open(os.path.join(output_path.split("/")[0], "summary.txt"), 'w').close()

    ncols = 600 if is_notebook() else 80
    for epoch in trange(start_epoch, num_epochs, desc="epoch ", ncols=ncols):
        trainval_loss = {'train': 0.0, 'val': 0.0}
        trainval_acc = {'train': 0.0, 'val': 0.0}
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], total=len(dataloaders[phase]), 
                    desc=" {}".format(phase), ncols=ncols, leave=False):

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
            trainval_loss[phase] = epoch_loss
            trainval_acc[phase] = epoch_acc

            writer.add_scalar("loss/{}".format(phase), epoch_loss, epoch)
            writer.add_scalar("accuracy/{}".format(phase), epoch_acc, epoch)

            with open(os.path.join(output_path, "log.txt"), 'a') as f:
                metrics = "{}, {}, {:.10f}, {:.10f}".format(phase, epoch, 
                    epoch_loss, epoch_acc)
                f.write(metrics + '\n')

            # deep copy the model
            if phase == 'val':
                torch.save({
                    'arch': arch, 'epoch': epoch, 'best_acc': best_acc,
                    'num_classes': len(dataloaders['train'].dataset.classes),
                    'optim_state_dict': optimizer.state_dict(),
                    'model_state_dict': model.state_dict(),
                }, os.path.join(output_path, 'checkpoint.pth'))

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save({
                        'arch': arch, 'epoch': epoch, 'best_acc': best_acc,
                        'num_classes': len(dataloaders['train'].dataset.classes),
                        'optim_state_dict': optimizer.state_dict(),
                        'model_state_dict': model.state_dict(),
                    }, os.path.join(output_path, '{}_best.pth'.format(arch)))

        writer.add_scalars('combined_loss', trainval_loss, epoch)
        writer.add_scalars('combined_accuracy', trainval_acc, epoch)

    print("Best Acc: {:4f}".format(best_acc))
    print("best trained model is saved to: {}".format(os.path.join(output_path, arch + '_best.pth')))
    with open(os.path.join(output_path.split("/")[0], "summary.txt"), 'a') as f:
        f.write("{}, {}\n".format(output_path, best_acc))


def main(arch="resnet18", data_path="dataset/", resume="", epochs=25, 
         batch_size=4, img_size=224, use_scheduler=False, **kwargs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    print("Training {} on {}".format(arch, device))
    if is_notebook():
        print("you can also check progress on tensorboard, execute in terminal:")
        print("  > tensorboard --logdir result/<model_name>/tb/")

    ## dataset
    dataloaders = prepare_dataloaders(data_path, img_size, batch_size)
    class_names = dataloaders['train'].dataset.classes
    n_class = len(class_names)

    ## models
    model, criterion, optimizer = prepare_model(arch, n_class)

    start_epoch = 0
    if resume != '':
        checkpoint = torch.load(resume)
        if checkpoint["arch"] != arch:
            raise ValueError
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
    else:
        tb_path = os.path.join("result", arch, "tb")
        if os.path.exists(tb_path) and len(os.listdir(tb_path)) > 0:
            import shutil
            for f in os.listdir(tb_path):
                p = os.path.join(tb_path, f)
                if os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.remove(os.path.join(tb_path, f))
    
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_model(model, arch, dataloaders, criterion, optimizer, scheduler=scheduler, 
        num_epochs=epochs, output_path=os.path.join("result", arch), start_epoch=start_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--arch", metavar="arch", default="resnet18", choices=available_models,
        help="model architecture: " + " | ".join(available_models) +  " (default: resnet18)"
    )
    parser.add_argument('--data-path', default='dataset/', help='dataset path')
    parser.add_argument("--resume", default="", type=str, help="checkpoint path to resume training")
    parser.add_argument("--epochs", type=int, default=25, help="number of epochs of training")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size for training")
    parser.add_argument("--img-size", type=int, default=224, help="image dataset size in training")
    parser.add_argument('--use-scheduler', action='store_true', help='use lr scheduler')
    args = vars(parser.parse_args())

    main(**args)
