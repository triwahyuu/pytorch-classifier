# PyTorch Transfer Learning for Image Classification

## Get Started
To get started with this project, open the [example notebook](example.ipynb).

### Setup  
You need to install the dependencies of this project, execute:
```
pip install -r requirements.txt
```

### Training  
To train model, use the `train.py` script. 
- put your dataset in `dataset` folder. Or if you just want to try, use [hymenoptera dataset from kaggle](https://kaggle.com/ajayrana/hymenoptera-data) and extract it in `dataset` folder.
- run train script:
``` bash
python train.py --arch resnet18 --data-path dataset/hymenoptera
```

all available arguments for the script:
```
usage: train.py [-h] [-a arch] [--data-path DATA_PATH] [--resume RESUME]
                [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                [--img-size IMG_SIZE] [--use-scheduler]

optional arguments:
  -h, --help            show help message and exit
  -a arch, --arch arch  model architecture: resnet18 | resnet34 | resnet50 |
                        resnet101 | resnet152 | resnext50_32x4d |
                        resnext101_32x8d | wide_resnet50_2 | wide_resnet101_2
                        | vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 |
                        vgg16_bn | vgg19_bn | vgg19 (default: resnet18)
  --data-path DATA_PATH
                        dataset path
  --resume RESUME       checkpoint path to resume training
  --epochs EPOCHS       number of epochs of training
  --batch-size BATCH_SIZE
                        batch size for training
  --img-size IMG_SIZE   image dataset size in training
  --use-scheduler       enable training using lr scheduler
```

### Inference
To run your trained model with and image input [`data/ants.jpg`](data/ants.jpg), execute:
```
python classify.py --model result/resnet18/resnet18_best.pth --input data/ants.jpg
```

all available arguments:
```
usage: classify.py [-h] [--model MODEL] [--input INPUT] [--no-visualise]

optional arguments:
  -h, --help      show help message and exit
  --model MODEL   pretrained model path
  --input INPUT   input path for single image or folder containing images
  --no-visualise  don't visualise classification result
```
