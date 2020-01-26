#!/usr/bin/env bash

set -o xtrace

resnet=$(echo resnet{18,34,50,101,152})
vgg=$(echo vgg{11,13,16,19})


function train(){
    for a in $@; do
        python3 train.py -a $a --epoch 100
    done
}

train $resnet $vgg