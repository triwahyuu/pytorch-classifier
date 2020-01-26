import sys
import os
sys.path.append(os.path.abspath(""))

from classify import Classifier

def test_classifier():
    classifier = Classifier("result/resnet18/20200124_213228/resnet18_best.pth")
    category = ['ants', 'bees']
    print(category[classifier("dataset/val/ants/10308379_1b6c72e180.jpg")])
    print(category[classifier("dataset/val/bees/151603988_2c6f7d14c7.jpg")])

if __name__ == "__main__":
    test_classifier()