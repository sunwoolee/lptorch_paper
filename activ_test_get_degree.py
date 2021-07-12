import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from model.resnet_cifar import ResNet18 as resnet18_cifar
from model.resnet import resnet18, resnet50
from model.mobilenet import mobilenet_v2


from utils import *
from math import ceil
import lptorch as lp

import argparse
import numpy as np
import pdb
from math import pi

def run():
    parser = argparse.ArgumentParser(description='Test activation/error format accuracy')
    parser.add_argument('--model', default='resnet18', help='test target model')
    parser.add_argument('--dataset', default='cifar10', help='test target dataset')
    args = parser.parse_args()

    current = os.getcwd()
    path = os.path.join(current, 'activ_test', args.dataset+'_'+args.model)
    os.chdir(path)
    cosine_value = np.load('result.npy')
    os.chdir(current)

    degree_value = np.arccos(cosine_value) / pi * 180
    cosine_value_mean = cosine_value.mean(0)
    degree_value_mean = degree_value.mean(0)
    print(cosine_value_mean)
    print(degree_value_mean)

if __name__ == "__main__":
    run()