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
from model.LSTM2 import RNNModel


from utils import *
from math import ceil
import lptorch as lp

import argparse
import numpy as np
import pdb

global model, first_module, trainset, testset, trainloader, testloader, cos

def run():
    parser = argparse.ArgumentParser(description='Test activation/error format accuracy')
    parser.add_argument('--model', default='resnet18', help='test target model')
    parser.add_argument('--dataset', default='cifar10', help='test target dataset')
    parser.add_argument('--iter', default=100, help='test target dataset')
    parser.add_argument('--gpu', default='0', help='gpu number')
    args = parser.parse_args()
    
    device = torch.device('cuda:'+args.gpu)

    model_list = {'imagenet':{'resnet18':resnet18, 'resnet50':resnet50, 'mobilenetv2':mobilenet_v2}, 'cifar10':{'resnet18':resnet18_cifar}, 'cifar100':{'resnet18':resnet18_cifar}, 'PTB':{'2LSTM':RNNModel}}

    if args.dataset not in model_list.keys():
        print('check your dataset!')
        return
    elif args.model not in model_list[args.dataset].keys():
        print('check your dataset & model!')
        return

    if args.dataset == 'cifar10':
        model = model_list[args.dataset][args.model](10).to(device)
    elif args.dataset == 'cifar100':
        model = model_list[args.dataset][args.model](100).to(device)
    elif args.dataset == 'imagenet':
        model = model_list[args.dataset][args.model]().to(device)
    elif args.dataset == 'PTB':
        model = model_list[args.dataset][args.model]('LSTM', 33278, 650, 650, 2, 0.5, True).to(device)

    if args.model == 'resnet18':
        first_module = model.conv1.module
    elif args.model == 'mobilenetv2':
        first_module = model.features[0][0].module
    elif args.model == '2LSTM':
        first_module = model.encoder

    state = load_checkpoint(args.dataset+'_'+args.model, device)
    model.load_state_dict(state['net'])

    dataset = args.dataset
    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    elif dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        ])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    elif dataset == 'imagenet':
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        trainset = torchvision.datasets.ImageFolder(root='../ImageNet/train', transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=8)
        testset = torchvision.datasets.ImageFolder(root='../ImageNet/val', transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    # elif dataset == 'PTB':

    criterion = nn.CrossEntropyLoss().to(device)
    room = 3
    fp8_format = [None, 
                # lp.quant.quant(lp.quant.custom_fp_format([5,5,5,5,5,5,5,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([5,5,5,5,5,5,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([5,5,5,5,5,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([5,5,5,5,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([5,5,5,4,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([5,5,4,4,4,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([5,4,4,4,4,4,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4]), room=room, tracking=False)]

                lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([5,4,4,4,4,4,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,5,4,4,4,4,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,5,4,4,4,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,5,4,4,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,5,4,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,5,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,5,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,5,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,5,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,5,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,4,5,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,4,4,5,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,4,4,4,5,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,4,4,4,4,5,3,2,1,0]), room=room, tracking=False),

                # lp.quant.quant(lp.quant.custom_fp_format([5,5,4,4,4,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,5,5,4,4,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,5,5,4,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,5,5,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,5,5,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,5,5,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,5,5,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,5,5,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,5,5,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,5,5,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,4,5,5,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,4,4,5,5,3,2,1,0]), room=room, tracking=False),
                
                # lp.quant.quant(lp.quant.custom_fp_format([5,5,5,4,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,5,5,5,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,5,5,5,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,5,5,5,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,5,5,5,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,5,5,5,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,5,5,5,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,5,5,5,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,5,5,5,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,5,5,5,3,2,1,0]), room=room, tracking=False),
                
                lp.quant.quant(lp.quant.custom_fp_format([5,5,5,5,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                lp.quant.quant(lp.quant.custom_fp_format([4,5,5,5,5,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                lp.quant.quant(lp.quant.custom_fp_format([4,4,5,5,5,5,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                lp.quant.quant(lp.quant.custom_fp_format([4,4,4,5,5,5,5,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,5,5,5,5,4,4,4,3,2,1,0]), room=room, tracking=False),
                lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,5,5,5,5,4,4,3,2,1,0]), room=room, tracking=False),
                lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,5,5,5,5,4,3,2,1,0]), room=room, tracking=False),
                lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,5,5,5,5,3,2,1,0]), room=room, tracking=False),
                
                lp.quant.quant(lp.quant.custom_fp_format([5,5,5,5,5,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                lp.quant.quant(lp.quant.custom_fp_format([4,5,5,5,5,5,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                lp.quant.quant(lp.quant.custom_fp_format([4,4,5,5,5,5,5,4,4,4,3,2,1,0]), room=room, tracking=False),
                lp.quant.quant(lp.quant.custom_fp_format([4,4,4,5,5,5,5,5,4,4,3,2,1,0]), room=room, tracking=False),
                lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,5,5,5,5,5,4,3,2,1,0]), room=room, tracking=False),
                lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,5,5,5,5,5,3,2,1,0]), room=room, tracking=False),
                
                lp.quant.quant(lp.quant.custom_fp_format([5,5,5,5,5,5,4,4,4,3,2,1,0]), room=room, tracking=False),
                lp.quant.quant(lp.quant.custom_fp_format([4,5,5,5,5,5,5,4,4,3,2,1,0]), room=room, tracking=False),
                lp.quant.quant(lp.quant.custom_fp_format([4,4,5,5,5,5,5,5,4,3,2,1,0]), room=room, tracking=False),
                lp.quant.quant(lp.quant.custom_fp_format([4,4,4,5,5,5,5,5,5,3,2,1,0]), room=room, tracking=False),
                
                lp.quant.quant(lp.quant.custom_fp_format([5,5,5,5,5,5,5,4,3,2,1,0]), room=room, tracking=False),
                lp.quant.quant(lp.quant.custom_fp_format([4,5,5,5,5,5,5,5,3,2,1,0]), room=room, tracking=False)]    
                
                
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,4,4,4,4,4,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,4,4,4,4,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,4,4,4,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,4,4,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,4,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,0]), room=room, tracking=False),
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,3,2,1,0]), room=room, tracking=False)]
                # lp.quant.quant(lp.quant.custom_fp_format([4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1,1,1,1,0]), room=room, tracking=False)]
                # lp.quant.quant(lp.quant.custom_fp_format([5,5,5,5,5,5,5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]), room=room, tracking=False)]
    format_label = ['base'] + [str(i) for i in range(len(fp8_format)-1)]

    cos = torch.nn.CosineSimilarity(0)
    def get_similarity(list):
        result = []
        for i in range(1,len(list)):
            result.append(torch.unsqueeze(cos(list[0], list[i]),0))
        return torch.unsqueeze(torch.cat(result),0)
        
    cosine_value = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx == int(args.iter):
            break
        model.eval()
        inputs, targets = inputs.to(device), targets.to(device)
        for idx, form in enumerate(fp8_format):
            # def savegrad(self, grad_input, grad_output):
            #     if not os.path.isdir('activ_test'):
            #         os.mkdir('activ_test')
            #     current = os.getcwd()
            #     path = os.path.join(current, 'activ_test', args.dataset+'_'+args.model)
            #     if not os.path.isdir(path):
            #         os.mkdir(path)
            #     os.chdir(path)
            #     np.save(format_label[idx]+'.npy', grad_output[0].cpu().numpy())
            #     os.chdir(current)

            # lp.set_error_quant(form)
            # # lp.set_activ_quant(form)
            # handle = first_module.register_backward_hook(savegrad)
            # outputs = model(inputs)
            # loss = criterion(outputs, targets)
            # loss.backward()
            # handle.remove()
            model.zero_grad()
            lp.set_error_quant(form)
            outputs = model(inputs.clone())
            loss = criterion(outputs, targets)
            loss.backward()
            if not os.path.isdir('activ_test'):
                os.mkdir('activ_test')
            current = os.getcwd()
            path = os.path.join(current, 'activ_test', args.dataset+'_'+args.model)
            if not os.path.isdir(path):
                os.mkdir(path)
            os.chdir(path)
            np.save(format_label[idx]+'.npy', first_module.weight.grad.cpu().numpy())
            os.chdir(current)

    result = []
    for label in format_label:
        current = os.getcwd()
        path = os.path.join(current, 'activ_test', args.dataset+'_'+args.model)
        os.chdir(path)
        result.append(torch.from_numpy(np.load(label+'.npy')).reshape(-1))
        os.chdir(current)
    cosine_value.append(get_similarity(result))
        
    cosine_value = torch.cat(cosine_value)
    current = os.getcwd()
    path = os.path.join(current, 'activ_test', args.dataset+'_'+args.model)
    os.chdir(path)
    np.save('result.npy', cosine_value.numpy())
    os.chdir(current)

if __name__ == "__main__":
    run()