from __future__ import print_function
import time
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from model.resnet_test import resnet18

import lptorch as lp
from utils import *


best_acc = 0
def run():
    global best_acc
    # Random seed
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    # Data
    print('==> Preparing data..')
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
    '''
    #### data directory in local computer ####
    trainset = torchvision.datasets.ImageFolder(root='C:/Users/User/Imagenet/train', transform=transform_train)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=8)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = torchvision.datasets.ImageFolder(root='C:/Users/User/Imagenet/val', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=8)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    '''
    #### data directory in server ####
    trainset = torchvision.datasets.ImageFolder(root='../ImageNet/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=8)

    testset = torchvision.datasets.ImageFolder(root='../ImageNet/val', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=8)
    
    fp8_format = [5,5,5,5,5,5,5,4,3,2,1,0]
    fp4_format = [1,1,1,1,1,1,1,0]
    afp4_format = [2,2,2,2,2,2,2,1,0]
    lp.set_activ_quant(lp.quant.quant(lp.quant.custom_fp_format(fp8_format), room=1))
    lp.set_error_quant(lp.quant.quant(lp.quant.custom_fp_format(fp8_format), room=1, stochastic=True))
    lp.set_kactiv_quant(lp.quant.quant(lp.quant.custom_fp_format(fp8_format), room=1))
    lp.set_kerror_quant(lp.quant.quant(lp.quant.custom_fp_format(fp4_format), room=1, stochastic=True))
    # lp.set_weight_quant(lp.quant.quant(lp.quant.custom_fp_format(fp4_format), room=0))
    # lp.set_grad_quant(lp.quant.quant(lp.quant.custom_fp_format(fp8_format), room=2))
    # lp.set_master_quant(lp.quant.quant(lp.quant.fp_format(exp_bit=6, man_bit=9), stochastic=True))
    # lp.set_hysteresis_update(False)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = torch.device("cuda")
    model = resnet18().to(device)

    bn_param = []
    base_param = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_param += list(m.parameters())
        elif isinstance(m, nn.Conv2d):
            base_param += list(m.parameters())
        elif isinstance(m, nn.Linear):
            base_param += list(m.parameters())
    # base_param = list(set(model.parameters()) - set(bn_param))

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer_base = lp.optim.SGD(base_param, lr=0.1, momentum=0.9, weight_decay=1e-4)
    optimizer_bn = lp.optim.SGD(bn_param, lr=0.1, momentum=0.9, weight_decay=1e-4, weight_quantize=False)

    
    lr_epoch = [30, 60]
    scheduler_base = optim.lr_scheduler.MultiStepLR(optimizer_base, milestones=lr_epoch, gamma=0.1)
    scheduler_bn = optim.lr_scheduler.MultiStepLR(optimizer_bn, milestones=lr_epoch, gamma=0.1)

    folder_name = 'imagenet_resnet18_error_FP4_subnormal_stochastic'
    

    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def set_scale():
        print('calculating initial scale...')
        model.train()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if batch_idx != 0 and batch_idx % 10 == 0:
                print(str(batch_idx)+'%..', end='', flush=True)
            if batch_idx == 100:
                print('')
                return
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets) #*1/16
            
            loss.backward()
        

    def train(epoch, end_epoch):
        print('\nEpoch: %d|%d \t LR: %.4f' % (epoch+1, end_epoch, scheduler_base.get_last_lr()[0]))
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        optimizer_base.zero_grad()
        optimizer_bn.zero_grad()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets) #*1/16
            
            loss.backward()
            # if batch_idx % 16 == 15:
            optimizer_base.step()
            optimizer_bn.step()
            optimizer_base.zero_grad()
            optimizer_bn.zero_grad()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            printProgressBar(batch_idx, len(trainloader)-1, 'training\t', 'Data: %.3f | Batch: %.3f | Loss: %.3f | Top1: %.3f%% | Top5: %.3f%%'
                % (data_time.val, batch_time.val, losses.avg, top1.avg, top5.avg))
            # Timer.show_time()
        
        save_train_status([epoch+1, losses.avg, top1.avg, top5.avg], folder_name)

    def test(epoch):
        global best_acc
        model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                # measure data loading time
                data_time.update(time.time() - end)

                inputs, targets = inputs.to(device), targets.to(device)

                # outputs, scale = model(inputs)
                outputs = model(inputs)
                # loss = criterion(outputs, scale, targets)
                loss = criterion(outputs, targets)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                printProgressBar(batch_idx, len(testloader)-1, 'test\t', 'Data: %.3f | Batch: %.3f | Loss: %.3f | Top1: %.3f%% | Top5: %.3f%%'
                    % (data_time.val, batch_time.val, losses.avg, top1.avg, top5.avg))

        save_test_status([epoch+1, losses.avg, top1.avg, top5.avg], folder_name)

        acc = top1.avg
        print('Saving..')
        state = {
            'net':model.state_dict(),
            'optimizer_base': optimizer_base.state_dict(),
            'optimizer_bn': optimizer_bn.state_dict(),
            'scheduler_base': scheduler_base.state_dict(),
            'scheduler_bn': scheduler_bn.state_dict(),
            'acc':acc,
            'epoch':epoch,
        }
        save_checkpoint(state, folder_name, epoch+1)
        if acc > best_acc:
            best_acc = acc

    # state = load_checkpoint(folder_name, device, 25)
    # lp.load_state_dict(model, state['net'])
    # optimizer_base.load_state_dict(state['optimizer_base'])
    # optimizer_bn.load_state_dict(state['optimizer_bn'])
    # # scheduler.load_state_dict(state['scheduler'])
    # for i in range(25):
    #     scheduler_base.step()
    #     scheduler_bn.step()
    # best_acc = state['acc']
    # start_epoch = state['epoch']+1

    start_epoch = 0
    best_acc = 0
    end_epoch = 90

    set_scale()
    for epoch in range(start_epoch, end_epoch):
        train(epoch, end_epoch)
        test(epoch)
        scheduler_base.step()
        scheduler_bn.step()


if __name__ == "__main__":
    run()
