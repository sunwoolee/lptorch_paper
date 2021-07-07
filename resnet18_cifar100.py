from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from model.resnet_cifar import ResNet18

from utils import *
from math import ceil
import lptorch as lp

best_acc = 0
def run():
    global best_acc
    # Data
    print('==> Preparing data..')
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # fp8_format = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,1,0]
    # fp8_format = [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,2,1,0]
    fp8_format = [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1,1,1,1,0]
    # log_format = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
    # log_format = [1,1,1,1,1,1,1,0]
    log_format = [1,1,1,0]
    lp.set_activ_quant(lp.quant.quant(lp.quant.custom_fp_format(fp8_format), room=1))
    lp.set_error_quant(lp.quant.quant(lp.quant.custom_fp_format(fp8_format), room=1))
    # lp.set_weight_quant(lp.quant.quant(lp.quant.custom_fp_format(fp8_format), room=0))
    lp.set_weight_quant(lp.quant.quant(lp.quant.custom_fp_format(log_format), room=0, ch_wise=True))
    # lp.set_weight_quant(lp.quant.quant(lp.quant.linear_format(3), room=0, ch_wise=True))
    lp.set_grad_quant(lp.quant.quant(lp.quant.custom_fp_format(fp8_format), room=2))
    lp.set_master_quant(lp.quant.quant(lp.quant.fp_format(exp_bit=6, man_bit=9), stochastic=True))
    lp.set_scale_fluctuate(False)
    lp.set_hysteresis_update(False)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    device = torch.device("cuda")
    model = ResNet18(100).to(device)

    bn_param = []
    base_param = []
    firstlast_param = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_param += list(m.parameters())
    base_param = list(set(model.parameters()) - set(bn_param))

    # bn_param = []
    # firstlast_param = []
    # base_param = []
    # set_first_conv = False
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):# or isinstance(m, nn.Conv2d):
    #         bn_param += list(m.parameters())
    #     elif isinstance(m, nn.Linear):
    #         firstlast_param += list(m.parameters())
    #     elif isinstance(m, nn.Conv2d):
    #         if set_first_conv:
    #             base_param += list(m.parameters())
    #         else:
    #             firstlast_param += list(m.parameters())
    #             set_first_conv = True

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer_base = lp.optim.SGD(base_param, lr=0.1, momentum=0.9, weight_decay=5e-4)
    # optimizer_fl = lp.optim.SGD(firstlast_param, lr=0.1, momentum=0.9, weight_decay=4e-5)
    # optimizer_fl = lp.optim.SGD(firstlast_param, lr=0.1, momentum=0.9, weight_decay=4e-5, quant=lp.quant.quant(lp.quant.custom_fp_format(fp8_format), room=0))
    optimizer_bn = lp.optim.SGD(bn_param, lr=0.1, momentum=0.9, weight_decay=5e-4, weight_quantize=False)

    lr_epoch = [60, 120, 160]
    scheduler_base = optim.lr_scheduler.MultiStepLR(optimizer_base, milestones=lr_epoch, gamma=0.2)
    # scheduler_fl = optim.lr_scheduler.MultiStepLR(optimizer_fl, milestones=lr_epoch, gamma=0.2)
    scheduler_bn = optim.lr_scheduler.MultiStepLR(optimizer_bn, milestones=lr_epoch, gamma=0.2)

    # folder_name = 'cifar100_resnet18_baseline'
    # folder_name = 'cifar100_resnet18_activ_FP152'
    # folder_name = 'cifar100_resnet18_activ_FP143'
    # folder_name = 'cifar100_resnet18_activ_LFP8'
    # folder_name = 'cifar100_resnet18_train_LFP8_w_hysteresis'
    # folder_name = 'cifar100_resnet18_train_LFP8_wo_hysteresis'
    # folder_name = 'cifar100_resnet18_train_FP4_w_hysteresis'
    # folder_name = 'cifar100_resnet18_train_FP4_wo_hysteresis'
    # folder_name = 'cifar100_resnet18_train_INT4_w_hysteresis'
    # folder_name = 'cifar100_resnet18_train_INT4_wo_hysteresis'
    # folder_name = 'cifar100_resnet18_train_INT6_w_hysteresis'
    # folder_name = 'cifar100_resnet18_train_INT6_wo_hysteresis'
    # folder_name = 'cifar100_resnet18_train_INT3_w_hysteresis'
    # folder_name = 'cifar100_resnet18_train_INT3_wo_hysteresis'
    # folder_name = 'cifar100_resnet18_train_FP3_w_hysteresis'
    folder_name = 'cifar100_resnet18_train_FP3_wo_hysteresis'   

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
            loss = criterion(outputs, targets)
            
            loss.backward()

    def train(epoch):
        print('\nEpoch: %d \t LR: %.3f' % (epoch+1, scheduler_base.get_last_lr()[0]))
        model.train()

        train_loss = 0
        correct = 0
        total = 0

        loss_data = [0]*5
        correct_data = [0]*5

        optimizer_base.zero_grad()
        # optimizer_fl.zero_grad()
        optimizer_bn.zero_grad()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            index = int(batch_idx/78)
            if index is 5:
                index = 4
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_base.step()
            # optimizer_fl.step()
            optimizer_bn.step()
            optimizer_base.zero_grad()
            # optimizer_fl.zero_grad()
            optimizer_bn.zero_grad()

            train_loss += loss.item()
            loss_data[index] += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            correct_data[index] += predicted.eq(targets).sum().item()

            printProgressBar(batch_idx, len(trainloader)-1, str(epoch)+'th epoch training ', 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        for i in range(4):
            loss_data[i] /= 78
            correct_data[i] /= 99.84
        loss_data[4] /= 79
        correct_data[4] /= 100.64
        
        save_train_status([loss_data, correct_data], folder_name)

    def test(epoch):
        global best_acc
        model.eval()

        test_loss = 0
        correct = 0
        total = 0
        acc = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                printProgressBar(batch_idx, len(testloader)-1, str(epoch)+'th epoch test ', 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        save_test_status([test_loss/100, correct/100], folder_name)

        acc = 100. * correct/total
        print('Saving..')
        state = {
            'net':model.state_dict(),
            'optimizer_base': optimizer_base.state_dict(),
            # 'optimizer_fl': optimizer_fl.state_dict(),
            'optimizer_bn': optimizer_bn.state_dict(),
            'scheduler_base': scheduler_base.state_dict(),
            # 'scheduler_fl': scheduler_fl.state_dict(),
            'scheduler_bn': scheduler_bn.state_dict(),
            'acc':acc,
            'epoch':epoch,
        }
        save_checkpoint(state, folder_name, epoch+1)
        if acc > best_acc:
            best_acc = acc

    # state = load_checkpoint(folder_name, device)
    # model.load_state_dict(state['net'])
    # optimizer.load_state_dict(state['optimizer'])
    # scheduler.load_state_dict(state['scheduler'])
    # best_acc = state['acc']
    # start_epoch = state['epoch']+1
    
    start_epoch = 0
    best_acc = 0
    end_epoch = 200

    set_scale()
    for epoch in range(start_epoch, end_epoch):
        train(epoch)
        test(epoch)
        scheduler_base.step()
        # scheduler_fl.step()
        scheduler_bn.step()

if __name__ == "__main__":
    run()