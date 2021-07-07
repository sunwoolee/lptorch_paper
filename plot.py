# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 11:30:52 2021

@author: User
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

global dataset
global model
global format_list

def get_file_list(target, activ_test_list, format_list):
    curdir = os.path.abspath(os.curdir)
    checkdir = os.path.abspath(os.path.join(curdir, 'checkpoint'))
    os.chdir(checkdir)
    file_list = os.listdir()
    os.chdir(curdir)
    target_file_list = []
    for file in file_list:
        if target in file:
            target_file_list.append(file)
    target_baseline = []
    target_activ_list = []
    target_format_list = []
    for file in target_file_list:
        if 'baseline' in file:
            target_baseline.append(file)
        for name in activ_test_list:
            if name in file:
                target_activ_list.append(file)
        if 'train' in file:
            for name in format_list:
                if name in file:
                    target_format_list.append(file)
    return target_baseline, target_activ_list, target_format_list            

class unit():
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.label = folder_name.replace(dataset+'_'+model+'_', '')
        train_epoch, train_loss, train_acc = self.get_train_status()
        test_epoch, test_loss, test_acc = self.get_test_status()
        self.epoch = {'train':train_epoch, 'test':test_epoch}
        self.loss = {'train':train_loss, 'test':test_loss}
        self.acc = {'train':train_acc, 'test':test_acc}
        my_form = ''
        for form in format_list:
            if form in folder_name:
                my_form = form
                break
        my_hysteresis = ''
        if 'w_hysteresis' in folder_name:
            my_hysteresis = 'w/ hysteresis'
        elif 'wo_hysteresis' in folder_name:
            my_hysteresis = 'w/o hysteresis'
        self.df = pd.DataFrame({'Dataset':[dataset], 'Model':[model], 'train loss':[self.loss['train'].min()],
                                'train acc':[self.acc['train'].max()], 'test loss':[self.loss['test'].min()],
                                'test acc':[self.acc['test'].max()], 'format':[my_form], 'hysteresis':[my_hysteresis]})
        
    def get_test_status(self):
        curdir = os.path.abspath(os.curdir)
        os.chdir(os.path.join(os.path.join(curdir, 'checkpoint'), self.folder_name))
        try:
            with open('test_status.txt', 'r') as f:
                loss_list = []
                acc_list = []
                i = 0
                for line in f:
                    vals = line.split(' ')
                    if len(vals) is 3:
                        loss_idx = 0
                        acc_idx = 1
                    elif len(vals) is 5:
                        loss_idx = 1
                        acc_idx = 2
                    loss_list.append(float(vals[loss_idx]))
                    acc_list.append(float(vals[acc_idx]))
                    i += 1
                return i, np.array(loss_list), np.array(acc_list)
        finally:
            os.chdir(curdir)
            
    def get_train_status(self):
        curdir = os.path.abspath(os.curdir)
        os.chdir(os.path.join(os.path.join(curdir, 'checkpoint'), self.folder_name))
        try:
            with open('train_status.txt', 'r') as f:
                loss_list = []
                acc_list = []
                i = 0
                for line in f:
                    if '['in line:
                        vals = line.replace(']', '').replace('[','').replace(',','').split(' ')
                    else:
                        vals = line.split(' ')
                    if len(vals) is 5:
                        loss_idx = 1
                        acc_idx = 2
                    elif len(vals) is 11:
                        loss = np.array([float(x) for x in vals[0:5]])
                        acc = np.array([float(x) for x in vals[5:10]])
                        weight = np.array([0.9984, 0.9984, 0.9984, 0.9984, 1.0064])
                        loss = loss * weight
                        acc = acc * weight
                        loss = loss.sum()/5
                        acc = acc.sum()/5
                        vals = [loss, acc]
                        loss_idx = 0
                        acc_idx = 1
                    loss_list.append(float(vals[loss_idx]))
                    acc_list.append(float(vals[acc_idx]))
                    i += 1
                return i, np.array(loss_list), np.array(acc_list)
        finally:
            os.chdir(curdir)

class unit_group():
    def __init__(self):
        self.groups = {}
        
    def add_unit(self, group_name, units):
        if group_name in self.groups.keys():
            self.groups[group_name] = self.groups[group_name] + units
        else:
            self.groups[group_name] = units
            
    def plot_line(self, ran, datas, labels, group_name, mode):
        ran = np.arange(ran)
        tu = ()
        for data in datas:
            tu = tu + (ran, data)
        sns.lineplot(data=datas, palette='deep', dashes=False)
        plt.legend(labels=labels)
        plt.title(dataset+' '+model+' '+group_name+' '+mode+' plot')
        plt.xlabel('Epoch')
        if 'loss' in mode:
            plt.ylabel('loss')
        else:
            plt.ylabel('Top 1 Acc (%)')
    
    def plot_loss_line(self, group_name, mode):
        ''' mode : 'train' or 'test' '''
        units = self.groups[group_name]
        epochs = [u.epoch[mode] for u in units]
        epochs = np.array(epochs).max()
        losses = [u.loss[mode] for u in units]
        labels = [u.label for u in units]
        self.plot_line(epochs, losses, labels, group_name, mode+' loss')
        
    def plot_acc_line(self, group_name, mode):
        ''' mode : 'train' or 'test' '''
        units = self.groups[group_name]
        epochs = [u.epoch[mode] for u in units]
        epochs = np.array(epochs).max()
        acces = [u.acc[mode] for u in units]
        labels = [u.label for u in units]
        self.plot_line(epochs, acces, labels, group_name, mode+' acc')
        
    def plot_bar(self, datas, labels, mode):
        sns.barplot(x=datas, y=labels, palette='YlGnBu')
        plt.title(dataset+' '+model+' '+mode)
        if 'loss' in mode:
            plt.xlabel('loss')
        else:
            plt.xlabel('Top 1 Acc (%)')
        
    def plot_loss_bar(self, group_name, mode):
        units = self.groups[group_name]
        losses = [u.loss[mode].min() for u in units]
        labels = [u.label for u in units]
        self.plot_bar(losses, labels, mode+' loss')
        
    def plot_acc_bar(self, group_name, mode):
        units = self.groups[group_name]
        acces = [u.acc[mode].max() for u in units]
        labels = [u.label for u in units]
        self.plot_bar(acces, labels, mode+' acc')
        
    def plot_bar_compare(self, dfs, mode):
        sns.barplot(x='format', y=mode, hue='hysteresis', data=dfs, palette='YlGnBu')
        plt.title(dataset+' '+model+' '+mode)
    
    def plot_loss_bar_compare(self, group_name, mode):
        units = self.groups[group_name]
        dfs = [u.df for u in units]
        dfs = pd.concat(dfs)
        self.plot_bar_compare(dfs, mode+' loss')
        
    def plot_acc_bar_compare(self, group_name, mode):
        units = self.groups[group_name]
        dfs = [u.df for u in units]
        dfs = pd.concat(dfs)
        self.plot_bar_compare(dfs, mode+' acc')
        
    
        
# %%
sns.set(rc={'figure.figsize':(11.7, 8.27)})
dataset = 'imagenet'
model = 'resnet18'
activ_test_list = ['activ_FP152', 'activ_FP143', 'activ_LFP8']
format_list = ['LFP8', 'FP4', 'FP3', 'INT8', 'INT6', 'INT4', 'INT3']
baseline, activ_test, hysteresis_test = get_file_list(dataset+'_'+model, activ_test_list, format_list)
my_unit_group = unit_group()
#my_units = [unit(n) for n in baseline] + [unit(n) for n in activ_test] + [unit(n) for n in hysteresis_test]
my_units = [unit(n) for n in hysteresis_test]
my_unit_group.add_unit('activ_test', my_units)
#my_unit_group.plot_loss_bar('activ_test', 'test')
my_unit_group.plot_acc_bar_compare('activ_test', 'train')