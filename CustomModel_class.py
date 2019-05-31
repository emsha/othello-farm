# CustomModel_class.py
# from https://medium.com/@stathis/design-by-evolution-393e41863f98
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from stateB import State
import torch.nn.functional as F
import torch.optim as optim
import random

class CustomModel():

    def __init__(self, build_info, image_s, CUDA=True):


        # image_s is a side of the square image

        # TODO: FIGURE OUT HOW TO COMPUTE PREVIOUS _ UNITS! FROM CONV LAHYERS
        # previous_units = 100
        previous_channels = 3
        self.model = nn.Sequential()
        
        # kernels = []

        for i, conv_layer_info in enumerate(build_info['conv_layers']):
            i = str(i)
            self.model.add_module(
                'conv_' + i,
                nn.Conv2d(previous_channels, conv_layer_info['n_filters']['val'], kernel_size=conv_layer_info['kernel_size']['val'], stride=1, padding=1)
                )
            # kernels.append(conv_layer_info['kernel_size']['val'])
            if conv_layer_info['activation']['val'] == 'tanh':
                self.model.add_module(
                    'tanh_'+i,
                    nn.Tanh()
                )
            if conv_layer_info['activation']['val'] == 'relu':
                self.model.add_module(
                    'relu_'+i,
                    nn.ReLU()
                )
            if conv_layer_info['activation']['val'] == 'sigmoid':
                self.model.add_module(
                    'sigm_'+i,
                    nn.Sigmoid()
                )
            if conv_layer_info['activation']['val'] == 'elu':
                self.model.add_module(
                    'elu_'+i,
                    nn.ELU()
                )
            previous_channels = conv_layer_info['n_filters']


        # compute first fc layer n_inputs
        # print(kernels)
        # out_s = image_s
        # for k in kernels:
        #     out_s = compute_dim(out_s, k)
        #     print(out_s)
        
        # last_n_filters = build_info['conv_layers'][-1]['n_filters']['val']
        # print('FINAL: ', out_s, last_n_filters)

        # self.model.add_module('print', Print())
        previous_units = compute_initial_fc_inputs(build_info, image_s)
        self.model.add_module('flatten', Flatten()) 
        print(previous_units)

        # init fc layers
        for i, fc_layer_info in enumerate(build_info['fc_layers']):
            i = str(i)
            self.model.add_module(
                'fc_' + i,
                nn.Linear(previous_units, fc_layer_info['n_units']['val'])
                )
            self.model.add_module(
                'dropout_' + i,
                nn.Dropout(p=fc_layer_info['dropout_rate']['val'])
                )
            if fc_layer_info['activation']['val'] == 'tanh':
                self.model.add_module(
                    'tanh_'+i,
                    nn.Tanh()
                )
            if fc_layer_info['activation']['val'] == 'relu':
                self.model.add_module(
                    'relu_'+i,
                    nn.ReLU()
                )
            if fc_layer_info['activation']['val'] == 'sigmoid':
                self.model.add_module(
                    'sigm_'+i,
                    nn.Sigmoid()
                )
            if fc_layer_info['activation']['val'] == 'elu':
                self.model.add_module(
                    'elu_'+i,
                    nn.ELU()
                )
            previous_units = fc_layer_info['n_units']['val']



        # clafication layer
        self.model.add_module(
            'classification_layer',
            nn.Linear(previous_units, 10)
            )
        self.model.add_module('sofmax', nn.LogSoftmax())
        self.model.cpu()
        
        if build_info['optimizer']['val'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(),
                                lr=build_info['weight_decay']['val'],
                                weight_decay=build_info['weight_decay']['val'])

        elif build_info['optimizer']['val'] == 'adadelta':
            optimizer = optim.Adadelta(self.model.parameters(),
                                    lr=build_info['weight_decay']['val'],
                                    weight_decay=build_info['weight_decay']['val'])

        elif build_info['optimizer']['val'] == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(),
                                    lr=build_info['weight_decay']['val'],
                                    weight_decay=build_info['weight_decay']['val'])
        else:
            optimizer = optim.SGD(self.model.parameters(),
                                lr=build_info['weight_decay']['val'],
                                weight_decay=build_info['weight_decay']['val'],
                                momentum=0.9)
        self.optimizer = optimizer
        self.cuda = False
        if CUDA:
            self.model.cuda()
            self.cuda = True

def compute_initial_fc_inputs(build_info, image_s):
    '''returns number of input layers for first fc layer given build info and image size (one side of square img)'''
    kernels = [l['kernel_size']['val'] for l in build_info['conv_layers']]
    out_s = image_s
    for k in kernels:
        out_s = (out_s-k+1)^2
    last_n_filters = build_info['conv_layers'][-1]['n_filters']['val']
    print('FINAL: ', out_s, last_n_filters)

    return (last_n_filters) * (out_s**2)
    

def compute_dim(orig_s, kernel_s):
    return (orig_s-kernel_s+1)^2

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    
    if type(stride) is not tuple:
        stride = (stride, stride)
    
    if type(pad) is not tuple:
        pad = (pad, pad)
    
    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
    
    return h, w

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x