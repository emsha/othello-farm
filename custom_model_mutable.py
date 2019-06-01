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

class CustomModelMutable():

    def __init__(self, build_info, image_s, CUDA=False):


        # image_s is a side of the square image

        previous_channels = 3
        self.model = nn.Sequential()
        self.image_size = image_s
        self.build_info = build_info
        # print(build_info)
        for i, conv_layer_info in enumerate(build_info['conv_layers']):
            i = str(i)
            # print("nfilters class is:{}".format(conv_layer_info['n_filters']['val'].__class__))
            # print("prev channels class is:{}".format(previous_channels.__class__))
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
            # print("\n\n\n", conv_layer_info, '\n\n\n')
            previous_channels = conv_layer_info['n_filters']['val']


        previous_units = compute_initial_fc_inputs(build_info, image_s)
        print('previous_units: {}'.format(previous_units))
        self.model.add_module('flatten', Flatten()) 
        # print(previous_units)

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
        self.criterion = nn.CrossEntropyLoss()
        # print('OPTIMIZER: {}'.format(optimizer))
        self.cuda = False
        if CUDA:
            self.model.cuda()
            self.cuda = True
    def clone(self):
        ''' 
            currently returns a new instance of CustomModelMutable 
            with identical weights and shape 
            except for the first conv layer,
            which has an added filter initialized to all ones
        '''
        
        # initialize build info and new net
        num_added_filters = 1
        new_build_info = self.build_info
        new_build_info['conv_layers'][0]['n_filters']['val'] += num_added_filters
        new_net = CustomModelMutable(new_build_info, self.image_size)

        # generate new filter(s) and add to new net
        new_size=self.model.conv_0.weight.size()
        filter_size = new_net.model.conv_0.kernel_size
        s = filter_size[0]
        new_filter = torch.zeros(filter_size)
        previous_channels = 3
        layer_i = 0
        if layer_i != 0:
            previous_channels = new_build_info['conv_layers'][layer_i-1]['n_filters']['val']
        new_weights = torch.cat((self.model.conv_0.weight.data, new_filter.expand(num_added_filters, previous_channels, s, s)))
        new_net.model.conv_0.weight.data = new_weights

        # update new layers to match old network
        i_to_mutate = 0
        is_last_conv_layer = (i_to_mutate==len(new_build_info['conv_layers'])-1)
        init_fc_inputs = compute_initial_fc_inputs(new_build_info, self.image_size)
        
        # update new layers to match old net
        for i, layer in enumerate(new_net.model):
            # print(i, layer)
            if i != i_to_mutate:
                try:
                    new_net.model[i].weight.data = self.model[i].weight.data
                    # print(i)
                except(AttributeError):
                    pass
        
        # correct new first fc layer if we're mutating the last conv layer, weights are all 1's
        if is_last_conv_layer:
            new_fc0_weights = new_net.model.fc_0.weight.data
            old_size = tuple(new_fc0_weights.size())
            target_size = (new_net.model.fc_0.out_features, new_net.model.fc_0.in_features)  
            n_to_add = target_size[1] - old_size[1]
            ones = torch.zeros(new_fc0_weights.size()[0], n_to_add)
            new_fc0_weights = torch.cat((new_fc0_weights, ones), dim=1)
            new_net.model.fc_0.weight.data = new_fc0_weights

        return new_net

def compute_initial_fc_inputs(build_info, image_s):
    '''returns number of input layers for first fc layer given build info and image size (one side of square img)'''
    kernels = [l['kernel_size']['val'] for l in build_info['conv_layers']]
    out_s = image_s #+2 #+2 for padding on each side
    # print('out_s: {}'.format(out_s))
    for k in kernels:
        print('out_s: {}'.format(out_s))
        out_s += 2
        out_s = (out_s-k+1)
    print('out_s: {}'.format(out_s))
    last_n_filters = build_info['conv_layers'][-1]['n_filters']['val']
    print('FINAL: ', out_s, last_n_filters)
    return (last_n_filters) * (out_s**2)
    



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