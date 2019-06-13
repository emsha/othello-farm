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
import copy

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
        # print('previous_units: {}'.format(previous_units))
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

    def clone_with_added_filter(self, n):
        '''
            return clone net with added filter at conv layer n
        '''
        # print("ADD TO CONV_{}".format(n))
        l_cur_old = None
        l_cur = None
        l_next = None
        l_next_old = None
        l_cur_i = None
        conv_list_old = [(n, m) for (n, m) in self.model.named_modules() if m.__class__ == torch.nn.modules.conv.Conv2d] 
        # for name, layer in conv_list_old:
            # print(name, layer.weight.data.size(), layer)
        if not 0 <= n < len(conv_list_old):
            raise Exception('net.clone_with_added_filter(n): n is not between 0 and num conv layers')
        # init new net
        n_filters = 1
        new_build_info = copy.deepcopy(self.build_info)
        new_build_info['conv_layers'][n]['n_filters']['val'] += n_filters
        new_net = CustomModelMutable(new_build_info, self.image_size)

        # get layers to change
        conv_list = [(n, m) for (n, m) in new_net.model.named_modules() if m.__class__ == torch.nn.modules.conv.Conv2d]        
        
        # print('old:')
        # for l in self.build_info.get('conv_layers'):
            # print(l)

        # print("new:")
        # for l in new_build_info.get('conv_layers'):
        #     print(l)

        
        l_cur_old = conv_list_old[n][1]
        l_cur = conv_list[n][1]
        if n != len(conv_list_old) - 1:
            l_next_old = conv_list_old[n+1][1]
            l_next = conv_list[n+1][1]

        # print(l_cur)
        # print(l_next)

        # '''
        # generate new filter(s) and add to new net
        # new_size=self.model.conv_0.weight.size()
        filter_size = l_cur.kernel_size
        s = filter_size[0]
        new_filter = torch.zeros(filter_size)
        previous_channels = 3
        if n != 0:
            previous_channels = l_cur.in_channels
        # print('prev channels: {}'.format(previous_channels))
        # print(self.model)
        # print(n)
        # print(new_net.model)
        # print(l_cur_old)
        # print(l_cur_old.weight.data.size())
        # print(n_filters, previous_channels)
        # print(new_filter.expand(n_filters, previous_channels, s, s).size())
        # print('\ninside clone\n')
        # print('old weight size', l_cur_old.weight.data.size())
        # print('new filter size', new_filter.size())
        # print('num to add', n_filters)
        # print('prev channels', previous_channels)
        # print(s)
        # print('----------')
        new_weights = torch.cat((l_cur_old.weight.data, new_filter.expand(n_filters, previous_channels, s, s)))
        
        # update new layers to match old network
        # is_last_conv_layer = (n==len(new_build_info['conv_layers'])-1)
        init_fc_inputs = compute_initial_fc_inputs(new_build_info, self.image_size)
        # update new layers to match old net
        for i, layer in enumerate(new_net.model):
        
            # print(i, layer)
            # if i != l_cur_i:
            try:
                new_net.model[i].weight.data = self.model[i].weight.data
                # print(i)
            except(AttributeError):
                # print('aterror)')
                pass
        l_cur.weight.data = new_weights
        # correct new first fc layer if we're mutating the last conv layer
        if not l_next:
            # print('changing last conv layer___________')
            new_fc0_weights = new_net.model.fc_0.weight.data
            old_size = tuple(new_fc0_weights.size())
            target_size = (new_net.model.fc_0.out_features, new_net.model.fc_0.in_features)  
            n_to_add = target_size[1] - old_size[1]
            ones = torch.zeros(new_fc0_weights.size()[0], n_to_add)
            new_fc0_weights = torch.cat((new_fc0_weights, ones), dim=1)
            new_net.model.fc_0.weight.data = new_fc0_weights
        else:
            next_conv_data = l_next.weight.data
            # print("old size:
            # {}".format(new_net.model.conv_1.weight.data.size()))
            new_ch_in = l_cur.out_channels
            new_fltr_s = l_next.kernel_size
            new_ch_out = l_next.out_channels
            s = new_fltr_s[0]
            new_filter = torch.zeros(1)

            new_filter = new_filter.expand(new_ch_out, 1, s, s)
            new_weights = torch.cat((l_next_old.weight.data, new_filter), 1)

            l_next.weight.data = new_weights
        
        # print(new_net.model)
        # print([l for l in new_net.build_info['fc_layers']])
        # print(new_net.model.fc_0.weight.data.size())
        return new_net

    def clone(self, add_conv_layer=False):
        ''' 
            TODO: UPDATE DATA
            currently returns a new instance of CustomModelMutable 
            with identical weights and shape 
            except for the first conv layer,
            which has an added filter initialized to all ones
        '''
        print('\n--------CLONING-------\n')
        conv_list = []
        for i, m in enumerate(self.model.modules()):
            if m.__class__ == torch.nn.modules.conv.Conv2d:
                conv_list.append(m)
        print(self.model)
        name_to_mutate = None
        name_after_to_mutate = None
        named_conv_list = [(n, m) for (n, m) in self.model.named_modules() if m.__class__ == torch.nn.modules.conv.Conv2d]
        conv_i_to_mutate = random.choice(list(range(len(named_conv_list))))
        name_to_mutate = named_conv_list[conv_i_to_mutate][0]
        is_last_conv_layer = False
        # print(named_conv_list)
        try:
            name_after_to_mutate = named_conv_list[conv_i_to_mutate+1][0]
        except:
            pass

        print(name_to_mutate)
        print(name_after_to_mutate)
        if not name_after_to_mutate:
            print('mutating final conv layer')
        


                #, m.__class__)
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
        else:
            next_conv_data = new_net.model.conv_1.weight.data
            # print("old size: {}".format(new_net.model.conv_1.weight.data.size()))
            new_ch_in = new_net.model.conv_0.out_channels
            new_fltr_s = new_net.model.conv_1.kernel_size
            new_ch_out = new_net.model.conv_1.out_channels
            s = new_fltr_s[0]
            new_filter = torch.zeros(1)

            new_filter = new_filter.expand(new_ch_out, 1, s, s)
            new_weights = torch.cat((self.model.conv_1.weight.data, new_filter), 1)

            new_net.model.conv_1.weight.data = new_weights
            # print("new size: {}".format(new_net.model.conv_1.weight.data.size()))


            # print(old_size, target_size)

            '''
                TODO: when changing conv layer, we've accounted for it being the last conv layer
                    now we have to account for there being other conv layer after it
                    update channels of next layer
                    need to do more init of filters
                    eventually should put this change conv layer thing into a function

            '''
        print('\n--------DONE CLONING-------\n')
        return new_net

def compute_initial_fc_inputs(build_info, image_s):
    '''returns number of input layers for first fc layer given build info and image size (one side of square img)'''
    kernels = [l['kernel_size']['val'] for l in build_info['conv_layers']]
    out_s = image_s #+2 #+2 for padding on each side
    # print('out_s: {}'.format(out_s))
    for k in kernels:
        # print('out_s: {}'.format(out_s))
        out_s += 2
        out_s = (out_s-k+1)
    # print('out_s: {}'.format(out_s))
    last_n_filters = build_info['conv_layers'][-1]['n_filters']['val']
    # print('FINAL: ', out_s, last_n_filters)
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