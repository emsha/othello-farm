# randomize_fn.py
# from https://medium.com/@stathis/design-by-evolution-393e41863f98
import layerspace as ls
import random

def random_value(space):
    '''sample random value from given space.'''
    val=None
    if space[2] == 'int':
        val = random.randint(space[0], space[1])
    if space[2] == 'list':
        val = random.sample(space[1], 1)[0]
    if space[2] == 'float':
        val = ((space[1] - space[0]) * random.random()) + space[0]
    return {'val': val, 'id': random.randint(0, 2**10)}


def randomize_network(bounded=True):
    '''create a random network.'''
    # global NET_SPACE, LAYER_SPACE
    net = dict()
    for k in ls.NET_SPACE.keys():
        net[k] = random_value(ls.NET_SPACE[k])

    if bounded:
        net['n_conv_layers']['val'] = min(net['n_conv_layers']['val'], 1)
        net['n_fc_layers']['val'] = min(net['n_fc_layers']['val'], 1)

    conv_layers = []
    for i in range(net['n_conv_layers']['val']):
        layer = dict()
        for k in ls.CONV_SPACE.keys():
            layer[k] = random_value(ls.CONV_SPACE[k])
        conv_layers.append(layer)
    net['conv_layers'] = conv_layers

    fc_layers = []
    for i in range(net['n_fc_layers']['val']):
        layer = dict()
        for k in ls.FC_SPACE.keys():
            layer[k] = random_value(ls.FC_SPACE[k])
        fc_layers.append(layer)
    net['fc_layers'] = fc_layers

    return net

    