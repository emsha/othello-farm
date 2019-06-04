# layerspace.py
# max shashoua
# definition of a space
# lower bound - upper bound, type param, mutation rate
# adapted from https://medium.com/@stathis/design-by-evolution-393e41863f98

CONV_SPACE = dict()
CONV_SPACE['n_filters'] = (3, 4, 'int', 0.05)
CONV_SPACE['kernel_size'] = (2, 3, 'int', 0.05) # can't be bigger than image plus padding
CONV_SPACE['activation'] =\
    (0,  ['linear', 'tanh', 'relu', 'sigmoid', 'elu'], 'list', 0.2)

FC_SPACE = dict()
FC_SPACE['n_units'] = (128, 1024, 'int', 0.15)
FC_SPACE['dropout_rate'] = (0.0, 0.7, 'float', 0.2)
FC_SPACE['activation'] =\
    (0,  ['linear', 'tanh', 'relu', 'sigmoid', 'elu'], 'list', 0.2)

NET_SPACE = dict()
NET_SPACE['n_conv_layers'] = (3, 5, 'int', 0.15)
NET_SPACE['n_fc_layers'] = (1, 1, 'int', 0.15)
NET_SPACE['lr'] = (0.0001, 0.1, 'float', 0.15)
NET_SPACE['weight_decay'] = (0.00001, 0.0004, 'float', 0.2)
NET_SPACE['optimizer'] =\
    (0, ['sgd', 'adam', 'adadelta', 'rmsprop'], 'list', 0.2)

MUTATION_SPACE = dict()
MUTATION_SPACE['point_mutation_rate'] = 0.01
MUTATION_SPACE['add_conv_filter_rate'] = 0.1
MUTATION_SPACE['add_conv_layer_rate'] = 0.1
MUTATION_SPACE['add_fc_node_rate'] = 0.1
MUTATION_SPACE['add_fc_layer_rate'] = 0.1