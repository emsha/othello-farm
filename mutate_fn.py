# mutate_fn.py
# from https://medium.com/@stathis/design-by-evolution-393e41863f98

def mutate_net(net):
	'''mutate a network'''
	global NET_SPACE, FC_SPACE, CONV_SPACE

	# mutate optimizer
	for k in ['lr', 'weight_decay', 'optimizer']:

		if random.random() < NET_SPACE[k][-1]:
			net[k] = random_value(NET_SPACE[k])

	# mutate layers
	for layer in net['conv_layers']:
		for k in CONV_SPACE.keys():
			if random.random() < CONV_SPACE[k][-1]:
				layer[k] = random_value(CONV_SPACE[k])

	for layer in net['fc_layers']:
		for k in FC_SPACE.keys():
			if random.random() < FC_SPACE[k][-1]:
				layer[k] = random_value(FC_SPACE[k])

	# mutate number of layers -- RANDOMLY ADD
	# conv
	if random.random() < NET_SPACE['n_conv_layers'][-1]:
		if net['n_conv_layers']['val'] < NET_SPACE['n_conv_layers'][1]:
			if random.random() < 0.5:
				layer = dict()
				for k in CONV_SPACE.keys():
					layer[k] = random_value(CONV_SPACE[k])
				net['conv_layers'].append(layer)
				# value & id update
				net['n_conv_layers']['val'] = len(net['conv_layers'])
				net['n_conv_layers']['id'] += 1
			else:
				if net['n_conv_layers']['val'] > 1:
					net['conv_layers'].pop()
					net['n_conv_layers']['val'] = len(net['conv_layers'])
					net['n_conv_layers']['id'] -= 1
	# fc
	if random.random() < NET_SPACE['n_fc_layers'][-1]:
		if net['n_fc_layers']['val'] < NET_SPACE['n_fc_layers'][1]:
			if random.random() < 0.5:
				layer = dict()
				for k in FC_SPACE.keys():
					layer[k] = random_value(FC_SPACE[k])
				net['fc_layers'].append(layer)
				# value & id update
				net['n_fc_layers']['val'] = len(net['fc_layers'])
				net['n_fc_layers']['id'] += 1
			else:
				if net['n_fc_layers']['val'] > 1:
					net['fc_layers'].pop()
					net['n_fc_layers']['val'] = len(net['fc_layers'])
					net['n_fc_layers']['id'] -= 1

	return net