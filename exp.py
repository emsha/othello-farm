# exp.py
from __future__ import absolute_import
import layerspace
import randomize_fn as rfn
import mutate_fn as mfn
import CustomModel_class as c
import custom_model_mutable as cmm
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import random
import copy


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train_net(net, epochs):
	print("    training a net")
	for epoch in range(epochs):  # loop over the dataset multiple times
	    running_loss = 0.0
	    for i, data in enumerate(random.sample(list(trainloader), 1000), 0):
	        # get the inputs
	        inputs, labels = data

	        # zero the parameter gradients
	        net.optimizer.zero_grad()

	        # forward + backward + optimize
	        outputs = net.model(inputs)
	        loss = net.criterion(outputs, labels)
	        loss.backward()
	        net.optimizer.step()

	        # print statistics
	        running_loss += loss.item()
	        if i % 500 == 499:    # print every 2000 mini-batches
	            print('[%d, %5d] loss: %.3f' %
	                  (epoch + 1, i + 1, running_loss / 500))
	            running_loss = 0.0

def train_nets(nets, epochs):
	for net in nets:
		train_net(net, epochs)

def test_nets(nets):
	res = []
	for net in nets:

		correct = 0
		total = 0
		with torch.no_grad():
		    for data in testloader:
		        images, labels = data
		        outputs = net.model(images)
		        _, predicted = torch.max(outputs.data, 1)
		        total += labels.size(0)
		        correct += (predicted == labels).sum().item()

		print('Accuracy of the network on the 10000 test images: %d %%' % (
		    100 * correct / total))

		res.append(correct/total)
		# class_correct = list(0. for i in range(10))
		# class_total = list(0. for i in range(10))
		# with torch.no_grad():
		#     for data in testloader:
		#         images, labels = data
		#         outputs = net.model(images)
		#         _, predicted = torch.max(outputs, 1)
		#         c = (predicted == labels).squeeze()
		#         for i in range(4):
		#             label = labels[i]
		#             class_correct[label] += c[i].item()
		#             class_total[label] += 1


		# for i in range(10):
		#     print('Accuracy of %5s : %2d %%' % (
		#         classes[i], 100 * class_correct[i] / class_total[i]))
	return res

def evolve_net(net, add_filter=True, add_fc_node=True, point=True):
	# returns new net with mutations on random layers
	if point:
		net.apply_point_mutations(.001, 1/5)
	if add_filter:
		net = net.clone_add_filter_rand()
	if add_fc_node:
		net = net.clone_add_fc_node_rand()
	return net

def evolve_nets(nets, res, top_n):
	# get top n
	top_n_i = sorted(range(len(res)), key = lambda i: res[i], reverse=True)[:top_n]
	top_nets = [nets[i] for i in top_n_i]
	n_dead = len(nets)-top_n
	l = []
	for i in range(n_dead):
		j = i%top_n
		l.append(j)
		nets[i+top_n] = nets[j].clone_add_filter_rand()
	print('    added {}'.format(l))

def gen_population(s):
	p = []
	for i in range(s):
		build_info = rfn.randomize_network(bounded=False)
		p.append(cmm.CustomModelMutable(build_info, 32, CUDA=False))
	return p

def print_pop(p):
	for i, net in enumerate(p):
		
		print(i, '\n\n        ')
		net.print_conv_layers()

# b = rfn.randomize_network(bounded=False)
# cmm.CustomModelMutable(b, 32, CUDA=False).clone_add_filter_rand()

# print('SETUPGLOBALS')
# g = 3
# e = 1
# pop = gen_population(4)
# for generation in range(g):

# 	print('====================\nGEN {}'.format(generation))
# 	# print_pop(pop)
# 	print('    training')
# 	train_nets(pop, e)
# 	print('    testing')
# 	res = test_nets(pop)
# 	print('    evolving')
# 	evolve_nets(pop, res, 2)


n = cmm.CustomModelMutable(rfn.randomize_network(bounded=False), 32, CUDA=False)
# print(n.model)
# n1 = n.clone_with_added_fc_node(0)
# print(n.model)
# print(n1.model)

'''
print('------------')

print(nn.Linear(2, 3).weight.data.size())
print(nn.Linear())

print('------------')

ins = 1
outs = 2
l0 = nn.Linear(ins, outs)
# ins = layer.weight.data.size()[1]
# outs = layer.weight.data.size()[0]

print(l0.weight.size())
# add node to this layer (add an out (init to 1's))
node=torch.ones(ins)
print(torch.cat((l0.weight.data, node.unsqueeze(0)), 0))

#node was added to prev layer, so edit this layer by adding ins to this one (init to 1s)
ins1 = 2
outs1 = 4
l1 = nn.Linear(ins1, outs1)
print(torch.cat((l1.weight.data, torch.ones(outs1).unsqueeze(1)), 1))

#yay!
'''
# l.weight.data




#(in, out)
# a = copy.deepcopy(n.model.conv_0.weight.data)

for i in range(10):
	print(n.model)
	print('---------training net----------')
	train_net(n, 1)
	# print('---------testing net----------')
	# test_nets([n])
	print('---------evolving net----------')
	n = evolve_net(n)
	print('---------testing net (post evolve)----------')
	test_nets([n])

# print(n.model.conv_0.weight-a)