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


n = cmm.CustomModelMutable(rfn.randomize_network(bounded=True), 32, CUDA=False)
# a = copy.deepcopy(n.model.conv_0.weight.data)
train_net(n, 2)
test_nets([n])
n.apply_point_mutations(.1, 1/2)
test_nets([n])
# print(n.model.conv_0.weight-a)