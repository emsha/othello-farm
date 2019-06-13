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


def train_net(net):
	print("training a net")
	for epoch in range(1):  # loop over the dataset multiple times
	    running_loss = 0.0
	    for i, data in enumerate(trainloader, 0):
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
	        if i % 2000 == 1999:    # print every 2000 mini-batches
	            print('[%d, %5d] loss: %.3f' %
	                  (epoch + 1, i + 1, running_loss / 2000))
	            running_loss = 0.0


def train_nets(l):
	for net in l:
		train_net(net)



build_info = rfn.randomize_network(bounded=False)
n0 = cmm.CustomModelMutable(build_info, 32, CUDA=False)
# print(n0.model)
# print('pre clone')
# print(n0.model.conv_0.weight.data.size())
# print(n0.model.conv_1.weight.data.size())
# print(n0.model.conv_2.weight.data.size())
# print(n0.build_info)
n0_0 = n0.clone_with_added_filter(2)
# print('\n\npost clone:')
# print(n0.model.conv_0.weight.data.size())
# print(n0.model.conv_1.weight.data.size())
# print(n0.model.conv_2.weight.data.size())
# print(n0.build_info)
n0_1 = n0.clone_with_added_filter(2)
# n0_2 = n0.clone_with_added_filter(2)

# leest = []
# leest.append(n0)
# leest.append(n0_0)
# leest.append(n0_1)
# leest.append(n0_2)
# print(n1.model)

train_net(n0_0)


















'''
for g in range(5):
	print('generation {}'.format(g))
	for net in nets:
		print("training a net")
		for epoch in range(1):  # loop over the dataset multiple times

		    running_loss = 0.0
		    for i, data in enumerate(trainloader, 0):
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
		        if i % 2000 == 1999:    # print every 2000 mini-batches
		            print('[%d, %5d] loss: %.3f' %
		                  (epoch + 1, i + 1, running_loss / 2000))
		            running_loss = 0.0
	print('clone last net')
	nets.append(nets[-1].clone())
	print([net for net.model in nets])

# clone
# nett_clone = nett.clone()
# nets.append(nett_clone)

# print(nett.model, nett_clone.model)
# test nets
# print("testing both nets, only first net trained")
for i, net in enumerate(nets):

	correct = 0
	total = 0
	with torch.no_grad():
	    for data in testloader:
	        images, labels = data
	        outputs = net.model(images)
	        _, predicted = torch.max(outputs.data, 1)
	        total += labels.size(0)
	        correct += (predicted == labels).sum().item()

	print('Accuracy of network %i on the 10000 test images: %d %%' % (i, 
	    100 * correct / total))

	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	with torch.no_grad():
	    for data in testloader:
	        images, labels = data
	        outputs = net.model(images)
	        _, predicted = torch.max(outputs, 1)
	        c = (predicted == labels).squeeze()
	        for i in range(4):
	            label = labels[i]
	            class_correct[label] += c[i].item()
	            class_total[label] += 1


	for i in range(10):
	    print('Accuracy of %5s : %2d %%' % (
	        classes[i], 100 * class_correct[i] / class_total[i]))

print("training both nets now")
# train both nets
for net in nets:
	print("training a net")
	for epoch in range(2):  # loop over the dataset multiple times

	    running_loss = 0.0
	    for i, data in enumerate(trainloader, 0):
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
	        if i % 2000 == 1999:    # print every 2000 mini-batches
	            print('[%d, %5d] loss: %.3f' %
	                  (epoch + 1, i + 1, running_loss / 2000))
	            running_loss = 0.0

# print('testing both nets, net 1 trained twice, net 2 cloned frm net 1 post first training, then trained again, so almost like 1.5x trained, where net 1 is 2x trained')
# test nets
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

	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	with torch.no_grad():
	    for data in testloader:
	        images, labels = data
	        outputs = net.model(images)
	        _, predicted = torch.max(outputs, 1)
	        c = (predicted == labels).squeeze()
	        for i in range(4):
	            label = labels[i]
	            class_correct[label] += c[i].item()
	            class_total[label] += 1


	for i in range(10):
	    print('Accuracy of %5s : %2d %%' % (
	        classes[i], 100 * class_correct[i] / class_total[i]))
# '''