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

build_info = rfn.randomize_network(bounded=False)
# print(build_info['conv_layers'].__class__)
nett = cmm.CustomModelMutable(build_info, 32, CUDA=False)
print(nett.model)
# new_net = nett.clone()
# print(new_net.model)
nets = [nett]






print('\n\nset up data:')

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

# train only first net
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
'''
# clone
nett_clone = nett.clone()
nets.append(nett_clone)

print(nett.model, nett_clone.model)
# test nets
print("testing both nets, only first net trained")
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

print('testing both nets, net 1 trained twice, net 2 cloned frm net 1 post first training, then trained again, so almost like 1.5x trained, where net 1 is 2x trained')
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
'''