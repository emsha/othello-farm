import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from stateB import State
import torch.nn.functional as F
import torch.optim as optim
import random

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

learning_rate = 0.001
batch_size = 100
num_epochs = 10
num_classes = 64
in_channels = 3
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=1)
        self.fc1 = nn.Linear(57600, 1000)#64*36, 1000)
        self.fc2 = nn.Linear(1000, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.2)

        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(out.reshape(out.size(0), -1))
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def train_on_batch(self, in_batch, label_batch, epochs, learning_rate):
        
        for _ in range(epochs):
            self.optimizer.zero_grad()
            outputs = self(in_batch)
            # loss = criterion(outputs, label_batch.long())
            loss = self.criterion(outputs, torch.max(label_batch, 1)[1])
            loss.backward()
            self.optimizer.step()

    def forward_indices(self, x):
        return torch.max(self.forward(x), 1)

    
    def clone(self, mutations=False):
        '''
        returns clone of net
        point mutates weights randomly if mutate=True
        '''
        # make clone
        clone = ConvNet()
        clone.load_state_dict(self.state_dict())
        if mutations:
            #conv1
            clone.conv1.weight.data.add_(ConvNet.generate_mutations(self.conv1, .0005, 150))
            
            # fc1
            clone.fc1.weight.data.add_(ConvNet.generate_mutations(self.fc1, .0005, 150))

            # fc2
            clone.fc2.weight.data.add_(ConvNet.generate_mutations(self.fc2, .0005, 150))
        return clone

    @staticmethod
    def generate_mutations(net_layer, bound, rate_denom):
        '''
        rate_denom: the chance of mutation is 1/rate_denom
        bound: mutations range from -bound to bound
        net_layer: is for example net.conv1
        returns tensor same size as layer with zeroes and a few mutations.
        to apply mutation: add returned tensor to net.layer.weight.data like layer.weight.data.add_(returned tensor)
        '''
        shape = net_layer.weight.size()
        mutations = torch.empty(shape).uniform_(-bound, bound)
        mask = torch.randint(0, rate_denom, shape)
        masked_mutations = torch.where(mask==0, mutations, torch.zeros(shape))
        
        return masked_mutations 

def main():
    net = ConvNet()
    # params = list(net.parameters())
    # print(len(params))
    # print(params[0].size())
    # net.mutate()
    clone = ConvNet.clone(net, mutations=True)
    print(net, clone)
    # ConvNet.copy(net)


if __name__ == '__main__':
    main()