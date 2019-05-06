import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from stateB import State
import torch.nn.functional as F
import torch.optim as optim

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

learning_rate = 0.01
batch_size = 100
num_epochs = 5
num_classes = 64
in_channels = 3
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(4096, 1000)
        self.fc2 = nn.Linear(1000, 64)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(out.reshape(out.size(0), -1))
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def train_on_batch(self, in_batch, label_batch, epochs, learning_rate):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.2)

        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = self(in_batch)
            # loss = criterion(outputs, label_batch.long())
            loss = criterion(outputs, torch.max(label_batch, 1)[1])
            loss.backward()
            optimizer.step()

    def forward_indices(self, x):
        return torch.max(self.forward(x), 1)

'''
input: 3 channels of padded 64 lists i.e. 3 channels of len 100 lists
output: 64 outputs repping each spot on the board



tensors: (batch size, channels, w, h)


4 (64 feature maps) 6 (128 feature maps) or 8 (256 feature maps) colvolutional layers -> 2 FC layers, 128 units, relu, output layer -> 64 (they do 60)
padding (done)
no pooling
'''
'''
self.layer2 = nn.Sequential(
    #(channels, )
    nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=1),
    nn.ReLU(),
    )
self.layer3 = nn.Sequential(
    #(channels, )
    nn.Conv2d(36, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    )
self.drop_out = nn.Dropout()
'''
def main():
    net = ConvNet()
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())


if __name__ == '__main__':
    main()