

import torch
import torch.nn as nn

class LeNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(400, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.maxpool1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.maxpool2(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.linear1(x)
        # print(x.shape)
        x = self.linear2(x)
        # print(x.shape)
        x = self.linear3(x)
        #print(x.shape)
        out = self.sigmoid(x)
        #print(x.shape)
        
        return out
    

if __name__ == '__main__':

    model = LeNet()
    x = torch.zeros((1,1,28,28))
    print(x.shape)
    out = model(x)