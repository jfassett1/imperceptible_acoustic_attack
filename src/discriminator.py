"""
Convolution Based Discriminator model.

Represents similarity of adversarial noise to target dataset. Should be close to zero.

"""

import torch
import torch.nn as nn
import pytorch_lightning

testvec = torch.randn(size=(1,1,80,100))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.layer1 = nn.Conv2d(1, 32, kernel_size=6, stride=1, padding=2)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=6, stride=2, padding=2)
        self.layer3 = nn.Conv2d(64, 64, kernel_size=6, stride=2, padding=2)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc = nn.Linear(1536,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        """

        """
        x = self.layer1(x)
        x = self.maxpool1(x)
        
        x = self.layer2(x)
        x = self.maxpool2(x)
        x = self.layer3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x) 
        
        return self.sigmoid(x)

if __name__ == "__main__":
    qq = Discriminator()
    l = qq(testvec)
    # print(qq.parameters())
