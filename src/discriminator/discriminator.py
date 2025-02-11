"""
Convolution Based Discriminator model.

Represents similarity of adversarial noise to target dataset. 0 indicates perfect match with data, 1 represents noise.

"""

import torch
import torch.nn as nn
import pytorch_lightning

testvec = torch.randn(size=(10,80,100))
testvec2 = torch.randn(size=(10,1,80,100))
class MelDiscriminator(nn.Module):
    def __init__(self):
        """
        Discriminator class for Log Mel Spectrograms
        Input must be (Batch_size, Channel, 80, 100)

        
        """
        super(MelDiscriminator, self).__init__()
        
        self.layer1 = nn.Conv2d(1, 32, kernel_size=6, stride=1, padding=2)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=6, stride=2, padding=2)
        self.layer3 = nn.Conv2d(64, 64, kernel_size=6, stride=2, padding=2)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc = nn.Linear(1536,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        
        if len(x.shape) < 4:
            x = x.unsqueeze(dim=1)
        x = self.layer1(x)
        x = self.maxpool1(x)
        
        x = self.layer2(x)
        x = self.maxpool2(x)
        x = self.layer3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x) 
        
        return self.sigmoid(x)
        

if __name__ == "__main__":
    qq = MelDiscriminator()
    l = qq(testvec)
    p = qq(testvec2)
    print(l.shape,p.shape)
    
