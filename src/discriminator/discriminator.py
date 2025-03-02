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
    """
    Discriminator class for Log Mel Spectrograms
    Input must be (Batch_size, Channel, 80, 100)
    """
    def __init__(self):
        super(MelDiscriminator, self).__init__()
        self.features = nn.Sequential(
            # Convolutional block 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Convolutional block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # Convolutional block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 10 * 10, 512),  # Adjust based on input size
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(dim=1)  # Add channel dimension if missing
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # Flatten for fully connected layer
        x = self.fc(x)
        return x
if __name__ == "__main__":
    qq = MelDiscriminator()
    l = qq(testvec)
    p = qq(testvec2)
    print(l.shape,p.shape)
    
