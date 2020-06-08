import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_channels=1, out_classes=2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.avgpool = nn.AvgPool2d(26) # MNIST only
        self.fc = nn.Linear(32, out_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
