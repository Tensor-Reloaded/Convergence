import torch
import torch.nn as nn

class BottleneckModel(nn.Module):
    def __init__(self, originalModule, bottleneck_size):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        assert hasattr(originalModule, 'classifier')
        assert isinstance(originalModule.classifier, nn.Linear)
        self.bottleneck = nn.Sequential(
            nn.Linear(originalModule.classifier.in_features, bottleneck_size),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_size, originalModule.classifier.out_features),
            nn.ReLU()
        )

        originalModule.classifier = nn.Identity() #do nothing just forward input

        self.module = originalModule


    def forward(self, x):
        x = self.module(x)
        x = self.bottleneck(x)
        x = self.classifier(x)
        return x

    def get_bottleneck_repr(self, x):
        x = self.module(x)
        x = self.bottleneck(x)
        return x