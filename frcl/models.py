import torch
import torch.nn as nn

#TODO: implement initialization using custom torch.generator
class MnistModel(nn.Module):

    def __init__(self, hid):
        super().__init__()
        self.hid = hid

        self.layers = nn.Sequential(
            nn.Linear(784, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU()
        )
    
    def forward(self, input):
        return self.layers(input)

class SplitMnistModel(MnistModel):

    def __init__(self, hid=256):
        super().__init__(hid)

class PermutedMnistModel(MnistModel):

    def __init__(self, hid=100):
        super().__init__(hid)

if __name__ == "__main__":
    smm = SplitMnistModel()
    pmm = PermutedMnistModel()

class OmniglotModel(nn.Module):

    @staticmethod
    def _get_block(in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, (3, 3), padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

    def __init__(self):
        super().__init__()
        self.b1 = self._get_block(1)
        self.b2 = self._get_block(64)
        self.b3 = self._get_block(64)
        self.b4 = self._get_block(64)
        self.flatten = nn.Flatten()
        self.hid = 64
    
    def forward(self, input):
        x = self.b1(input)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        return self.flatten(x)

