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