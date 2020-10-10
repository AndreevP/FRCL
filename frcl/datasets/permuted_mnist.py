import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import abc

class BasePermutation:
    
    def _set_np_random_state(self, value):
        if isinstance(value, int) or value is None:
            self.np_random_state = np.random.RandomState(seed=value)
            return
        if isinstance(value, np.random.RandomState):
            self.np_random_state = value
            return
        raise Exception(
            "np random state must be initialized"
            " by int, None or np.random.RandomState instance, "
            " got {} instead".format(type(value)))
    
    def _set_torch_random_gen(self, value):
        gen = torch.Generator()
        if value is None:
            gen.seed()
            self.torch_random_gen = gen
            return
        if isinstance(value, int):
            gen.manual_seed(value)
            self.torch_random_gen = gen
            return
        if isinstance(value, torch.Generator):
            self.torch_random_gen = value
            return
        raise Exception(
            "torch random generator must be initialized"
            " by int, None or via torch.Generator instance, "
            " got {} instead".format(type(value)))

    def __init__(self, seed=None):
        self._set_np_random_state(seed)
        self._set_torch_random_gen(seed)
    
    @abc.abstractmethod
    def __call__(self, vector):
        pass

class RandomShufflePermutation(BasePermutation):

    def __init__(self, seed=None):
        super().__init__(seed=seed)
        self.permutation = self.np_random_state.permutation(784)
    
    def __call__(self, vector):
        return vector[self.permutation]

class PermutedMnistDataset(Dataset):

    def __init__(self, permutation, train=True, data_root="../data/", normalize=False):
        self.permutation = permutation
        self.normalize = normalize
        self.full_dataset = torchvision.datasets.MNIST(
            data_root, download=True, train=train)
        
        self.transform = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        im, _cls = self.full_dataset[index]
        tsr = self.transform(im)
        if self.normalize:
            tsr -= torch.mean(tsr)
            tsr /= (torch.std(tsr) + 1e-7)
        return (self.permutation(tsr.view(-1)), _cls)
    
    def __len__(self):
        return len(self.full_dataset)

if __name__ == "__main__":
    perm = RandomShufflePermutation(seed=42)
    ds = PermutedMnistDataset(perm, normalize=True)
    print('train ds length:', len(ds))

    i = 218
    data = ds[i]
    print(data[0].shape)
    print(torch.std(data[0]))
    print(torch.mean(data[0]))