import abc
from .models import SplitMnistModel, PermutedMnistModel
from .datasets.split_mnist import SplitMnistDataset
from .datasets.permuted_mnist import RandomShufflePermutation, PermutedMnistDataset
import numpy as np

class Task(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractproperty
    def model(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, i):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

class SplitMnistTask(Task):

    @property
    def model(self):
        return SplitMnistModel()

    def __init__(self):
        super().__init__()
        self.tasks = [(0, 1), (2, 3), (4 ,5), (6, 7), (8, 9)]
    
    def __getitem__(self, i):
        lbl_0, lbl_1 = self.tasks[i]
        train_ds = SplitMnistDataset(lbl_0, lbl_1, normalize=True)
        test_ds = SplitMnistDataset(lbl_0, lbl_1, train=False, normalize=True)
        return train_ds, test_ds
    
    def __len__(self):
        return len(self.tasks)

class PermutedMnistTask(Task):
    
    def get_permutation(self, i):
        if len(self.permutations) > i:
            return self.permutations[i]
        n_seeds = i - len(self.permutations) + 1
        seeds = self.np_random_state.randint(10000, size=n_seeds)
        for seed in seeds:
            self.permutations.append(RandomShufflePermutation(seed=seed))
        return self.permutations[i]

    @property
    def model(self):
        return PermutedMnistModel()
    
    def __init__(self, n_tasks, seed=None):
        super().__init__()
        self.n_tasks = n_tasks
        self.permutations = []
        if seed is None or isinstance(seed, int):
            np_random_state = np.random.RandomState(seed=seed)
            seeds = np_random_state.randint(1, high=self.n_tasks * 10, size=self.n_tasks)
            seeds = seeds.cumsum()
        elif isinstance(seed, list):
            assert(len(seed) == self.n_tasks)
            seeds = seed
        else:
            raise Exception("seed must be None, int or list of ints, "
                    "got '{}' instead".format(type(seed)))
        for i in range(len(seeds)):
            seed = seeds[i]
            self.permutations.append(RandomShufflePermutation(seed=int(seed)))
    
    def __getitem__(self, i):
        permutation = self.permutations[i]
        train_ds = PermutedMnistDataset(permutation, normalize=True)
        test_ds = PermutedMnistDataset(permutation, train=False, normalize=True)
        return train_ds, test_ds
    
    def __len__(self):
        return self.n_tasks

