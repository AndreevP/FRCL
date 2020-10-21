import abc
from .models import SplitMnistModel, PermutedMnistModel, OmniglotModel
from .datasets.split_mnist import SplitMnistDataset
from .datasets.permuted_mnist import RandomShufflePermutation, PermutedMnistDataset
from .datasets.omniglot import OmniglotOneAlphabetDataset
from .utils import ScaleDataset
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

    def __init__(self, return_n_classes=True):
        super().__init__()
        self.return_n_classes = return_n_classes
        self.tasks = [(0, 1), (2, 3), (4 ,5), (6, 7), (8, 9)]
    
    def __getitem__(self, i):
        lbl_0, lbl_1 = self.tasks[i]
        train_ds = SplitMnistDataset(lbl_0, lbl_1, normalize=True)
        test_ds = SplitMnistDataset(lbl_0, lbl_1, train=False, normalize=True)
        if not self.return_n_classes:
            return train_ds, test_ds
        return train_ds, test_ds, 2
    
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
    
    def __init__(self, n_tasks, seed=None, return_n_classes=True):
        super().__init__()
        self.return_n_classes=return_n_classes
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
        if not self.return_n_classes:
            return train_ds, test_ds
        return train_ds, test_ds, 10
    
    def __len__(self):
        return self.n_tasks

class OmniglotTask(Task):

    @property
    def model(self):
        return OmniglotModel()
    
    def __init__(self, task_indices=None, n_tasks=None, return_n_classes=True, n_scale=20):
        '''
        :Parameters:
        task_indices : list : numbers of tasks to use
        n_tasks : int : count of first tasks to use
        return_n_classes: bool : should task return n_classes for each task
        n_scale : int : parameter to scale dataset (for comfortable learning)
        '''
        super().__init__()
        self.return_n_classes = return_n_classes
        if task_indices is not None and n_tasks is not None:
            raise Exception("only one parameter of 'task_indices' and 'n_tasks'"
                " can be initialized")
        self.task_indices = list(range(50))
        if task_indices is not None:
            for i in task_indices:
                assert 0 <= i and i < 50 , 'task numbers must be in [0, 50)'
            self.task_indices = task_indices
        if n_tasks is not None:
            self.task_indices = list(range(n_tasks))
        self.alphabets = OmniglotOneAlphabetDataset.alphabets
        self.n_scale = n_scale
        self.lazy_datasets = {}
    
    def __len__(self):
        return len(self.task_indices)
    
    def __getitem__(self, i):
        i_task = self.task_indices[i]
        i_alph = self.alphabets[i_task]
        if i_alph in self.lazy_datasets.keys():
            train_ds, test_ds = self.lazy_datasets[i_alph]
        else:
            train_ds = ScaleDataset(OmniglotOneAlphabetDataset(i_alph), self.n_scale)
            test_ds = ScaleDataset(OmniglotOneAlphabetDataset(i_alph, train=False), self.n_scale)
            self.lazy_datasets[i_alph] = (train_ds, test_ds)
        n_classes = train_ds.dataset.n_classes
        if not self.return_n_classes:
            return train_ds, test_ds
        return train_ds, test_ds, n_classes

