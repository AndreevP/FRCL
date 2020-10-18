import abc
from .models import SplitMnistModel
from .datasets.split_mnist import SplitMnistDataset

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


