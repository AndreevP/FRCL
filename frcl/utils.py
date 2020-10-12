import torch
from torch.utils.data import DataLoader
import abc

class CLBaseEpochQualifier:

    def __init__(self):
        self.dataloader_descr = []

    def append(self, len_dataloader):
        self.dataloader_descr.append(len_dataloader)
    
    @abc.abstractmethod
    def get_iterations_count(self):
        pass

    def __iter__(self):
        if len(self.dataloader_descr) == 0:
            raise Exception("Qualifier is empty")
        for i in range(self.get_iterations_count()):
            yield i

class CLLastDLEpochQualifier(CLBaseEpochQualifier):

    def get_iterations_count(self):
        return self.dataloader_descr[-1]
    

class CLMinDLEpochQualifier(CLBaseEpochQualifier):

    def get_iterations_count(self):
        return min(self.dataloader_descr)

class CLMaxDLEpochQualifier(CLBaseEpochQualifier):

    def get_iterations_count(self):
        return max(self.dataloader_descr)

class CLDataLoaderIter:

    def __init__(self, cl_dataloader):
        self.cl_dataloader = cl_dataloader
        self.qualifier_iter = iter(self.cl_dataloader.epoch_qualifier)
        self.dataloaders_iter = []
        for dataloader in self.cl_dataloader.dataloaders:
            self.dataloaders_iter.append(iter(dataloader))

    
    def __next__(self):
        try:
            it = next(self.qualifier_iter)
        except StopIteration:
            raise

        batch = []
        for i in range(len(self.dataloaders_iter)):
            curr_batch = None
            try:
                curr_batch = next(self.dataloaders_iter[i])
            except StopIteration:
                dataloader = self.cl_dataloader.dataloaders[i]
                self.dataloaders_iter[i] = iter(dataloader)
                curr_batch = next(self.dataloaders_iter[i])
            batch.append(curr_batch)
        return batch

class CLDataLoader:

    def __init__(self, epoch_qualifier_method='last'):
        self.dataloaders = []
        if epoch_qualifier_method == 'last':
            self.epoch_qualifier = CLLastDLEpochQualifier()
        if epoch_qualifier_method == 'min':
            self.epoch_qualifier = CLMinDLEpochQualifier()
        if epoch_qualifier_method == 'max':
            self.epoch_qualifier = CLMaxDLEpochQualifier()

    def append(self, dataset, batch_size):
        curr_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.dataloaders.append(curr_dataloader)
        self.epoch_qualifier.append(len(curr_dataloader))
    
    def __iter__(self):
        return CLDataLoaderIter(self)

def cl_batch_to_device(batch, device='cpu'):
    for _batch in batch:
        _batch[0].to(device); _batch[1].to(device)

def cl_batch_target_float(batch):
    for i in range(len(batch)):
        batch[i][1] = batch[i][1].float()