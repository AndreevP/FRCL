import torch
import torchvision
from torch.utils.data import Dataset

class SplitMnistDataset(Dataset):

    def __init__(
        self, lbl_1, lbl_2, 
        train=True, data_root="../data/", normalize=False):

        '''
        Creates split mnist dataset
        Parameters:
        :lbl_1: int : first lable of class to load
        :lbl_2: int : second lable of class to load
        :data_root: str : directory where the full mnist dataset will be stored
        '''

        assert(lbl_1 in range(10))
        assert(lbl_2 in range(10))
        assert(lbl_1 != lbl_2)
        self.lbl_1 = lbl_1
        self.lbl_2 = lbl_2

        self.full_dataset = torchvision.datasets.MNIST(data_root, download=True, train=train)
        self.dataset_indices = []

        for i, data in enumerate(self.full_dataset):
            _, lbl = data
            if lbl == self.lbl_1:
                self.dataset_indices.append(i)
            if lbl == self.lbl_2:
                self.dataset_indices.append(i)
        
        self.transform = torchvision.transforms.ToTensor()
        self.normalize = normalize

        
    def __getitem__(self, index):
        im, _cls = self.full_dataset[self.dataset_indices[index]]
        tsr = self.transform(im)
        if self.normalize:
            tsr -= torch.mean(tsr)
            tsr /= (1e-7 + torch.std(tsr))
        return (tsr.view(-1), _cls - min(self.lbl_1, self.lbl_2)) #return binary labels
    
    def __len__(self):
        return len(self.dataset_indices)
            
if __name__ == "__main__":

    ds = SplitMnistDataset(0, 1, train=True)
    ds_test = SplitMnistDataset(0, 1, train=False)
    print('train ds length:', len(ds))
    print('test ds length:', len(ds_test))
    i = 218
    data = ds[i]
    assert(data[1] in [0, 1])
    data = ds_test[i + 57]
    assert(data[1] in [0, 1])


         



