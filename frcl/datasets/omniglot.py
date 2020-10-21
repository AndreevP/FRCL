import torch
from torchvision.datasets import Omniglot
import torchvision
from torchvision.datasets.utils import list_dir, list_files
import numpy as np
import warnings
from os.path import join
from torchvision.transforms import Compose
from torchvision.transforms import Resize, ToTensor, RandomAffine, RandomRotation

class OmniglotOneAlphabetDataset(Omniglot):

    alphabets = ['Alphabet_of_the_Magi',
        'Anglo-Saxon_Futhorc',
        'Arcadian',
        'Armenian',
        'Asomtavruli_(Georgian)',
        'Balinese',
        'Bengali',
        'Blackfoot_(Canadian_Aboriginal_Syllabics)',
        'Braille',
        'Burmese_(Myanmar)',
        'Cyrillic',
        'Early_Aramaic',
        'Futurama',
        'Grantha',
        'Greek',
        'Gujarati',
        'Hebrew',
        'Inuktitut_(Canadian_Aboriginal_Syllabics)',
        'Japanese_(hiragana)',
        'Japanese_(katakana)',
        'Korean',
        'Latin',
        'Malay_(Jawi_-_Arabic)',
        'Mkhedruli_(Georgian)',
        'N_Ko',
        'Ojibwe_(Canadian_Aboriginal_Syllabics)',
        'Sanskrit',
        'Syriac_(Estrangelo)',
        'Tagalog',
        'Tifinagh',
        'Angelic',
        'Atemayar_Qelisayer',
        'Atlantean',
        'Aurek-Besh',
        'Avesta',
        'Ge_ez',
        'Glagolitic',
        'Gurmukhi',
        'Kannada',
        'Keble',
        'Malayalam',
        'Manipuri',
        'Mongolian',
        'Old_Church_Slavonic_(Cyrillic)',
        'Oriya',
        'Sylheti',
        'Syriac_(Serto)',
        'Tengwar',
        'Tibetan',
        'ULOG']

    def __init__(
        self, alphabet_name, train=True, 
        data_root="../data/", test_ratio=0.3, target_type='int', 
        transform=Compose([RandomAffine(30, translate=(0.2, 0.2))])):
        
        assert 0. < test_ratio and test_ratio < 1., 'test ration must be in [0, 1]'
        assert alphabet_name in self.alphabets, "alphabet '{}' not presented".format(alphabet_name)
        self.alphabet_name = alphabet_name
        background = False
        if self.alphabets.index(alphabet_name) < 30:
            background = True
        transforms_list = []
        if train and transform is not None:
            transforms_list.append(transform)
        transforms_list.extend([
            Resize((28, 28)), # see https://arxiv.org/pdf/1606.04080.pdf
            ToTensor()])
        transform = Compose(transforms_list)
        super().__init__(
            data_root, background=background, 
            transform = transform, download=True)
        # super().__init__(
        #     data_root, background=background, download=True) 
        self.alph_index = self._alphabets.index(self.alphabet_name)

        alphs_counts = [sum([len(list_files(join(self.target_folder, a, c), '.png')) for c in list_dir(join(self.target_folder, a))]) for a in self._alphabets]

        n_char_samples = 20
        
        self.alph_counts = alphs_counts[self.alph_index]
        assert(self.alph_counts % n_char_samples == 0)

        self.alph_start = np.cumsum(alphs_counts)[self.alph_index] - self.alph_counts
        char_indices = np.arange(self.alph_start, self.alph_start + self.alph_counts)
        train_ratio = 1. - test_ratio

        if train:
            indices = char_indices.reshape(-1, n_char_samples)[:, :int(n_char_samples * train_ratio)]
        else:
            indices = char_indices.reshape(-1, n_char_samples)[:, int(n_char_samples * train_ratio):]
        classes = np.cumsum(
            np.zeros_like(indices).astype(target_type) + 1, axis=0) - 1
        
        self.indices = indices.flatten()
        self.classes = classes.flatten()
        self.n_classes = self.alph_counts // n_char_samples

        if train and len(self.indices) == 0:
            raise Exception('Train dataset is empty, consider using smaller test_ratio')
        if train and len(self.indices) == self.alph_counts:
            warnings.warn("Train dataset takes all available data")
        if not train and len(self.indices) == self.alph_counts:
            warnings.warn("Test dataset takes all available data")
        if not train and len(self.indices) == 0:
            raise Exception('Test dataset is empty, consider using greater test_ratio')
        self.train = train
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        im, target = super().__getitem__(self.indices[i])
        return im, self.classes[i]

if __name__ == "__main__":
    ds = OmniglotOneAlphabetDataset('Cyrillic')
    assert(ds.n_classes == 33)
    assert(len(ds) == 33 * 14)
    assert(ds[0][1] == 0)
    assert(ds[13][1] == 0)
    assert(ds[14][1] == 1)
    print(ds[0][0].shape)
    ds_t = OmniglotOneAlphabetDataset('Cyrillic', train=False)
    assert(len(ds_t) == 33 * 6)