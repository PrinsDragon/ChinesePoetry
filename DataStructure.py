import torch
from torch.utils.data import Dataset

class PoetryDataSet(Dataset):
    def __init__(self, poem):
        super(PoetryDataSet, self).__init__()
        self.poem = poem
        self.length = len(poem)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.Tensor(self.poem[index][0]).long().cuda(), \
               torch.Tensor(self.poem[index][1]).long().cuda(), \
               torch.Tensor(self.poem[index][2]).long().cuda(), \
               torch.Tensor(self.poem[index][3]).long().cuda(),


