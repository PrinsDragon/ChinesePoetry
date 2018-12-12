import torch
from torch.utils.data import Dataset

START = 0
END = 1

# TODO: add END to each sentence of the dataset

class PoetryDataSet(Dataset):
    def __init__(self, poem, mode="simple"):
        super(PoetryDataSet, self).__init__()
        self.poem = poem
        self.poem_num = len(poem)

    def __len__(self):
        return self.poem_num

    def __getitem__(self, index):
        return torch.Tensor(self.poem[index][0] + [END]).long().cuda(), \
               torch.Tensor(self.poem[index][1] + [END]).long().cuda(), \
               torch.Tensor(self.poem[index][2] + [END]).long().cuda(), \
               torch.Tensor(self.poem[index][3] + [END]).long().cuda(),

def build_rev_dict(dictionary):
    rev_dict = {}
    for word in dictionary:
        cur_id = dictionary[word]
        rev_dict[cur_id] = word
    return rev_dict
