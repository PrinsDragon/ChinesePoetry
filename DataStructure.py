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


def build_rev_dict(dictionary):
    rev_dict = {}
    for word in dictionary:
        cur_id = dictionary[word]
        rev_dict[cur_id] = word
    return rev_dict


def get_poem(poem_list, rev_dict):
    ret = [[], [], [], []]
    for i, sent in enumerate(poem_list):
        for w in sent:
            ret[i].append(rev_dict[int(w)])
    return ret
