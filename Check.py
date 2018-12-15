import pickle
import argparse
import random

import matplotlib.pyplot as plt
plt.switch_backend("agg")
import matplotlib.ticker as ticker

import torch
from torch.utils.data import DataLoader

from Model import Model
from DataStructure import PoetryDataSet, build_rev_dict, get_poem

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", default=59, type=int)
args = parser.parse_args()

STEP_ID = args.epoch
model_path = "checkpoint/EPOCH_{}_model.torch".format(STEP_ID)

word_dict_file = open("data/proc/dict.pkl", "rb")
word_dict = pickle.load(word_dict_file)
rev_word_dict = build_rev_dict(word_dict)
word_matrix = torch.load("data/proc/matrix.torch")

batch_size = 100
vocabulary_size = len(word_dict)
embedding_dim = 128
fc_dim = 256
hidden_size = 64

dataset_name = "train"
dataset_file = open("data/proc/{}.pkl".format(dataset_name), "rb")
dataset_list = pickle.load(dataset_file)
dataset = PoetryDataSet(dataset_list)
dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

model = Model(vocab_size=vocabulary_size, embedding_dim=embedding_dim, batch_size=batch_size, fc_dim=fc_dim,
              hidden_size=hidden_size, word_vec_matrix=word_matrix).cuda()

model.load_state_dict(torch.load(model_path))

for poem_id, poem in enumerate(dataloader):
    project_first = model(poem[0])
    pred_first = project_first.max(dim=2)[1]
    acc_first = sum([sum([1 if p == t else 0 for p, t in zip(pred, tag)]) / 7
                     for pred, tag in zip(pred_first, poem[1])]) / batch_size

    project_third = model(poem[2])
    pred_third = project_third.max(dim=2)[1]
    acc_third = sum([sum([1 if p == t else 0 for p, t in zip(pred, tag)]) / 7
                     for pred, tag in zip(pred_third, poem[3])]) / batch_size

    acc = (acc_first + acc_third) / 2

    for i in range(batch_size):
        result = [poem[0][i], pred_first[i], poem[2][i], pred_third[i]]
        result = get_poem(result, rev_word_dict)

        ori = [poem[0][i], poem[1][i], poem[2][i], poem[3][i]]
        ori = get_poem(ori, rev_word_dict)

        pass

