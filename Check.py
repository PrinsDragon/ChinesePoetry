import pickle
import argparse
import random

import matplotlib.pyplot as plt
plt.switch_backend("agg")
import matplotlib.ticker as ticker

import torch
from torch.utils.data import DataLoader

from Model import Encoder, Decoder
from DataStructure import PoetryDataSet, build_rev_dict, get_poem

parser = argparse.ArgumentParser()
parser.add_argument("--step", default=5000, type=int)
args = parser.parse_args()

STEP_ID = args.step

# plt.figure()
# plt.plot(train_plot)
# plt.savefig("checkpoint/STEP_{}_train_loss.png".format(STEP_ID))

word_dict_file = open("data/proc/dict.pkl", "rb")
word_dict = pickle.load(word_dict_file)
rev_word_dict = build_rev_dict(word_dict)
word_matrix = torch.load("data/proc/matrix.torch")

dataset_name = "val"
dataset_file = open("data/proc/{}.pkl".format(dataset_name), "rb")
dataset_list = pickle.load(dataset_file)
dataset = PoetryDataSet(dataset_list)
dataloader = DataLoader(dataset)

seq_length = 7
target_length = 7
vocabulary_size = len(word_dict)
embedding_dim = 128
hidden_size = 128
teacher_forcing_ratio = 0.5

START = 0
END = 1

encoder = Encoder(vocab_size=vocabulary_size, hidden_size=hidden_size).cuda()
decoder = Decoder(vocab_size=vocabulary_size, hidden_size=hidden_size).cuda()

encoder.cuda()
decoder.cuda()

def proc_poem(input_tensor):
    predict = []

    # for one poem
    encoder_hidden = encoder.init_hidden()

    input_length = input_tensor.shape[0]

    encoder_outputs = torch.zeros(input_length, encoder.hidden_size).cuda()

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[START]]).cuda()

    decoder_hidden = encoder_hidden

    # Without teacher forcing: use its own predictions as the next input
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        top_v, top_i = decoder_output.topk(1)
        decoder_input = top_i.squeeze().detach()  # detach from history as input

        predict.append(int(decoder_input))

        if decoder_input.item() == END:
            break

    return predict

for poem in dataloader:
    poem = [p.view(-1, 1) for p in poem]
    predict = [[int(w) for w in poem[0]], proc_poem(poem[0]), [int(w) for w in poem[2]], proc_poem(poem[2])]
    result = get_poem(predict, rev_word_dict)


# plot_file = open("checkpoint/STEP_{}_plot.pkl".format(STEP_ID), "rb")
# plot_dict = pickle.load(plot_file)
#
# train_plot = plot_dict["train"]
# valid_plot = plot_dict["valid"]

