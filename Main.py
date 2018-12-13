import pickle
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Model import Encoder, Decoder
from DataStructure import PoetryDataSet

OUTPUT_FILE = open("checkpoint/output.txt", "w", encoding="utf-8")
SAVE_Delta = 1000


def output(string):
    print(string)
    print(string, file=OUTPUT_FILE)


word_dict_file = open("data/proc/dict.pkl", "rb")
word_dict = pickle.load(word_dict_file)
word_matrix = torch.load("data/proc/matrix.torch")

dataset_name = ["train", "val"]
dataset_file = [open("data/proc/{}.pkl".format(n), "rb") for n in dataset_name]
dataset_list = [pickle.load(f) for f in dataset_file]

epoch_num = 1000
seq_length = 7
vocabulary_size = len(word_dict)
embedding_dim = 128
hidden_size = 128
teacher_forcing_ratio = 0.5

START = 0
END = 1

train_dataset = PoetryDataSet(dataset_list[0])
valid_dataset = PoetryDataSet(dataset_list[1])

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, shuffle=True, batch_size=1, drop_last=True)

encoder = Encoder(vocab_size=vocabulary_size, hidden_size=hidden_size).cuda()
decoder = Decoder(vocab_size=vocabulary_size, hidden_size=hidden_size).cuda()

encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=0.01, weight_decay=0.0001)
decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=0.01, weight_decay=0.0001)
loss_func = nn.NLLLoss(reduce=True, size_average=True)

plot_loss = {"train": [], "valid": []}
global_step = 0

class SaveHelper:
    def __init__(self, delta):
        self.global_step = 0
        self.delta = delta
        self.save_op = False if delta == -1 else True

    def step(self):
        self.global_step += 1
        if self.save_op and (self.global_step % self.delta == 0):
            torch.save(encoder.state_dict(), "checkpoint/STEP_{}_Encoder.torch".format(self.global_step))
            torch.save(decoder.state_dict(), "checkpoint/STEP_{}_Decoder.torch".format(self.global_step))
            plot_file = open("checkpoint/STEP_{}_plot.pkl".format(self.global_step), "wb")
            pickle.dump(plot_loss, plot_file)

            print("# Save at step: {}".format(self.global_step))


save_helper = SaveHelper(delta=SAVE_Delta)


def proc_poem(input_tensor, target_tensor, evaluate=False):
    # for one poem
    encoder_hidden = encoder.init_hidden()

    input_length = input_tensor.shape[0]
    target_length = target_tensor.shape[0]

    encoder_outputs = torch.zeros(input_length, encoder.hidden_size).cuda()

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[START]]).cuda()

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if evaluate:
        use_teacher_forcing = False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += loss_func(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            top_v, top_i = decoder_output.topk(1)
            decoder_input = top_i.squeeze().detach()  # detach from history as input

            loss += loss_func(decoder_output, target_tensor[di])
            if decoder_input.item() == END:
                break

    return loss


def train(epoch_id, plot_loss):
    encoder.train()
    decoder.train()

    running_loss = 0.

    for poem_id, poem in enumerate(train_dataloader):
        encoder.zero_grad()
        decoder.zero_grad()

        poem = [p.view(-1, 1) for p in poem]

        loss = proc_poem(poem[0], poem[1]) + proc_poem(poem[2], poem[3])
        cur_loss = float(loss) / 2 / seq_length
        plot_loss.append(cur_loss)
        if poem_id % 100 == 0:
            print("poem id: {}, loss={:.5f}".format(poem_id, cur_loss))

        save_helper.step()

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        running_loss += loss / 2 / seq_length

    running_loss /= train_dataset.poem_num
    output("Train: [%d/%d] Loss: %.5f" % (epoch_id, epoch_num, running_loss))


def valid(epoch_id, plot_loss):
    encoder.eval()
    decoder.eval()

    running_loss = 0.

    for poem_id, poem in enumerate(valid_dataloader):
        poem = [p.view(-1, 1) for p in poem]
        loss = proc_poem(poem[0], poem[1], evaluate=True) + proc_poem(poem[2], poem[3], evaluate=True)
        plot_loss.append(float(loss) / 2 / seq_length)
        running_loss += loss / 2 / seq_length

    running_loss /= train_dataset.poem_num
    output("Valid: [%d/%d] Loss: %.5f" % (epoch_id, epoch_num, running_loss))


for epoch_id in range(1, epoch_num + 1):
    train(epoch_id, plot_loss["train"])
    valid(epoch_id, plot_loss["valid"])
