import pickle
import random
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Model import EncoderRNN, LuongAttnDecoderRNN
from DataStructure import masked_cross_entropy, random_batch, get_poem, build_rev_dict

directory = "checkpoint_{}".format(time.strftime('%H-%M-%S', time.localtime(time.time())))
os.makedirs(directory)
OUTPUT_FILE = open("{}/output.txt".format(directory), "w", encoding="utf-8")
SAVE_Delta = 100


def output(string, end="\n"):
    print(string, end=end)
    print(string, end=end, file=OUTPUT_FILE)


word_dict_file = open("data/proc/dict.pkl", "rb")
word_dict = pickle.load(word_dict_file)
rev_dict = build_rev_dict(word_dict)
word_matrix = torch.load("data/proc/matrix.torch")

dataset_name = ["train", "val"]
dataset_file = [open("data/proc/{}_proc.pkl".format(n), "rb") for n in dataset_name]
dataset_list = {n: pickle.load(f) for n, f in zip(dataset_name, dataset_file)}

step_num = 50000
seq_length = 7
vocabulary_size = len(word_dict)
embedding_dim = 128
hidden_size = 128
teacher_forcing_ratio = 0.8
batch_size = 100
layer_num = 2
attention_model = "general"
clip = 50.0
eval_delta = 1

PAD = 0
SOS = 1
EOS = 2

encoder = EncoderRNN(vocabulary_size, hidden_size, layer_num).cuda()
decoder = LuongAttnDecoderRNN(attention_model, hidden_size, vocabulary_size, layer_num).cuda()

encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.01, weight_decay=0.0001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.01, weight_decay=0.0001)
loss_func = nn.CrossEntropyLoss(reduce=True, size_average=True)


class SaveHelper:
    def __init__(self, delta):
        self.global_step = 0
        self.delta = delta
        self.save_op = False if delta == -1 else True
        self.plot_loss = {"train": [], "valid": []}

    def step(self):
        self.global_step += 1
        if self.save_op and (self.global_step % self.delta == 0):
            torch.save(encoder.state_dict(), "{}/STEP_{}_Encoder.torch".format(directory, self.global_step))
            torch.save(decoder.state_dict(), "{}/STEP_{}_Decoder.torch".format(directory, self.global_step))
            plot_file = open("{}/STEP_{}_plot.pkl".format(directory, self.global_step), "wb")
            pickle.dump(self.plot_loss, plot_file)

            output("# Save at step: {}".format(self.global_step))


save_helper = SaveHelper(delta=SAVE_Delta)


def train(input_batches, target_batches):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0.

    # Run words through encoder
    encoder_outputs, encoder_hidden, class_out = encoder(input_batches, None)

    # Prepare input and output variables
    decoder_input = torch.Tensor([SOS] * batch_size).long().cuda()
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = target_batches.shape[0]
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size).cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        loss += loss_func(decoder_output, target_batches[t])
        loss += loss_func(class_out[t], target_batches[t]) * 0.5

        all_decoder_outputs[t] = decoder_output
        is_teacher = random.random() < teacher_forcing_ratio
        top1 = decoder_output.max(1)[1]
        decoder_input = target_batches[t] if is_teacher else top1

        # decoder_input = target_batches[t]  # Next input is current target

    loss /= max_target_length

    # Loss calculation and back propagation
    # loss = masked_cross_entropy(
    #     all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
    #     target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
    #     [max_target_length for _ in range(batch_size)]
    # )
    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    sample_index = random.randint(0, batch_size - 1)
    poem = [input_batches.transpose(0, 1)[sample_index],
            all_decoder_outputs.max(-1)[1].transpose(0, 1)[sample_index],
            target_batches.transpose(0, 1)[sample_index]]

    return float(loss), ec, dc, poem


def evaluate(input_batches, target_batches):
    encoder.eval()
    decoder.eval()

    # Run words through encoder
    encoder_outputs, encoder_hidden, class_out = encoder(input_batches, None)

    loss = 0.

    # Prepare input and output variables
    decoder_input = torch.Tensor([SOS] * batch_size).long().cuda()
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = target_batches.shape[0]
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size).cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        loss += loss_func(decoder_output, target_batches[t])
        loss += loss_func(class_out[t], target_batches[t]) * 0.5

        all_decoder_outputs[t] = decoder_output
        top1 = decoder_output.max(1)[1]
        decoder_input = top1

    # Loss calculation and back propagation
    # loss = masked_cross_entropy(
    #     all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
    #     target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
    #     [max_target_length for _ in range(batch_size)]
    # )

    loss /= batch_size

    sample_index = random.randint(0, batch_size - 1)
    poem = [input_batches.transpose(0, 1)[sample_index],
            all_decoder_outputs.max(-1)[1].transpose(0, 1)[sample_index],
            target_batches.transpose(0, 1)[sample_index]]

    encoder.train()
    decoder.train()

    return float(loss), poem


train_loss = 0
for step_id in range(1, step_num + 1):
    # Get training data for this cycle
    input_batches, target_batches = random_batch(dataset_list["train"], batch_size)

    # Run the train function
    loss, ec, dc, poem = train(input_batches, target_batches)
    train_loss += loss

    save_helper.plot_loss["train"].append(loss)

    if step_id % eval_delta == 0:
        output("Train: [{}/{}] loss={:.5f}".format(step_id, step_num, train_loss / eval_delta))
        train_loss = 0
        poem = get_poem(poem, rev_dict)
        for sent in poem:
            for i in range(7):
                output(sent[i], end="")
            output("\t", end="")
        output("")
        # output("-----------------------------")
        #
        # input_batches, target_batches = random_batch(dataset_list["val"], batch_size)
        # loss, poem = evaluate(input_batches, target_batches)
        # output("Eval: [{}/{}] loss={:.5f}".format(step_id, step_num, loss))
        # poem = get_poem(poem, rev_dict)
        # for sent in poem:
        #     for i in range(7):
        #         output(sent[i], end="")
        #     output("\t", end="")
        # output("")
        output("=============================")

    save_helper.step()
