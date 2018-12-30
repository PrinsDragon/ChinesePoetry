import pickle
import random
import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Model import EncoderRNN, LuongAttnDecoderRNN
from DataStructure import masked_cross_entropy, random_batch, get_poem, build_rev_dict, cal_bleu, sequential_batch

TAG = "4Sent-Middle-py"
directory = "checkpoint/checkpoint_{}_{}".format(TAG, time.strftime('%H-%M-%S', time.localtime(time.time())))
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

sound_dict_file = open("data/proc/sound_dict_s.pkl", "rb")
sound_dict_s = pickle.load(sound_dict_file)

sound_dict = sound_dict_s[0]
char2num = sound_dict_s[1]
char2final = sound_dict_s[2]

dataset_name = ["train", "val"]
dataset_file = [open("data/proc/{}_proc.pkl".format(n), "rb") for n in dataset_name]
dataset_list = {n: pickle.load(f) for n, f in zip(dataset_name, dataset_file)}

step_num = 50000
seq_length = 7
vocabulary_size = len(word_dict)
embedding_dim = 128

extra_dim_sound = 32
extra_dim_num = 32

hidden_size = 128
teacher_forcing_ratio = 0.8
batch_size = 100
layer_num = 2
attention_model = "concat"
clip = 50.0
eval_delta = 100

PAD = 0
SOS = 1
EOS = 2

encoder = EncoderRNN(input_size=vocabulary_size, hidden_size=hidden_size, n_layers=layer_num,
                     extra_size_sound=len(sound_dict), extra_size_num=5,
                     extra_dim_sound=extra_dim_sound, extra_dim_num=extra_dim_num, word_matrix=word_matrix,
                     ).cuda()
decoder = LuongAttnDecoderRNN(attn_model=attention_model, hidden_size=hidden_size, output_size=vocabulary_size,
                              n_layers=layer_num, extra_size_sound=len(sound_dict), extra_size_num=5,
                              extra_dim_sound=extra_dim_sound, extra_dim_num=extra_dim_num).cuda()

# checkpoint_to_load = 11800
# encoder.load_state_dict(torch.load("checkpoint/STEP_{}_Encoder.torch".format(checkpoint_to_load)))
# decoder.load_state_dict(torch.load("checkpoint/STEP_{}_Decoder.torch".format(checkpoint_to_load)))

encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.01, weight_decay=0.0001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.01, weight_decay=0.0001)
loss_func = nn.CrossEntropyLoss(reduce=True, size_average=True)

class SaveHelper:
    def __init__(self, delta):
        self.global_step = 0
        self.delta = delta
        self.save_op = False if delta == -1 else True
        self.plot_loss = {"train": [], "valid": [], "train4": []}

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
    pred_note = torch.zeros(max_target_length, batch_size).long().cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        # decoder_output[:, word_dict["不"]] *= 0.6
        # decoder_output[:, word_dict["一"]] *= 0.6
        #
        # for j in range(batch_size):
        #     for i in range(t):
        #         decoder_output[j][pred_note[i][j]] *= 0.6

        loss += loss_func(decoder_output, target_batches[t])
        loss += loss_func(class_out[t], target_batches[t])

        is_teacher = random.random() < teacher_forcing_ratio
        top1 = decoder_output.max(1)[1]

        pred_note[t] = top1
        all_decoder_outputs[t] = decoder_output

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

    return float(loss), all_decoder_outputs.max(-1)[1], poem


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
    pred_note = torch.zeros(max_target_length, batch_size).long().cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        # decoder_output[:, word_dict["不"]] *= 0.6
        # decoder_output[:, word_dict["一"]] *= 0.6
        #
        # for j in range(batch_size):
        #     for i in range(t):
        #         decoder_output[j][pred_note[i][j]] *= 0.6

        loss += loss_func(decoder_output, target_batches[t])
        loss += loss_func(class_out[t], target_batches[t])

        all_decoder_outputs[t] = decoder_output
        top1 = decoder_output.max(1)[1]
        decoder_input = top1

    loss /= max_target_length

    sample_index = random.randint(0, batch_size - 1)
    poem = [input_batches.transpose(0, 1)[sample_index],
            all_decoder_outputs.max(-1)[1].transpose(0, 1)[sample_index],
            target_batches.transpose(0, 1)[sample_index]]

    encoder.train()
    decoder.train()

    return float(loss), all_decoder_outputs.max(-1)[1], poem


# train_loss = 0
for step_id in range(1, step_num + 1):
    # Get training data for this cycle
    input_batches, target_batches, res_batches = random_batch(dataset_list["train"], batch_size)

    # Run the train function
    loss, pred, poem = train(input_batches, target_batches)

    if step_id < 1000:
        output("Train: [{}/{}] loss={:.5f}".format(step_id, step_num, loss))
        poem = get_poem(poem, rev_dict)
        for sent in poem:
            for i in range(len(sent)):
                output(sent[i], end="")
            output("\t", end="")
        output("")
        output("-----------------------------")
        continue

    # four sentence
    middle = torch.zeros(1, batch_size).long().cuda()
    loss4, pred4, poem4 = train(torch.cat([input_batches, middle, pred], 0), res_batches)
    # loss4, pred4, poem4 = train(torch.cat([input_batches, pred], 0), res_batches)

    output("Train: [{}/{}] loss={:.5f} loss4={:.5f}".format(step_id, step_num, loss, loss4))

    poem = get_poem(poem, rev_dict)
    for sent in poem:
        for i in range(len(sent)):
            output(sent[i], end="")
        output("\t", end="")
    output("")
    output("-----------------------------")

    poem4 = get_poem(poem4, rev_dict)
    for sent in poem4:
        for i in range(len(sent)):
            output(sent[i], end="")
        output("\t", end="")
    output("")
    output("=============================")

    save_helper.plot_loss["train"].append(loss)
    save_helper.plot_loss["train4"].append(loss4)

    if step_id % eval_delta == 0:
        total_eval_pred = []
        total_eval_gold = []

        for input_batches, target_batches, res_batches in sequential_batch(dataset_list["val"], batch_size):
            loss, pred, poem = evaluate(input_batches, target_batches)

            middle = torch.zeros(1, batch_size).long().cuda()
            loss4, pred4, poem4 = evaluate(torch.cat([input_batches, middle, pred], 0), res_batches)
            # loss4, pred4, poem4 = evaluate(torch.cat([input_batches, pred], 0), res_batches)

            output("Eval: loss={:.5f} loss4={:.5f}".format(loss, loss4))

            poem = get_poem(poem, rev_dict)
            for sent in poem:
                for i in range(len(sent)):
                    output(sent[i], end="")
                output("\t", end="")
            output("")
            output("-----------------------------")

            poem4 = get_poem(poem4, rev_dict)
            for sent in poem4:
                for i in range(len(sent)):
                    output(sent[i], end="")
                output("\t", end="")
            output("")
            output("*****************************")

            # eval_pred = torch.cat([pred, pred4[:7, :], pred4[8:, :]], 0).transpose(0, 1).cpu().detach().numpy()
            # eval_gold = torch.cat([target_batches, res_batches[:7, :], res_batches[8:, :]]).transpose(0, 1).unsqueeze(1)\
            #     .cpu().detach().numpy()
            eval_pred = torch.cat([pred, pred4], 0).transpose(0, 1).cpu().detach().numpy()
            eval_gold = torch.cat([target_batches, res_batches]).transpose(0, 1).unsqueeze(1)\
                .cpu().detach().numpy()

            total_eval_pred.append(eval_pred)
            total_eval_gold.append(eval_gold)

        total_eval_gold = np.concatenate(total_eval_gold, axis=0)
        total_eval_pred = np.concatenate(total_eval_pred, axis=0)
        bleu = cal_bleu(total_eval_pred, total_eval_gold)
        output(bleu)
        output("=============================")

    save_helper.step()
