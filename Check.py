import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn

from DataStructure import get_poem, build_rev_dict, cal_bleu, sequential_batch
from Model import EncoderRNN, LuongAttnDecoderRNN

word_dict_file = open("data/proc/dict.pkl", "rb")
word_dict = pickle.load(word_dict_file)
rev_dict = build_rev_dict(word_dict)
word_matrix = torch.load("data/proc/matrix.torch")

sound_dict_file = open("data/proc/sound_dict_s.pkl", "rb")
sound_dict_s = pickle.load(sound_dict_file)

sound_dict = sound_dict_s[0]
char2num = sound_dict_s[1]
char2final = sound_dict_s[2]

dataset_name = ["test", "val"]
dataset_file = [open("data/proc/{}_proc.pkl".format(n), "rb") for n in dataset_name]
dataset_list = {n: pickle.load(f) for n, f in zip(dataset_name, dataset_file)}

seq_length = 7
vocabulary_size = len(word_dict)
embedding_dim = 128

extra_dim_sound = 32
extra_dim_num = 32

hidden_size = 128
teacher_forcing_ratio = 0.8
batch_size = 50
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

encoder4 = EncoderRNN(input_size=vocabulary_size, hidden_size=hidden_size, n_layers=layer_num,
                      extra_size_sound=len(sound_dict), extra_size_num=5,
                      extra_dim_sound=extra_dim_sound, extra_dim_num=extra_dim_num, word_matrix=word_matrix,
                      ).cuda()
decoder4 = LuongAttnDecoderRNN(attn_model=attention_model, hidden_size=hidden_size, output_size=vocabulary_size,
                               n_layers=layer_num, extra_size_sound=len(sound_dict), extra_size_num=5,
                               extra_dim_sound=extra_dim_sound, extra_dim_num=extra_dim_num).cuda()

checkpoint_to_load = 4700
encoder.load_state_dict(torch.load("checkpoint/STEP_{}_Encoder.torch".format(checkpoint_to_load)))
decoder.load_state_dict(torch.load("checkpoint/STEP_{}_Decoder.torch".format(checkpoint_to_load)))
checkpoint_to_load = 900
encoder4.load_state_dict(torch.load("checkpoint/STEP_{}_Encoder.torch".format(checkpoint_to_load)))
decoder4.load_state_dict(torch.load("checkpoint/STEP_{}_Decoder.torch".format(checkpoint_to_load)))

loss_func = nn.CrossEntropyLoss(reduce=True, size_average=True)

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
        # loss += loss_func(class_out[t], target_batches[t])

        all_decoder_outputs[t] = decoder_output
        top1 = decoder_output.max(1)[1]
        decoder_input = top1

        pred_note[t] = decoder_input

    loss /= max_target_length

    sample_index = random.randint(0, batch_size - 1)
    poem = [input_batches.transpose(0, 1)[sample_index],
            all_decoder_outputs.max(-1)[1].transpose(0, 1)[sample_index],
            target_batches.transpose(0, 1)[sample_index]]

    encoder.train()
    decoder.train()

    return float(loss), all_decoder_outputs.max(-1)[1], poem

def evaluate4(input_batches, target_batches):
    encoder4.eval()
    decoder4.eval()

    # Run words through encoder
    encoder_outputs, encoder_hidden, class_out = encoder4(input_batches, None)

    loss = 0.

    # Prepare input and output variables
    decoder_input = torch.Tensor([SOS] * batch_size).long().cuda()
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = target_batches.shape[0]
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size).cuda()
    pred_note = torch.zeros(max_target_length, batch_size).long().cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder4(
            decoder_input, decoder_hidden, encoder_outputs
        )

        # decoder_output[:, word_dict["不"]] *= 0.6
        # decoder_output[:, word_dict["一"]] *= 0.6
        #

        # if t == 8:
        #     decoder_output[:, word_dict["一"]] *= 0.8
        #
        # for j in range(batch_size):
        #     for i in range(t):
        #         decoder_output[j][pred_note[i][j]] *= 0.6

        loss += loss_func(decoder_output, target_batches[t])
        # loss += loss_func(class_out[t], target_batches[t])

        all_decoder_outputs[t] = decoder_output
        top1 = decoder_output.max(1)[1]
        decoder_input = top1

        pred_note[t] = decoder_input

    loss /= max_target_length

    sample_index = random.randint(0, batch_size - 1)
    poem = [input_batches.transpose(0, 1)[sample_index],
            all_decoder_outputs.max(-1)[1].transpose(0, 1)[sample_index],
            target_batches.transpose(0, 1)[sample_index]]

    encoder4.train()
    decoder4.train()

    return float(loss), all_decoder_outputs.max(-1)[1], poem

def test(input_batches, batch_size):
    encoder.eval()
    decoder.eval()

    # Run words through encoder
    encoder_outputs, encoder_hidden, class_out = encoder(input_batches, None)

    # Prepare input and output variables
    decoder_input = torch.Tensor([SOS] * batch_size).long().cuda()
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = input_batches.shape[0]
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size).cuda()
    pred_note = torch.zeros(max_target_length, batch_size).long().cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        # decoder_output[:, word_dict["不"]] *= 0.6
        # decoder_output[:, word_dict["一"]] *= 0.6

        for j in range(batch_size):
            for i in range(t):
                decoder_output[j][pred_note[i][j]] *= 0.6

        all_decoder_outputs[t] = decoder_output
        top1 = decoder_output.max(1)[1]
        decoder_input = top1

        pred_note[t] = decoder_input

    sample_index = 0
    poem = [input_batches.transpose(0, 1)[sample_index],
            all_decoder_outputs.max(-1)[1].transpose(0, 1)[sample_index]]

    return all_decoder_outputs.max(-1)[1], poem

def test4(input_batches, batch_size):
    encoder.eval()
    decoder.eval()

    # Run words through encoder
    encoder_outputs, encoder_hidden, class_out = encoder4(input_batches, None)

    # Prepare input and output variables
    decoder_input = torch.Tensor([SOS] * batch_size).long().cuda()
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = input_batches.shape[0]
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size).cuda()
    pred_note = torch.zeros(max_target_length, batch_size).long().cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder4(
            decoder_input, decoder_hidden, encoder_outputs
        )

        decoder_output[:, word_dict["不"]] *= 0.6
        decoder_output[:, word_dict["一"]] *= 0.6
        decoder_output[:, word_dict["无"]] *= 0.9

        # if t == 6:
        #     decoder_output[:, word_dict["事"]] *= 0.6

        if t == 8:
            decoder_output[:, word_dict["一"]] *= 0.8

        for j in range(batch_size):
            for i in range(t):
                decoder_output[j][pred_note[i][j]] *= 0.6

        all_decoder_outputs[t] = decoder_output
        top1 = decoder_output.max(1)[1]
        decoder_input = top1

        pred_note[t] = decoder_input

    sample_index = 0
    poem = [input_batches.transpose(0, 1)[sample_index],
            all_decoder_outputs.max(-1)[1].transpose(0, 1)[sample_index]]

    return all_decoder_outputs.max(-1)[1], poem


# total_eval_pred = []
# total_eval_gold = []
#
# for input_batches, target_batches, res_batches in sequential_batch(dataset_list["val"], batch_size):
#     loss, pred, poem = evaluate(input_batches, target_batches)
#
#     middle = torch.zeros(1, batch_size).long().cuda()
#     loss4, pred4, poem4 = evaluate4(torch.cat([input_batches, middle, pred], 0), res_batches)
#     # loss4, pred4, poem4 = evaluate(torch.cat([input_batches, pred], 0), res_batches)
#
#     print("Eval: loss={:.5f} loss4={:.5f}".format(loss, loss4))
#
#     poem = get_poem(poem, rev_dict)
#     for sent in poem:
#         for i in range(len(sent)):
#             print(sent[i], end="")
#         print("\t", end="")
#     print("")
#     print("-----------------------------")
#
#     poem4 = get_poem(poem4, rev_dict)
#     for sent in poem4:
#         for i in range(len(sent)):
#             print(sent[i], end="")
#         print("\t", end="")
#     print("")
#     print("*****************************")
#
#     # eval_pred = torch.cat([pred, pred4[:7, :], pred4[8:, :]], 0).transpose(0, 1).cpu().detach().numpy()
#     # eval_gold = torch.cat([target_batches, res_batches[:7, :], res_batches[8:, :]]).transpose(0, 1).unsqueeze(1)\
#     #     .cpu().detach().numpy()
#     eval_pred = torch.cat([pred, pred4], 0).transpose(0, 1).cpu().detach().numpy()
#     eval_gold = torch.cat([target_batches, res_batches]).transpose(0, 1).unsqueeze(1)\
#         .cpu().detach().numpy()
#
#     total_eval_pred.append(eval_pred)
#     total_eval_gold.append(eval_gold)
#
# total_eval_gold = np.concatenate(total_eval_gold, axis=0)
# total_eval_pred = np.concatenate(total_eval_pred, axis=0)
# bleu = cal_bleu(total_eval_pred, total_eval_gold)
# print(bleu)
# print("=============================")

output_file = open("data/output_{}.txt".format(time.strftime('%H-%M-%S', time.localtime(time.time()))), "w", encoding="utf-8")

for i, input_batches in enumerate(sequential_batch(dataset_list["test"], 1, test=True)):
    pred, poem = test(input_batches, batch_size=1)

    middle = torch.zeros(1, 1).long().cuda()
    pred4, poem4 = test4(torch.cat([input_batches, middle, pred], 0), batch_size=1)

    poem4 = get_poem(poem4, rev_dict)

    first = "".join(poem4[0][0: 7])
    second = "".join(poem4[0][8: 15])
    third = "".join(poem4[1][0: 7])
    fourth = "".join(poem4[1][8: 15])

    print("{}\t{}\t{}".format(second, third, fourth), file=output_file)
    print("{}: {}\t{}\t{}\t{}".format(i, first, second, third, fourth))

