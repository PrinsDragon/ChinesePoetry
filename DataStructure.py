import random

import torch
from nltk.translate.bleu_score import corpus_bleu
from torch.autograd import Variable
from torch.nn import functional

PAD = 0
SOS = 1
EOS = 2


# START = 0
# END = 1
#
# # TODO: add END to each sentence of the dataset
#
# class PoetryDataSet(Dataset):
#     def __init__(self, poem, mode="simple"):
#         super(PoetryDataSet, self).__init__()
#         self.poem = poem
#         self.poem_num = len(poem)
#
#     def __len__(self):
#         return self.poem_num
#
#     def __getitem__(self, index):
#         return torch.Tensor(self.poem[index][0] + [END]).long().cuda(), \
#                torch.Tensor(self.poem[index][1] + [END]).long().cuda(), \
#                torch.Tensor(self.poem[index][2] + [END]).long().cuda(), \
#                torch.Tensor(self.poem[index][3] + [END]).long().cuda(),

def build_rev_dict(dictionary):
    rev_dict = {}
    for word in dictionary:
        cur_id = dictionary[word]
        rev_dict[cur_id] = word
    return rev_dict


def get_poem(poem_list, rev_dict):
    ret = []
    for i, sent in enumerate(poem_list):
        ret.append([])
        for w in sent:
            ret[i].append(rev_dict[int(w)])
    return ret


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.Tensor([i for i in range(max_len)]).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    length = Variable(torch.Tensor(length).long()).cuda()

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


def random_batch(pairs, batch_size=50):
    input_seqs = []
    target_seqs = []
    res_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)

        input_seqs.append(pair[0])
        target_seqs.append(pair[1])

        res_seqs.append(pair[2] + [PAD] + pair[3])
        # res_seqs.append(pair[2] + pair[3])

        # input_seqs.append(pair[2])
        # target_seqs.append(pair[3])

    input_var = torch.Tensor(input_seqs).long().transpose(0, 1).cuda()
    target_var = torch.Tensor(target_seqs).long().transpose(0, 1).cuda()
    res_var = torch.Tensor(res_seqs).long().transpose(0, 1).cuda()

    return input_var, target_var, res_var


def sequential_batch(pairs, batch_size=50, test=False):
    times = len(pairs) // batch_size
    middle = torch.zeros(batch_size, 1)
    # for i in range(times):
    #     pair = torch.Tensor(pairs[batch_size * i: batch_size * (i + 1)])
    #     yield torch.cat([pair[:, 0, :], pair[:, 1, :]], 1).long().transpose(0, 1).cuda(), \
    #           torch.cat([pair[:, 2, :], pair[:, 3, :]], 1).long().transpose(0, 1).cuda()

    if test:
        for i in range(times):
            pair = torch.Tensor(pairs[batch_size * i: batch_size * (i + 1)])
            yield pair[:, 0, :].long().transpose(0, 1).cuda()
    else:
        for i in range(times):
            pair = torch.Tensor(pairs[batch_size * i: batch_size * (i + 1)])
            yield pair[:, 0, :].long().transpose(0, 1).cuda(), \
                  pair[:, 1, :].long().transpose(0, 1).cuda(), \
                  torch.cat([pair[:, 2, :], middle, pair[:, 3, :]], 1).long().transpose(0, 1).cuda()
    # for i in range(times):
    #     pair = torch.Tensor(pairs[batch_size * i: batch_size * (i + 1)])
    #     yield pair[:, 2, :].long().transpose(0, 1).cuda(), \
    #           pair[:, 3, :].long().transpose(0, 1).cuda()


def cal_bleu(pred, gold):
    one_gram = corpus_bleu(gold, pred, weights=(1, 0, 0, 0))
    two_gram = corpus_bleu(gold, pred, weights=(0.5, 0.5, 0, 0))
    three_gram = corpus_bleu(gold, pred, weights=(0.33, 0.33, 0.33, 0))
    four_gram = corpus_bleu(gold, pred, weights=(0.25, 0.25, 0.25, 0.25))

    return one_gram, two_gram, three_gram, four_gram

# pred = torch.Tensor([[1, 2]]).numpy()
# gold = torch.Tensor([[[1, 2]]]).numpy()
# t = cal_bleu(pred, gold)
# pass
