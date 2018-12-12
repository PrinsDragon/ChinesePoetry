import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, sentence, hidden):
        embedded = self.embedding(sentence).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size).cuda()


# class Decoder(nn.Module):
#     def __init__(self, hidden_size, vocab_size):
#         super(Decoder, self).__init__()
#         self.hidden_size = hidden_size
#
#         self.embedding = nn.Embedding(vocab_size, hidden_size)
#         self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
#         self.out = nn.Linear(hidden_size, vocab_size)
#         self.softmax = nn.LogSoftmax(dim=1)
#
#     def forward(self, word, hidden):
#         output = self.embedding(word).view(1, 1, -1)
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#         output = self.softmax(self.out(output[0]))
#         return output, hidden
#
#     def init_hidden(self):
#         return torch.zeros(1, 1, self.hidden_size).cuda()


class Decoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, dropout=0.1, seq_length=8):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.seq_length = seq_length

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.seq_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, word, hidden, encoder_outputs):
        embedded = self.embedding(word).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size).cuda()

