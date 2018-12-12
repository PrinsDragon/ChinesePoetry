import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

"""Init: input_size, hidden_size, bidirectional=True, batch_first=True"""
"""Forward: vec_seq, len_seq"""


class LSTM(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, bidirectional=True, batch_first=True):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional,
                            batch_first=batch_first)

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.layer_num = 1
        self.direction_num = 2 if bidirectional else 1

    def init_hidden(self):
        # The axes semantics are (num_layers*dir_num, minibatch_size, hidden_dim)
        return (torch.zeros(self.layer_num*self.direction_num, self.batch_size, self.hidden_size).cuda(),
                torch.zeros(self.layer_num*self.direction_num, self.batch_size, self.hidden_size).cuda())

    def forward(self, vec_seq, hidden_state):
        lstm_out, hidden_state = self.lstm(vec_seq, hidden_state)
        return lstm_out, hidden_state


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, batch_size, fc_dim, hidden_size, word_vec_matrix=None,
                 bidirectional=True, batch_first=True, dropout=0.5):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if word_vec_matrix is not None:
            self.embedding.weight.data.copy_(word_vec_matrix)

        self.basic_lstm = LSTM(batch_size=batch_size, input_size=embedding_dim, hidden_size=hidden_size,
                               bidirectional=bidirectional, batch_first=batch_first)

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, vocab_size),
            nn.ReLU()
        )

        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, first):
        hidden_state = self.basic_lstm.init_hidden()
        first = self.embedding(first)

        second, hidden_state = self.basic_lstm(first, hidden_state)
        third, hidden_state = self.basic_lstm(second, hidden_state)
        fourth, hidden_state = self.basic_lstm(third, hidden_state)

        first_project = self.classifier(first)
        second_project = self.classifier(second)
        third_project = self.classifier(third)
        fourth_project = self.classifier(fourth)

        return first_project, second_project, third_project, fourth_project




