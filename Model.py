import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

sound_dict_file = open("data/proc/sound_dict_s.pkl", "rb")
sound_dict_s = pickle.load(sound_dict_file)

sound_dict = sound_dict_s[0]
char2num = sound_dict_s[1]
char2final = sound_dict_s[2]

char2num[0] = 0
char2num[1] = 0
char2num[2] = 0

char2final[0] = 0
char2final[1] = 0
char2final[2] = 0

def get_sound_seqs(char_seq):
    sound_num_seq = torch.empty_like(char_seq).long().cuda()
    sound_final_seq = torch.empty_like(char_seq).long().cuda()

    if len(char_seq.size()) == 2:
        for i in range(char_seq.shape[0]):
            for j in range(char_seq.shape[1]):
                sound_num_seq[i][j] = char2num[int(char_seq[i][j])]
                sound_final_seq[i][j] = char2final[int(char_seq[i][j])]
    else:
        for i in range(char_seq.shape[0]):
            sound_num_seq[i] = char2num[int(char_seq[i])]
            sound_final_seq[i] = char2final[int(char_seq[i])]

    return sound_num_seq, sound_final_seq

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1,
                 extra_size_sound=0, extra_size_num=5, extra_dim_sound=32, extra_dim_num=32, word_matrix=None):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
        if word_matrix is None:
            init.xavier_normal_(self.embedding.weight)
        else:
            self.embedding.weight.data.copy_(word_matrix)

        self.extra_sound_embedding = nn.Embedding(extra_size_sound, extra_dim_sound)
        init.xavier_normal_(self.extra_sound_embedding.weight)
        self.extra_num_embedding = nn.Embedding(extra_size_num, extra_dim_num)
        init.xavier_normal_(self.extra_num_embedding.weight)

        self.gru = nn.GRU(hidden_size+extra_dim_sound+extra_dim_num,
                          hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

        self.classifier = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, input_size))

    def forward(self, input_seqs, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)

        num_seqs, sound_seqs = get_sound_seqs(input_seqs)

        sound_embedded = self.extra_sound_embedding(sound_seqs)
        num_embedded = self.extra_num_embedding(num_seqs)

        joint_embedded = torch.cat([embedded, sound_embedded, num_embedded], -1)

        # outputs, hidden = self.gru(embedded, hidden)
        outputs, hidden = self.gru(joint_embedded, hidden)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs

        class_output = self.classifier(outputs)

        return outputs, hidden, class_output

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = torch.zeros(this_batch_size, max_len).cuda()  # B x S

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.mm(encoder_output.t())
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.mm(energy.t())
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.mm(energy.t())
            return energy

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1,
                 extra_size_sound=0, extra_size_num=5, extra_dim_sound=32, extra_dim_num=32):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        init.xavier_normal_(self.embedding.weight)

        self.extra_sound_embedding = nn.Embedding(extra_size_sound, extra_dim_sound)
        init.xavier_normal_(self.extra_sound_embedding.weight)
        self.extra_num_embedding = nn.Embedding(extra_size_num, extra_dim_num)
        init.xavier_normal_(self.extra_num_embedding.weight)

        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size+extra_dim_num+extra_dim_sound, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)

        num_seqs, sound_seqs = get_sound_seqs(input_seq)

        sound_embedded = self.extra_sound_embedding(sound_seqs)
        num_embedded = self.extra_num_embedding(num_seqs)
        embedded = torch.cat([embedded, sound_embedded, num_embedded], -1)

        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, -1)  # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights
