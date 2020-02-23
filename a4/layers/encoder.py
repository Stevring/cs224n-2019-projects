import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, bidir):
        super(Encoder, self).__init__()
        self.num_dir = 2 if bidir else 1
        self.hidden_size = hidden_size // self.num_dir
        self.bigru = nn.GRU(input_size, self.hidden_size, 1, dropout=dropout, bidirectional=bidir)

    def forward(self, x, length):
        packed = nn.utils.rnn.pack_padded_sequence(x, length)
        packed_output, h_n = self.bigru(packed)
        # output: (seq_len, batch, num_directions * hidden_size)
        # h_n : (num_layers * num_directions, batch, hidden_size)
        output = nn.utils.rnn.pad_packed_sequence(packed_output)[0]
        return output, h_n[1]
