import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class EncoderRNN(nn.Module):

    def __init__(self, hidden_size, embedding, num_layers = 1,
                 droput = 0, bidirectional = False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = nn.GRU(
            input_size = hidden_size,
            hidden_size = hidden_size,
            dropout = droput,
            bidirectional = bidirectional,
            batch_first = True,
            num_layers = num_layers
        )

    def forward(self, input_seq, input_lengths, hidden = None):

        embedded = self.embedding(input_seq)
        packed = pack_padded_sequence(embedded, input_lengths, batch_first = True)
        output , hidden = self.gru(packed, hidden)
        output, _ = pad_packed_sequence(output, batch_first = True)

        return output, hidden


class ContextRNN(nn.Module):

    def __init__(self, hidden_size, num_layers = 1, dropout = 0, bidirectional = False ):

        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size = hidden_size,
            hidden_size = hidden_size,
            dropout = dropout,
            bidirectional = bidirectional,
            batch_first = True,
            num_layers = num_layers,
        )

    def forward(self, input_seq, hidden = None):

        output, hidden = self.gru(input_seq, hidden)

        return output, hidden

class HREDDecoderRNN(nn.Module):

    def __init__(self, embedding, hidden_size, context_hidden_size,
                 output_size, num_layers = 1, dropout = 0.1):

        super(HREDDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.context_hidden_size = context_hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = embedding
        self.gru = nn.GRU(input_size = hidden_size,
                          hidden_size = hidden_size,
                          num_layers = num_layers,
                          batch_first = True)
        self.embedding_linear = nn.Linear(in_features = hidden_size,
                                          out_features = hidden_size * 2)
        self.hidden_linear = nn.Linear(in_features = hidden_size,
                                       out_features = hidden_size * 2,
                                       bias = False)
        self.context_linear = nn.Linear(in_features = context_hidden_size,
                                        out_features = hidden_size * 2,
                                        bias = False)
        self.maxout = Maxout(2)
        self.out = nn.Linear(in_features = hidden_size,
                             out_features = output_size)

    def forward(self, input_step, last_hidden, context_hidden):
        embedded = self.embedding(input_step)
        embedded = F.dropout(embedded, p = self.dropout)
        rnn_out , hidden = self.gru(embedded, last_hidden)
        pre_active = self.embedding_linear(embedded.squeeze(1)) \
                    + self.hidden_linear(rnn_out.squeeze(1)) \
                    + self.context_linear(context_hidden)
        pre_active = self.maxout(pre_active)

        return self.out(pre_active), hidden

class Maxout(nn.Module):

    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def forward(self,x):

        assert x.size(-1) % self.pool_size == 0
        m, _ = x.view(*x.size()[:-1],x.size(-1)// self.pool_size,self.pool_size).max(-1)
        return m


def l2_pooling(hiddens, src_len):
    return torch.stack(
        [
            torch.sqrt(
                torch.sum(torch.pow(hiddens[b][:src_len[b]], 2), dim=0)
                /src_len[b].type(torch.FloatTensor).cuda()
            )
            for b in range(hiddens.size(0))
        ]
    )


