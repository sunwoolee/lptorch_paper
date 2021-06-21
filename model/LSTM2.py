import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lptorch as lp

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = lp.nn.NQLayer(nn.Dropout(dropout))
        self.encoder = nn.Embedding(ntoken, ninp)
        self.qinput = self.qinput = lp.nn.QLayer()
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(lp.nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            raise ValueError( """An invalid option for `--model` was supplied,
                                options are ['LSTM', 'GRU']""")
        self.decoder = lp.nn.QLayer(nn.Linear(nhid, ntoken), lp.nn.F.sub_max_rnn, last=True, tracking=[True, False])

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.module.weight)
        nn.init.uniform_(self.decoder.module.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.qinput(self.encoder(input)))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
