# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
from io import open

from model.LSTM2 import RNNModel, RNNModel_origin
from utils import *
import lptorch as lp
import pdb

# fp8_format = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,1,0]
# fp8_format = [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,2,1,0]
fp8_format = [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1,1,1,1,0]
# log_format = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
log_format = [1,1,1,1,1,1,1,0]
lp.set_activ_quant(lp.quant.quant(lp.quant.custom_fp_format(fp8_format), room=1))
lp.set_error_quant(lp.quant.quant(lp.quant.custom_fp_format(fp8_format), room=1))
# lp.set_weight_quant(lp.quant.quant(lp.quant.custom_fp_format(fp8_format), room=0, ch_wise=False))
# lp.set_weight_quant(lp.quant.quant(lp.quant.custom_fp_format(log_format), room=0, ch_wise=True))
lp.set_weight_quant(lp.quant.quant(lp.quant.linear_format(6), room=0, ch_wise=True))
lp.set_grad_quant(lp.quant.quant(lp.quant.custom_fp_format(fp8_format), room=2))
lp.set_master_quant(lp.quant.quant(lp.quant.fp_format(exp_bit=6, man_bit=9), stochastic=True))
lp.set_scale_fluctuate(False)
# lp.set_hysteresis_update(False)

# python word_language_model.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# folder_name = 'LSTM2_baseline'
# folder_name = 'LSTM2_baseline_HFP8_scheme'
# folder_name = 'LSTM2_activ_FP152'
# folder_name = 'LSTM2_activ_FP143'
# folder_name = 'LSTM2_activ_LFP8''

# folder_name = 'LSTM2_train_FP4_w_hysteresis'
# folder_name = 'LSTM2_train_FP4_wo_hysteresis'
# folder_name = 'LSTM2_train_INT4_w_hysteresis'
# folder_name = 'LSTM2_train_INT4_wo_hysteresis'
folder_name = 'LSTM2_train_INT6_w_hysteresis'
# folder_name = 'LSTM2_train_INT6_wo_hysteresis'
# folder_name = 'LSTM2_train_INT8_w_hysteresis'
# folder_name = 'LSTM2_train_INT8_wo_hysteresis'
# folder_name = 'LSTM2_train_LFP8_w_hysteresis'
# folder_name = 'LSTM2_train_LFP8_wo_hysteresis'

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='../data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='./checkpoint',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
optimizer = lp.optim.SGD(model.parameters(), lr=float(args.lr), weight_decay=0., momentum=0.)
criterion = nn.NLLLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def set_scale():
    # print('calculating initial scale...')
    model.train()
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        if batch != 0 and batch % 10 == 0:
            print(str(batch)+'%..', end='', flush=True)
        if batch == 100:
            print('')
            return
        data, targets = get_batch(train_data, i)
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

def set_scale_eval():
    # print('calculating initial scale...')
    model.eval()
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        if batch != 0 and batch % 10 == 0:
            print(str(batch)+'%..', end='', flush=True)
        if batch == 100:
            print('')
            return
        data, targets = get_batch(train_data, i)
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)


def evaluate(data_source, epoch=None):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    loss = total_loss / (len(data_source) - 1)
    if epoch is None:
        with open(os.path.join(args.save, folder_name, 'test_result.txt'), 'w') as f:
            f.write('loss is '+ str(loss)+ '\tppl is '+ str(math.exp(loss)))
    else:
        save_test_status([epoch, loss, math.exp(loss)], folder_name)
    
    return loss


def train(epoch):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    top_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(p.grad, alpha=-lr)
        optimizer.step(lr=lr)

        total_loss += loss.item()
        top_loss += loss.item()
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        
        if args.dry_run:
            break
    top_loss /= (len(train_data) // args.bptt)
    save_train_status([epoch, top_loss, math.exp(top_loss)], folder_name)


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

lr_epoch = [11, 16, 26, 31, 37]

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        set_scale()
        # model.decoder.qblock.scale1.mul_(0).add_(4)
        train(epoch)
        set_scale_eval()
        val_loss = evaluate(val_data, epoch)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(os.path.join(args.save, folder_name, 'best_model.pt'), 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        # else:
        #     # Anneal the learning rate if no improvement has been seen in the validation dataset.
        #     lr /= 4.0
        if epoch in lr_epoch:
            lr /= 4.0


        # elif epoch < 6:
        #     lr /= 4.0
        
        # if epoch == 6:
        #     lr = 1
        # elif epoch > 6:
        #     lr = lr * 0.8

        state = {
            'net':model.state_dict(),
            'val_loss':val_loss,
            'epoch':epoch,
            'lr':lr,
        }
        save_checkpoint(state, folder_name, epoch)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(os.path.join(args.save, folder_name, 'best_model.pt'), 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
set_scale_eval()
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
