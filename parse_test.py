import sys
import torch
import argparse
import hashlib
import os
import numpy as np
from data import Corpus
from torch.optim import SGD, Adam
from splitcross import SplitCrossEntropyLoss
from model import RNNModel
from parse_arg import *
from utils import batchify, get_batch, repackage_hidden

filename = "15620575173148172.pt"

fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if args.philly:
    fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

ntokens = len(corpus.dictionary)

#initialize the model
model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.chunk_size, args.nlayers,
                       args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)


with open(filename,"rb") as f:
    model, criterion, optimizer = torch.load(f)

#prepare data
eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

def idx2text(index):
    global corpus
    text = [corpus.dictionary.idx2word[idx] for idx in index]
    text = " ".join(text)
    return text

# get model prepared
hidden = model.init_hidden(args.batch_size)
hidden = repackage_hidden(hidden)
bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
# Prevent excessively small or negative sequence lengths
seq_len = max(5, int(np.random.normal(bptt, 5)))
data, targets = get_batch(train_data, 10, args, seq_len=seq_len)
output, hidden, distances = model(data, hidden, return_d=True)

texts = [idx2text(idx) for idx in data[:,]]
np.save("output.npy",output.cpu().data.numpy())
np.save("distances.npy",distances.cpu().data.numpy())
torch.save(texts,"texts.pt")