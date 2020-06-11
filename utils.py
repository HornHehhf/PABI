import numpy as np
import math
import torch
import random


def set_random_seed(seed):
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_minibatches_idx(n, minibatch_size, shuffle=True):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(math.ceil(n / minibatch_size)):
        minibatches.append(idx_list[minibatch_start: minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    return minibatches


def generate_sent_seqs(input_ids, sent_ids):
    word_seqs = []
    seq = [input_ids[0]]
    for x in range(len(sent_ids))[1:]:
        if sent_ids[x] == sent_ids[x - 1]:
            seq.append(input_ids[x])
        else:
            word_seqs.append(seq)
            seq = [input_ids[x]]
    word_seqs.append(seq)
    return word_seqs
