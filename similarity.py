import numpy as np
from collections import Counter
import copy
import math
import matplotlib.pyplot as plt
import pickle as pkl
import random
from scipy.signal import savgol_filter
from scipy import stats


from data import get_data, split_data, get_vocabulary, process_data, get_incidental_data, get_kgram_auxilary, \
    get_kgram_knowledge
from utils import set_random_seed, generate_sent_seqs


def noisy(x):
    return 1 - (x * np.log(36) - x * np.log(x) - (1-x) * np.log(1-x))/np.log(37)


def noisy3(x):
    return 1 - (x * np.log(2) - x * np.log(x) - (1 - x) * np.log(1 - x)) / np.log(3)


def noisy2(x):
    return 1 - (- x * np.log(x) - (1 - x) * np.log(1 - x)) / np.log(2)


def sqrt_version(x):
    return np.sqrt(x)


def ratio(x):
    return (x - 0.38) / (0.61 - 0.38)


class myseq:
    empty_char = 'U'

    def __init__(self,seq):
        self.seq = list(seq)
        self.original_seq = copy.deepcopy(self.seq)
        self.n = len(self.seq)

    def maskerInit(self, seed):
        self.toMask = np.random.RandomState(seed=seed).permutation(self.n)
        self.toMaskId = 0

    def maskerReset(self):
        self.toMaskId = 0
        self.seq = copy.deepcopy(self.original_seq)

    def maskNext(self):
        if self.toMaskId >= self.n:
            print("masker already finished. did nothing here.")
            return
        self.seq[self.toMask[self.toMaskId]] = myseq.empty_char
        self.toMaskId += 1

    def count(self):
        # bb,ii,oo [t]: #possibilities ending at place t-1 with B/I/O. t=0,1,...,len(seq)
        bb = 0
        ii = 0
        oo = 1
        for step in range(0, self.n):
            bb_prv = bb
            ii_prv = ii
            oo_prv = oo
            bb = bb_prv+oo_prv+ii_prv
            ii = bb_prv+ii_prv
            oo = bb
            if self.seq[step] != myseq.empty_char:
                if self.seq[step] != 'B':
                    bb = 0
                if self.seq[step] != 'I':
                    ii = 0
                if self.seq[step] != 'O':
                    oo = 0
        return bb+ii+oo

    def early_stop(self,seed):
        self.maskerInit(seed)
        allCnt = [self.count()]
        for step in range(self.n):
            self.maskNext()
            if step in [3, 7, 10, 14, 18]:
                allCnt.append(self.count())
            else:
                allCnt.append(1)
        return allCnt


class genSeq:
    def __init__(self, length, seed):
        self.len = length
        self.seed = seed
        random.seed(seed)

    def nextSeq(self):
        seq = ['O']
        for i in range(self.len):
            r = random.random()
            if seq[-1] == 'O':
                if r < 0.5:
                    seq.append('B')
                else:
                    seq.append('O')
            else:
                if r < 1.0/3:
                    seq.append('B')
                elif r < 2.0/3:
                    seq.append('I')
                else:
                    seq.append('O')
        return seq[1:]

    def reset(self):
        random.seed(self.seed)


def get_bio_constraint_similarity():
    NumExp = 1000
    NumSeq = 1000
    seqlength = 19
    seqgen = genSeq(seqlength, 0)

    allIk = []
    for seqid in range(NumSeq):
        seq = seqgen.nextSeq()
        allCnt = []
        for seed in range(0, NumExp):
            a = myseq(seq)
            allCnt += a.early_stop(seed)
            # print(allCnt)
        allCnt = np.array(allCnt).reshape((NumExp, a.n + 1))
        Ik = np.mean(np.log(allCnt), axis=0)
        Ik = Ik[::-1]
        allIk = np.concatenate((allIk, Ik))
    allIk = np.array(allIk).reshape((NumSeq, seqlength + 1))
    # print(allIk)
    full_size = seqlength * np.log(3)
    similarity = np.mean(1 - allIk / full_size, axis=0)
    print('similarity', similarity)


def get_stats():
    config = {'unknown_freq': 2, 'gold_ratio': 0.1, 'inc_option': 'auxiliary', 'auxiliary_option': 'detection', 'seed': 66}
    dir_path = '/path/to/working/dir'
    set_random_seed(config['seed'])
    train_file = dir_path + '/data/ontonotes.development.ner'
    print('load data')
    train_data = get_data(train_file)
    gold_data, inc_data = split_data(train_data, config)
    print('get vocabulary')
    word_to_ix, pos_to_ix, ner_to_ix = get_vocabulary(train_data, config)
    config['ner_to_ix'] = ner_to_ix
    config['pos_to_ix'] = pos_to_ix
    config['word_to_ix'] = word_to_ix
    config['output_size'] = len(ner_to_ix)
    print('ner_to_ix', ner_to_ix)
    print('word_to_ix', len(word_to_ix))
    print('process data')
    inc_input_ids, inc_sent_ids, inc_pos_ids, inc_ner_ids = process_data(inc_data, word_to_ix, pos_to_ix, ner_to_ix)
    inc_ner_ids = get_incidental_data(inc_sent_ids, inc_input_ids, inc_pos_ids, inc_ner_ids, config)
    inc_label_counter = Counter()
    for label in inc_ner_ids:
        # if label[0] == 'B' or label[0] == 'I':
        #    label = label[2:]
        inc_label_counter[label] += 1 / len(inc_ner_ids)
    print('inc label counter', inc_label_counter)
    inputs, sent_ids, pos_labels, ner_labels = inc_data
    word_seqs = generate_sent_seqs(inputs, sent_ids)
    pos_seqs = generate_sent_seqs(pos_labels, sent_ids)
    ner_seqs = generate_sent_seqs(ner_labels, sent_ids)
    inc_data = []
    sent_counter = Counter()
    for x in range(len(word_seqs)):
        inc_data.append((word_seqs[x], pos_seqs[x], ner_seqs[x]))
        sent_counter[len(word_seqs[x])] += 1 / len(word_seqs)
    print('average sent length', len(sent_ids) / len(word_seqs))
    print('sent length distribution', sent_counter.items())
    # print('kgram for pos')
    # get_kgram_knowledge(inc_data, max_k=5, option='pos')
    # print('kgram for ner')
    # get_kgram_knowledge(inc_data, max_k=5, option='ner')
    # print('pos kgram for ner')
    # get_kgram_auxilary(inc_data, max_k=5, option='pos-ner')


def compute_correlation():
    X_p = [0.8, 0.6, 0.4, 0.2]
    Y_p = [0.91, 0.78, 0.61, 0.30]
    X_n = [0.81, 0.66, 0.53, 0.42, 0.31, 0.22, 0.14]
    Y_n = [0.91, 0.83, 0.74, 0.52, 0.35, 0.22, 0.09]
    X_pn = [0.65, 0.53, 0.42, 0.34, 0.49, 0.40, 0.32, 0.25, 0.32, 0.26, 0.21, 0.17]
    Y_pn = [0.87, 0.78, 0.65, 0.48, 0.78, 0.65, 0.52, 0.34, 0.61, 0.43, 0.35, 0.22]
    x_a = [0.90, 0.24]
    y_a = [0.70, 0.61]
    x_k = [0.60, 0.43, 0.67]
    y_k = [0.26, 0.17, 0.57]
    x_c = [0.14, 0.34, 0.53, 0.67, 0.84]
    # y_c = [0.09, 0.35, 0.65, 0.83, 0.96]
    y_c = [0.13, 0.48, 0.78, 0.91, 1.04]
    X = []
    Y = []
    X.extend(X_p)
    X.extend(X_n)
    X.extend(X_pn)
    X.extend(x_a)
    # X.extend(x_k)
    X.extend(x_c)
    Y.extend(Y_p)
    Y.extend(Y_n)
    Y.extend(Y_pn)
    Y.extend(y_a)
    # Y.extend(y_k)
    Y.extend(y_c)
    X = np.sqrt(np.array(X))
    print('spearmanr correlation', stats.spearmanr(X, Y))
    print('kendalltau correlation', stats.kendalltau(X, Y))
    print('perason correlation', stats.pearsonr(X, Y))


if __name__ == '__main__':
    # for x in range(10)[1:]:
    #    print(noisy(x * 0.1))
    # performance = [0.59, 0.58, 0.56, 0.53, 0.49, 0.56, 0.53, 0.50, 0.46, 0.52, 0.48, 0.46, 0.43]
    # for x in performance:
    #    print(ratio(x))
    # get_stats()
    # get_bio_constraint_similarity()
    compute_correlation()
