# import torchtext.vocab as vocab
import numpy as np
import random
import copy
from collections import Counter

from utils import generate_sent_seqs


def get_kgram_knowledge(data, max_k=1, option='pos', unique_ratio=1.0, freq_gate=2):
    print('option', option)
    k_gram_list = []
    for k in range(max_k + 1)[1:]:
        print('k', k)
        kgram_labels = {}
        kgram_freq = {}
        for (word_seq, pos_seq, ner_seq) in data:
            n = len(word_seq)
            for i in range(n - k + 1):
                cur_gram = " ".join(word_seq[i: i + k])
                if option == 'pos':
                    cur_label = " ".join(pos_seq[i: i + k])
                elif option == 'ner':
                    cur_label = " ".join(ner_seq[i: i + k])
                if cur_gram in kgram_labels:
                    kgram_labels[cur_gram].add(cur_label)
                    kgram_freq[cur_gram] += 1
                else:
                    kgram_labels[cur_gram] = set([cur_label])
                    kgram_freq[cur_gram] = 1
        labels_num = np.array([len(list(x)) for x in list(kgram_labels.values())])
        freq_num = np.array(list(kgram_freq.values()))
        total_num = np.sum(freq_num)
        kgram_num = len(kgram_labels)
        min_label_num = np.min(labels_num)
        max_label_num = np.max(labels_num)
        average_label_num = np.sum(labels_num * freq_num) / total_num
        unique_percent = np.sum((labels_num == 1) * freq_num) / total_num
        unique_freq_gate_percent = np.sum((labels_num == 1) * (freq_num >= freq_gate) * freq_num) / total_num
        print('total num', total_num)
        print('kgram num', kgram_num)
        print('min label num', min_label_num)
        print('max label num',  max_label_num)
        print('average label num', average_label_num)
        print('unique percentage', unique_percent)
        print('unique freq gate percentage', unique_freq_gate_percent)
        k_gram_list.append((total_num, kgram_num, min_label_num, max_label_num, average_label_num, unique_percent))
        inc_labels = []
        for (word_seq, pos_seq, ner_seq) in data:
            n = len(word_seq)
            inc_seq = ['U'] * n
            for i in range(n - k + 1):
                cur_gram = " ".join(word_seq[i: i + k])
                if kgram_freq[cur_gram] >= freq_gate and len(kgram_labels[cur_gram]) == 1:
                    inc_seq[i: i + k] = ner_seq[i: i + k]
            inc_labels.extend(inc_seq)
        print('partial unk rate', np.mean(np.array(inc_labels) == 'U'))
        label_counter = Counter()
        total_labeled = np.sum(np.array(inc_labels) != 'U')
        for inc_label in inc_labels:
            if inc_label != 'U':
                label_counter[inc_label] += 1 / total_labeled
        print('label counter', label_counter)
        if unique_percent > unique_ratio:
            break
    return k_gram_list


def get_kgram_auxilary(data, max_k=1, option='pos-ner', unique_ratio=1.0, freq_gate=2):
    print('option', option)
    k_gram_list = []
    for k in range(max_k + 1)[1:]:
        print('k', k)
        kgram_labels = {}
        kgram_freq = {}
        for (word_seq, pos_seq, ner_seq) in data:
            n = len(word_seq)
            for i in range(n - k + 1):
                if option == 'pos-ner':
                    cur_gram = " ".join(pos_seq[i: i + k])
                    cur_label = " ".join(ner_seq[i: i + k])
                elif option == 'ner-pos':
                    cur_gram = " ".join(ner_seq[i: i + k])
                    cur_label = " ".join(pos_seq[i: i + k])
                if cur_gram in kgram_labels:
                    kgram_labels[cur_gram].add(cur_label)
                    kgram_freq[cur_gram] += 1
                else:
                    kgram_labels[cur_gram] = set([cur_label])
                    kgram_freq[cur_gram] = 1
        labels_num = np.array([len(list(x)) for x in list(kgram_labels.values())])
        freq_num = np.array(list(kgram_freq.values()))
        total_num = np.sum(freq_num)
        kgram_num = len(kgram_labels)
        min_label_num = np.min(labels_num)
        max_label_num = np.max(labels_num)
        average_label_num = np.sum(labels_num * freq_num) / total_num
        unique_percent = np.sum((labels_num == 1) * freq_num) / total_num
        unique_freq_gate_percent = np.sum((labels_num == 1) * (freq_num >= freq_gate) * freq_num) / total_num
        print('total num', total_num)
        print('kgram num', kgram_num)
        print('min label num', min_label_num)
        print('max label num',  max_label_num)
        print('average label num', average_label_num)
        print('unique percentage', unique_percent)
        print('unique freq gate percentage', unique_freq_gate_percent)
        k_gram_list.append((total_num, kgram_num, min_label_num, max_label_num, average_label_num, unique_percent))
        inc_labels = []
        for (word_seq, pos_seq, ner_seq) in data:
            n = len(word_seq)
            inc_seq = ['U'] * n
            for i in range(n - k + 1):
                if option == 'pos-ner':
                    cur_gram = " ".join(pos_seq[i: i + k])
                elif option == 'ner-pos':
                    cur_gram = " ".join(ner_seq[i: i + k])
                if kgram_freq[cur_gram] >= freq_gate and len(kgram_labels[cur_gram]) == 1:
                    inc_seq[i: i + k] = ner_seq[i: i + k]
            inc_labels.extend(inc_seq)
        print('pos for ner partial unk rate', np.mean(np.array(inc_labels) == 'U'))
        label_counter = Counter()
        total_labeled = np.sum(np.array(inc_labels) != 'U')
        for inc_label in inc_labels:
            if inc_label != 'U':
                label_counter[inc_label] += 1 / total_labeled
        print('label counter', label_counter)
        if unique_percent > unique_ratio:
            break
    return k_gram_list


def get_data(file):
    data = []
    fin = open(file)
    lines = fin.readlines()
    word_seq = []
    pos_seq = []
    ner_seq = []
    vocabulary = set()
    word_vocabulary = set()
    pos_vocabulary = set()
    ner_vocabulary = set()
    word_num = 0
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            if len(word_seq) != 0:
                data.append((word_seq, pos_seq, ner_seq))
                word_seq = []
                pos_seq = []
                ner_seq = []
        else:
            items = line.split()
            word_seq.append(items[0])
            pos_seq.append(items[1])
            ner_seq.append(items[3])
            word_num += 1
            for c in items[0]:
                vocabulary.add(c)
            for c in items[1]:
                vocabulary.add(c)
            for c in items[2]:
                vocabulary.add(c)
            word_vocabulary.add(items[0])
            pos_vocabulary.add(items[1])
            ner_vocabulary.add(items[3])
    fin.close()
    # data = data[:100]
    print('vocabulary', len(vocabulary), vocabulary)
    print('sentence num', len(data))
    print('word num', word_num)
    print('word vocabulary', len(word_vocabulary))
    print('pos vocabulary', len(pos_vocabulary))
    print('ner vocabulary', len(ner_vocabulary))
    inputs = []
    sent_ids = []
    pos_labels = []
    ner_labels = []
    for sent_id in range(len(data)):
        word_seq, pos_seq, ner_seq = data[sent_id]
        inputs.extend(word_seq)
        sent_ids.extend([sent_id] * len(word_seq))
        pos_labels.extend(pos_seq)
        ner_labels.extend(ner_seq)
    return inputs, sent_ids, pos_labels, ner_labels


def split_data(data, config):
    inputs, sent_ids, pos_labels, ner_labels = data
    input_seqs = generate_sent_seqs(inputs, sent_ids)
    pos_seqs = generate_sent_seqs(pos_labels, sent_ids)
    ner_seqs = generate_sent_seqs(ner_labels, sent_ids)
    seq_ids = np.arange(len(input_seqs))
    np.random.shuffle(seq_ids)
    gold_seq_ids = seq_ids[: int(len(seq_ids) * config['gold_ratio'])]
    inc_seq_ids = seq_ids[int(len(seq_ids) * config['gold_ratio']):]
    gold_inputs = []
    gold_sent_ids = []
    gold_pos_labels = []
    gold_ner_labels = []
    inc_inputs = []
    inc_sent_ids = []
    inc_pos_labels = []
    inc_ner_labels = []
    for x in range(len(gold_seq_ids)):
        seq_id = gold_seq_ids[x]
        gold_inputs.extend(input_seqs[seq_id])
        gold_sent_ids.extend([x] * len(input_seqs[seq_id]))
        gold_pos_labels.extend(pos_seqs[seq_id])
        gold_ner_labels.extend(ner_seqs[seq_id])
    for x in range(len(inc_seq_ids)):
        seq_id = inc_seq_ids[x]
        inc_inputs.extend(input_seqs[seq_id])
        inc_sent_ids.extend([x] * len(input_seqs[seq_id]))
        inc_pos_labels.extend(pos_seqs[seq_id])
        inc_ner_labels.extend(ner_seqs[seq_id])
    gold_data = gold_inputs, gold_sent_ids, gold_pos_labels, gold_ner_labels
    inc_data = inc_inputs, inc_sent_ids, inc_pos_labels, inc_ner_labels
    return gold_data, inc_data


def get_vocabulary(data, config):
    inputs, sent_ids, pos_labels, ner_labels = data
    word_to_count = {}
    pos_to_count = {}
    ner_to_count = {}
    for word in inputs:
        if word in word_to_count:
            word_to_count[word] += 1
        else:
            word_to_count[word] = 1
    for pos in pos_labels:
        if pos in pos_to_count:
            pos_to_count[pos] += 1
        else:
            pos_to_count[pos] = 1
    for ner in ner_labels:
        if ner in ner_to_count:
            ner_to_count[ner] += 1
        else:
            ner_to_count[ner] = 1
    word_to_ix = {'UNKNOWN': 0, 'START_MINUS_ONE': 1, 'START_MINUS_TWO': 2, 'END_PLUS_ONE': 3, 'END_PLUS_TWO': 4}
    pos_to_ix = {'AFX': 0}
    ner_to_ix = {'O': 0}
    unknown_ratio = 0
    for word in word_to_count.keys():
        if word_to_count[word] >= config['unknown_freq']:
            word_to_ix[word] = len(word_to_ix)
        else:
            unknown_ratio += word_to_count[word]
    unknown_ratio /= len(inputs)
    print('unknown word ratio', unknown_ratio)
    for pos in pos_to_count.keys():
        pos_to_ix[pos] = len(pos_to_ix)
    for ner in ner_to_count.keys():
        if ner not in ner_to_ix:
            if ner[0] == 'B':
                ner_to_ix[ner] = len(ner_to_ix)
                ner_to_ix['I'+ner[1:]] = len(ner_to_ix)
            elif ner[0] == 'I':
                ner_to_ix['B'+ner[1:]] = len(ner_to_ix)
                ner_to_ix[ner] = len(ner_to_ix)
    return word_to_ix, pos_to_ix, ner_to_ix


def get_vocab_embeddings(word_to_ix):
    glove = vocab.GloVe(name='840B', dim=300)
    ix_to_word = {}
    for word in word_to_ix.keys():
        ix_to_word[word_to_ix[word]] = word
    embeddings = []
    random_num = 0
    for ix in range(len(ix_to_word)):
        word = ix_to_word[ix]
        if word in glove.stoi:
            embeddings.append(glove.vectors[glove.stoi[word]].cpu().data.numpy().tolist())
        elif word.lower() in glove.stoi:
            embeddings.append(glove.vectors[glove.stoi[word.lower()]].cpu().data.numpy().tolist())
        else:
            embeddings.append(list(np.random.rand(300)))
            random_num += 1
    print('random embeddings', random_num)
    embeddings = np.array(embeddings)
    return embeddings


def process_data(data, word_to_ix, pos_to_ix, ner_to_ix):
    inputs, sent_ids, pos_labels, ner_labels = data
    input_ids = []
    for word in inputs:
        if word in word_to_ix:
            input_ids.append(word_to_ix[word])
        else:
            input_ids.append(word_to_ix["UNKNOWN"])
    pos_ids = [pos_to_ix[pos] for pos in pos_labels]
    ner_ids = [ner_to_ix[ner] for ner in ner_labels]
    return input_ids, sent_ids, pos_ids, ner_ids


def get_word_features(input_ids, sent_ids, vocab_embeddings):
    word_seqs = generate_sent_seqs(input_ids, sent_ids)
    input_features = []
    for seq in word_seqs:
        seq = [2, 1] + seq + [3, 4]
        seq = [seq[x - 2: x + 3] for x in range(len(seq))[2:-2]]
        seq_features = [list(vocab_embeddings[x].reshape(-1)) for x in seq]
        input_features.extend(seq_features)
    assert len(input_ids) == len(input_features)
    assert len(input_features[0]) == 5 * vocab_embeddings.shape[1]
    return input_features


def combine_two_datasets(data_one, data_two):
    inputs_one, sent_ids_one, labels_one, confidences_one = \
        data_one['inputs'], data_one['sent_ids'], data_one['labels'], data_one['confidences']
    inputs_two, sent_ids_two, labels_two, confidences_two = \
        data_two['inputs'], data_two['sent_ids'], data_two['labels'], data_two['confidences']
    inputs, sent_ids, labels, confidences = [], [], [], []
    inputs.extend(inputs_one)
    inputs.extend(inputs_two)
    sent_ids.extend(sent_ids_one)
    sent_ids.extend(sent_ids_two)
    labels.extend(labels_one)
    labels.extend(labels_two)
    confidences.extend(confidences_one)
    confidences.extend(confidences_two)
    data = {'inputs': inputs, 'sent_ids': sent_ids, 'labels': labels, 'confidences': confidences}
    return data


def get_incidental_data(sent_ids, input_ids, pos_labels, ner_labels, config):
    if config['inc_option'] == 'partial':
        return get_partial_data(ner_labels, config)
    elif config['inc_option'] == 'noisy':
        return get_noisy_data(ner_labels, config)
    elif config['inc_option'] == 'partial+noisy':
        return get_partial_noisy_data(ner_labels, config)
    elif config['inc_option'] == 'auxiliary':
        return get_auxiliary_data(sent_ids, pos_labels, ner_labels, config)
    elif config['inc_option'] == 'knowledge':
        return get_knowledge_data(sent_ids, input_ids, ner_labels, config)
    elif config['inc_option'] == 'partial+constraints':
        return get_partial_data(ner_labels, config)


def get_partial_data(ner_labels, config):
    partial_ner_labels = copy.deepcopy(ner_labels)
    ids = np.arange(len(ner_labels))
    np.random.shuffle(ids)
    unknown_ids = ids[:int(len(ner_labels) * config['partial_unk_rate'])]
    for cur_id in unknown_ids:
        partial_ner_labels[cur_id] = -1
    return partial_ner_labels


def get_noisy_data(ner_labels, config):
    noisy_ner_labels = copy.deepcopy(ner_labels)
    ids = np.arange(len(ner_labels))
    np.random.shuffle(ids)
    diff_ids = ids[:int(len(ner_labels) * config['noisy_diff_rate'])]
    for cur_id in diff_ids:
        cur_ner_label = ner_labels[cur_id]
        ner_seq = list(np.arange(config['output_size']))
        ner_seq.remove(cur_ner_label)
        noisy_ner_label = random.choice(ner_seq)
        assert noisy_ner_label != cur_ner_label
        noisy_ner_labels[cur_id] = noisy_ner_label
    return noisy_ner_labels


def get_partial_noisy_data(ner_labels, config):
    partial_noisy_ner_labels = copy.deepcopy(ner_labels)
    ids = np.arange(len(ner_labels))
    # shuffle for noisy
    np.random.shuffle(ids)
    diff_ids = ids[:int(len(ner_labels) * config['noisy_diff_rate'])]
    for cur_id in diff_ids:
        cur_ner_label = ner_labels[cur_id]
        ner_seq = list(np.arange(config['output_size']))
        ner_seq.remove(cur_ner_label)
        noisy_ner_label = random.choice(ner_seq)
        assert noisy_ner_label != cur_ner_label
        partial_noisy_ner_labels[cur_id] = noisy_ner_label
    # shuffle for partial
    np.random.shuffle(ids)
    unknown_ids = ids[:int(len(ner_labels) * config['partial_unk_rate'])]
    for cur_id in unknown_ids:
        partial_noisy_ner_labels[cur_id] = -1
    return partial_noisy_ner_labels


def get_auxiliary_data(sent_ids, pos_labels, ner_labels, config):
    if config['auxiliary_option'] == 'detection':
        return get_detection_data(ner_labels, config)
    elif config['auxiliary_option'] == 'coarse':
        return get_coarse_ner_data(ner_labels, config)
    elif config['auxiliary_option'] == 'pos':
        return get_pos_data(sent_ids, pos_labels, ner_labels, config)


def get_detection_data(ner_labels, config):
    ner_to_ix = config['ner_to_ix']
    detection_labels = copy.deepcopy(ner_labels)
    ix_to_ner = {}
    for key, value in ner_to_ix.items():
        ix_to_ner[value] = key
    for cur_id in range(len(ner_labels)):
        detection_labels[cur_id] = ix_to_ner[ner_labels[cur_id]][0]
    return detection_labels


def get_coarse_ner_data(ner_labels, config):
    ner_to_ix = config['ner_to_ix']
    replacements = {'B-PERSON': 'B-PERSON',
                    'I-PERSON': 'I-PERSON',
                    'B-ORG': 'B-ORG',
                    'I-ORG': 'I-ORG',
                    'B-LOC': 'B-LOC',
                    'I-LOC': 'I-LOC',
                    'B-FAC': 'B-LOC',
                    'I-FAC': 'I-LOC',
                    'B-GPE': 'B-LOC',
                    'I-GPE': 'I-LOC',
                    'B-NORP': 'B-MISC',
                    'I-NORP': 'I-MISC',
                    'B-PRODUCT': 'B-MISC',
                    'I-PRODUCT': 'I-MISC',
                    'B-EVENT': 'B-MISC',
                    'I-EVENT': 'I-MISC',
                    'B-LANGUAGE': 'B-MISC',
                    'I-LANGUAGE': 'I-MISC',
                    'B-WORK_OF_ART': 'O',
                    'I-WORK_OF_ART': 'O',
                    'B-LAW': 'O',
                    'I-LAW': 'O',
                    'B-DATE': 'O',
                    'I-DATE': 'O',
                    'B-TIME': 'O',
                    'I-TIME': 'O',
                    'B-PERCENT': 'O',
                    'I-PERCENT': 'O',
                    'B-MONEY': 'O',
                    'I-MONEY': 'O',
                    'B-QUANTITY': 'O',
                    'I-QUANTITY': 'O',
                    'B-ORDINAL': 'O',
                    'I-ORDINAL': 'O',
                    'B-CARDINAL': 'O',
                    'I-CARDINAL': 'O',
                    'O': 'O'}
    coarse_labels = copy.deepcopy(ner_labels)
    ix_to_ner = {}
    for key, value in ner_to_ix.items():
        ix_to_ner[value] = key
    for cur_id in range(len(ner_labels)):
        coarse_labels[cur_id] = replacements[ix_to_ner[ner_labels[cur_id]]]
    return coarse_labels


def get_pos_data(sent_ids, pos_labels, ner_labels, config):
    pos_seqs = generate_sent_seqs(pos_labels, sent_ids)
    ner_seqs = generate_sent_seqs(ner_labels, sent_ids)
    k = config['k-gram']
    kgram_labels = {}
    kgram_freq = {}
    for x in range(len(pos_seqs)):
        pos_seq = pos_seqs[x]
        ner_seq = ner_seqs[x]
        pos_seq_str = [str(pos) for pos in pos_seq]
        ner_seq_str = [str(ner) for ner in ner_seq]
        n = len(pos_seq)
        for i in range(n - k + 1):
            cur_gram = " ".join(pos_seq_str[i: i + k])
            cur_label = " ".join(ner_seq_str[i: i + k])
            if cur_gram in kgram_labels:
                kgram_labels[cur_gram].add(cur_label)
                kgram_freq[cur_gram] += 1
            else:
                kgram_labels[cur_gram] = set([cur_label])
                kgram_freq[cur_gram] = 1
    inc_labels = []
    for x in range(len(pos_seqs)):
        pos_seq = pos_seqs[x]
        ner_seq = ner_seqs[x]
        pos_seq_str = [str(pos) for pos in pos_seq]
        n = len(pos_seq)
        inc_seq = [-1] * n
        for i in range(n - k + 1):
            cur_gram = " ".join(pos_seq_str[i: i + k])
            if kgram_freq[cur_gram] >= config['k-gram-freq-gate'] and len(kgram_labels[cur_gram]) == 1:
                inc_seq[i: i + k] = ner_seq[i: i + k]
        inc_labels.extend(inc_seq)

    assert len(inc_labels) == len(sent_ids)
    print('pos for ner partial unk rate', np.mean(np.array(inc_labels) == -1))
    return inc_labels


def get_knowledge_data(sent_ids, input_ids, ner_labels, config):
    word_seqs = generate_sent_seqs(input_ids, sent_ids)
    ner_seqs = generate_sent_seqs(ner_labels, sent_ids)
    k = config['k-gram']
    kgram_labels = {}
    kgram_freq = {}
    for x in range(len(word_seqs)):
        word_seq = word_seqs[x]
        ner_seq = ner_seqs[x]
        word_seq_str = [str(word) for word in word_seq]
        ner_seq_str = [str(ner) for ner in ner_seq]
        n = len(word_seq)
        for i in range(n - k + 1):
            cur_gram = " ".join(word_seq_str[i: i + k])
            cur_label = " ".join(ner_seq_str[i: i + k])
            if cur_gram in kgram_labels:
                kgram_labels[cur_gram].add(cur_label)
                kgram_freq[cur_gram] += 1
            else:
                kgram_labels[cur_gram] = set([cur_label])
                kgram_freq[cur_gram] = 1
    inc_labels = []
    for x in range(len(word_seqs)):
        word_seq = word_seqs[x]
        ner_seq = ner_seqs[x]
        word_seq_str = [str(word) for word in word_seq]
        n = len(word_seq)
        inc_seq = [-1] * n
        for i in range(n - k + 1):
            cur_gram = " ".join(word_seq_str[i: i + k])
            if kgram_freq[cur_gram] >= config['k-gram-freq-gate'] and len(kgram_labels[cur_gram]) == 1:
                inc_seq[i: i + k] = ner_seq[i: i + k]
        inc_labels.extend(inc_seq)

    assert len(inc_labels) == len(sent_ids)
    print('pos for ner partial unk rate', np.mean(np.array(inc_labels) == -1))
    return inc_labels
