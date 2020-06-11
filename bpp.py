import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.optim as optim
from collections import Counter
from seqeval.metrics import classification_report
import sys

from utils import get_minibatches_idx, generate_sent_seqs, set_random_seed
from data import get_data, get_vocabulary, get_vocab_embeddings, process_data, get_word_features, split_data, \
    combine_two_datasets, get_incidental_data
from models import MLPNet


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def predict(test_inputs, model, config):
    # test inputs: (T, D)
    model.eval()
    pred_np = []
    minibatches_idx = get_minibatches_idx(len(test_inputs), minibatch_size=config['test_batch_size'], shuffle=False)
    for minibatch in minibatches_idx:
        inputs = torch.Tensor(np.array([test_inputs[x] for x in minibatch]))
        inputs = Variable(inputs.cuda().squeeze())
        # (B, D)
        outputs = model(inputs)
        pred_np.extend(list(np.exp(outputs.cpu().data.numpy())))
        # (B, C)
    pred_np = np.array(pred_np)
    # (T, C)
    return pred_np


def evaluate(test_data, model, ner_to_ix, config):
    ix_to_ner = {}
    for ner in ner_to_ix.keys():
        ix_to_ner[ner_to_ix[ner]] = ner
    test_inputs = test_data['inputs']
    sent_ids = test_data['sent_ids']
    test_targets = test_data['labels']
    pred_np = predict(test_inputs, model, config)
    if 'constraint_option' in config and config['constraint_option'] == 'BIO':
        inc_labels = [-1] * len(sent_ids)
        sent_probs_list = generate_sent_seqs(pred_np.tolist(), sent_ids)
        sent_inc_labels_list = generate_sent_seqs(inc_labels, sent_ids)
        predictions = []
        for x in range(len(sent_probs_list)):
            sent_probs = sent_probs_list[x]
            sent_inc_labels = sent_inc_labels_list[x]
            probs = get_combined_partial_probs(sent_probs, sent_inc_labels)
            sent_preds, sent_confidences = BIO_inference(probs)
            predictions.extend(sent_preds)
        predictions = np.array(predictions)
        assert len(predictions) == len(inc_labels)
        test_predictions = np.array(predictions)
    else:
        test_predictions = np.argmax(pred_np, axis=1)
    gold_labels = [ix_to_ner[ner] for ner in test_targets]
    pred_labels = [ix_to_ner[ner] for ner in test_predictions]
    gold_labels = generate_sent_seqs(gold_labels, sent_ids)
    pred_labels = generate_sent_seqs(pred_labels, sent_ids)
    # test_accuracy = np.sum(test_predictions == test_targets)
    # print(gold_labels)
    # print(pred_labels)
    test_accuracy = classification_report(gold_labels, pred_labels)
    return test_accuracy


def train(train_data, dev_data, model, loss_function, optimizer,  ner_to_ix, config):
    total_loss_list = []
    for epoch in range(config['epoch_num']):
        model.train()
        print('current epoch: ', epoch, end='\r\n')
        total_loss = 0
        minibatches_idx = get_minibatches_idx(len(train_data['inputs']), minibatch_size=config['train_batch_size'],
                                              shuffle=True)
        for minibatch in minibatches_idx:
            inputs = torch.Tensor(np.array([train_data['inputs'][x] for x in minibatch]))
            targets = torch.Tensor(np.array([train_data['labels'][x] for x in minibatch]))
            confidences = torch.Tensor(np.array([train_data['confidences'][x] for x in minibatch]))
            inputs, targets = Variable(inputs.cuda()).squeeze(), Variable(targets.cuda()).squeeze().long()
            confidences = Variable(confidences.cuda(), requires_grad=False).squeeze()
            # inputs: (B, d), targets: B, confidences: B
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            # print('loss', loss)
            loss = torch.sum(loss * confidences)
            total_loss += loss
            loss.backward()
            optimizer.step()
        total_loss_list.append(total_loss.cpu().data.item())
        print('train loss', total_loss)
        # train_accuracy = evaluate(train_data, model, ner_to_ix, config)
        # print('train accuracy', train_accuracy)
        dev_accuracy = evaluate(dev_data, model, ner_to_ix, config)
        print('dev accuracy', dev_accuracy)


def get_noisy_prob(noisy_labels, noise_ratio, label_num):
    noisy_prob = np.ones((len(noisy_labels), label_num)) * noise_ratio / (label_num - 1)
    # (T, C)
    for x in range(len(noisy_labels)):
        noisy_prob[x][noisy_labels[x]] = 1 - noise_ratio
    return noisy_prob


def get_auxiliary_masks(config):
    ner_to_ix = config['ner_to_ix']
    auxiliary_masks = {}
    if config['auxiliary_option'] == 'detection':
        mask_B = [0] * len(ner_to_ix)
        mask_I = [0] * len(ner_to_ix)
        mask_O = [0] * len(ner_to_ix)
        for x in range(len(ner_to_ix)):
            if x % 2 == 1:
                mask_B[x] = 1
        for x in range(len(ner_to_ix))[1:]:
            if x % 2 == 0:
                mask_I[x] = 1
        mask_O[0] = 1
        auxiliary_masks['B'] = mask_B
        auxiliary_masks['I'] = mask_I
        auxiliary_masks['O'] = mask_O
    elif config['auxiliary_option'] == 'coarse':
        mask_B_PER = [0] * len(ner_to_ix)
        mask_I_PER = [0] * len(ner_to_ix)
        mask_B_ORG = [0] * len(ner_to_ix)
        mask_I_ORG = [0] * len(ner_to_ix)
        mask_B_LOC = [0] * len(ner_to_ix)
        mask_I_LOC = [0] * len(ner_to_ix)
        mask_B_MISC = [0] * len(ner_to_ix)
        mask_I_MISC = [0] * len(ner_to_ix)
        mask_O = [0] * len(ner_to_ix)
        mask_B_PER[ner_to_ix['B-PERSON']] = 1
        mask_I_PER[ner_to_ix['I-PERSON']] = 1
        mask_B_ORG[ner_to_ix['B-ORG']] = 1
        mask_I_ORG[ner_to_ix['I-ORG']] = 1
        mask_B_LOC[ner_to_ix['B-LOC']] = 1
        mask_I_LOC[ner_to_ix['I-LOC']] = 1
        mask_B_LOC[ner_to_ix['B-FAC']] = 1
        mask_I_LOC[ner_to_ix['I-FAC']] = 1
        mask_B_LOC[ner_to_ix['B-GPE']] = 1
        mask_I_LOC[ner_to_ix['I-GPE']] = 1
        mask_B_MISC[ner_to_ix['B-NORP']] = 1
        mask_I_MISC[ner_to_ix['I-NORP']] = 1
        mask_B_MISC[ner_to_ix['B-PRODUCT']] = 1
        mask_I_MISC[ner_to_ix['I-PRODUCT']] = 1
        mask_B_MISC[ner_to_ix['B-EVENT']] = 1
        mask_I_MISC[ner_to_ix['I-EVENT']] = 1
        mask_B_MISC[ner_to_ix['B-LANGUAGE']] = 1
        mask_I_MISC[ner_to_ix['I-LANGUAGE']] = 1
        mask_O[ner_to_ix['B-WORK_OF_ART']] = 1
        mask_O[ner_to_ix['I-WORK_OF_ART']] = 1
        mask_O[ner_to_ix['B-LAW']] = 1
        mask_O[ner_to_ix['I-LAW']] = 1
        mask_O[ner_to_ix['B-DATE']] = 1
        mask_O[ner_to_ix['I-DATE']] = 1
        mask_O[ner_to_ix['B-TIME']] = 1
        mask_O[ner_to_ix['I-TIME']] = 1
        mask_O[ner_to_ix['B-PERCENT']] = 1
        mask_O[ner_to_ix['I-PERCENT']] = 1
        mask_O[ner_to_ix['B-MONEY']] = 1
        mask_O[ner_to_ix['I-MONEY']] = 1
        mask_O[ner_to_ix['B-QUANTITY']] = 1
        mask_O[ner_to_ix['I-QUANTITY']] = 1
        mask_O[ner_to_ix['B-ORDINAL']] = 1
        mask_O[ner_to_ix['I-ORDINAL']] = 1
        mask_O[ner_to_ix['B-CARDINAL']] = 1
        mask_O[ner_to_ix['I-CARDINAL']] = 1
        mask_O[ner_to_ix['O']] = 1
        auxiliary_masks['B-PERSON'] = mask_B_PER
        auxiliary_masks['I-PERSON'] = mask_I_PER
        auxiliary_masks['B-ORG'] = mask_B_ORG
        auxiliary_masks['I-ORG'] = mask_I_ORG
        auxiliary_masks['B-LOC'] = mask_B_LOC
        auxiliary_masks['I-LOC'] = mask_I_LOC
        auxiliary_masks['B-MISC'] = mask_B_MISC
        auxiliary_masks['I-MISC'] = mask_I_MISC
        auxiliary_masks['O'] = mask_O
    return auxiliary_masks


def get_combined_partial_probs(sent_probs, sent_inc_labels):
    min_inf = -1000000000
    n = len(sent_probs)
    c = len(sent_probs[0])
    probs = np.ones((n, c)) * min_inf
    for x in range(n):
        if sent_inc_labels[x] == -1:
            probs[x] = np.log(sent_probs[x])
        else:
            probs[x][sent_inc_labels[x]] = 0
    # print('probs', probs)
    return probs


def BIO_inference(probs):
    min_inf = -1000000000
    n = len(probs)
    c = len(probs[0])
    trans_matrix = np.zeros((c, c))
    for t in range(c):
        if t != 0 and t % 2 == 0:
            for s in range(c):
                if not (s == t - 1 or s == t):
                    trans_matrix[s][t] = min_inf
    scores = np.zeros((n, c))
    scores_index = np.zeros((n, c), dtype=int)
    scores[0] = probs[0]
    for x in range(n)[1:]:
        for t in range(c):
            max_value = scores[x-1, 0] + trans_matrix[0, t]
            max_index = 0
            for s in range(c)[1:]:
                if scores[x-1, s] + trans_matrix[s, t] > max_value:
                    max_value = scores[x-1, s] + trans_matrix[s, t]
                    max_index = s
            scores[x, t] = max_value + probs[x, t]
            scores_index[x, t] = max_index
    labels = [0] * n
    confidences = [0.0] * n
    labels[n - 1] = np.argmax(scores[n - 1])
    confidences[n - 1] = np.exp(np.max(scores[n - 1]))
    for x in range(n)[:-1][::-1]:
        labels[x] = scores_index[x + 1, labels[x + 1]]
        confidences[x] = np.exp(probs[x, labels[x]])
    # print('scores', scores)
    # print('scores indexes', scores_index)
    # print('probs', probs)
    '''
    if np.mean(np.array(labels) == np.argmax(probs, axis=1)) < 1:
        print('max predictions violate the BIO constraints')
        print('labels', labels)
        print('confidences', confidences)
        print('max predictions', np.argmax(probs, axis=1))
        print('max confidences', np.exp(np.max(probs, axis=1)))
    '''
    return labels, confidences


def inference(pred_np, sentence_ids, inc_labels, config):
    # inputs: pred_np (T, C), sentence_ids (T, C)
    # inc_option: unlabeled, partial, noisy, auxiliary, constraints, knowledge
    inc_option = config['inc_option']
    if inc_option == 'partial':
        # inc_labels: partial labels T
        predictions = np.argmax(pred_np, axis=1)
        confidences = np.max(pred_np, axis=1)
        for x in range(len(inc_labels)):
            if inc_labels[x] != -1:
                predictions[x] = inc_labels[x]
                confidences[x] = 1.0
    elif inc_option == 'noisy':
        # inc_labels: noisy labels T
        noisy_prob = get_noisy_prob(inc_labels, config['noisy_diff_rate'], config['output_size'])
        pred_np += noisy_prob * config['noisy_lambda']
        predictions = np.argmax(pred_np, axis=1)
        confidences = np.max(pred_np, axis=1)
    elif inc_option == 'constraints':
        if config['constraint_option'] == 'BIO':
            inc_labels = [-1] * len(sentence_ids)
            sent_probs_list = generate_sent_seqs(pred_np.tolist(), sentence_ids)
            sent_inc_labels_list = generate_sent_seqs(inc_labels, sentence_ids)
            predictions = []
            confidences = []
            for x in range(len(sent_probs_list)):
                sent_probs = sent_probs_list[x]
                sent_inc_labels = sent_inc_labels_list[x]
                probs = get_combined_partial_probs(sent_probs, sent_inc_labels)
                sent_preds, sent_confidences = BIO_inference(probs)
                predictions.extend(sent_preds)
                confidences.extend(sent_confidences)
            predictions = np.array(predictions)
            confidences = np.array(confidences)
            assert len(predictions) == len(inc_labels)
            assert len(confidences) == len(inc_labels)
        # a list of sentence with (S, C)
    elif inc_option == 'partial+noisy':
        noisy_prob = get_noisy_prob(inc_labels, config['noisy_diff_rate'], config['output_size'])
        for x in range(len(inc_labels)):
            if inc_labels[x] == -1:
                noisy_prob[x] = np.ones(config['output_size']) * 1 / config['output_size']
        pred_np += noisy_prob * config['noisy_lambda']
        predictions = np.argmax(pred_np, axis=1)
        confidences = np.max(pred_np, axis=1)
    elif inc_option == 'auxiliary':
        if config['auxiliary_option'] == 'detection':
            auxiliary_masks = get_auxiliary_masks(config)
            # print('auxiliary masks', auxiliary_masks)
            detection_masks = np.array([auxiliary_masks[label] for label in inc_labels])
            # pred_np = (pred_np * detection_masks) / (np.sum(pred_np * detection_masks, axis=1).reshape(-1, 1))
            # print('detection masks', detection_masks[:100])
            pred_np = pred_np * detection_masks
            predictions = np.argmax(pred_np, axis=1)
            confidences = np.max(pred_np, axis=1)
        elif config['auxiliary_option'] == 'coarse':
            auxiliary_masks = get_auxiliary_masks(config)
            # print('auxiliary masks', auxiliary_masks)
            coarse_masks = np.array([auxiliary_masks[label] for label in inc_labels])
            # print('coarse masks', coarse_masks[:100])
            # pred_np = (pred_np * coarse_masks) / (np.sum(pred_np * coarse_masks, axis=1).reshape(-1, 1))
            pred_np = pred_np * coarse_masks
            predictions = np.argmax(pred_np, axis=1)
            confidences = np.max(pred_np, axis=1)
        elif config['auxiliary_option'] == 'pos':
            pos_prob = np.zeros((len(inc_labels), config['output_size']))
            for x in range(len(inc_labels)):
                if inc_labels[x] == -1:
                    pos_prob[x] = np.ones(config['output_size']) * 1 / config['output_size']
                else:
                    pos_prob[x][inc_labels[x]] = 1.0
            pred_np += pos_prob * config['inc_lambda']
            predictions = np.argmax(pred_np, axis=1)
            confidences = np.max(pred_np, axis=1)
            # for x in range(len(inc_labels)):
            #     if inc_labels[x] != -1:
            #         predictions[x] = inc_labels[x]
            #         confidences[x] = 1.0
    elif inc_option == 'knowledge':
        know_prob = np.zeros((len(inc_labels), config['output_size']))
        for x in range(len(inc_labels)):
            if inc_labels[x] == -1:
                know_prob[x] = np.ones(config['output_size']) * 1 / config['output_size']
            else:
                know_prob[x][inc_labels[x]] = 1.0
        pred_np += know_prob * config['inc_lambda']
        predictions = np.argmax(pred_np, axis=1)
        confidences = np.max(pred_np, axis=1)
        # for x in range(len(inc_labels)):
        #     if inc_labels[x] != -1:
        #         predictions[x] = inc_labels[x]
        #         confidences[x] = 1.0
    elif inc_option == 'partial+constraints':
        if config['constraint_option'] == 'BIO':
            sent_probs_list = generate_sent_seqs(pred_np.tolist(), sentence_ids)
            sent_inc_labels_list = generate_sent_seqs(inc_labels, sentence_ids)
            predictions = []
            confidences = []
            for x in range(len(sent_probs_list)):
                sent_probs = sent_probs_list[x]
                sent_inc_labels = sent_inc_labels_list[x]
                probs = get_combined_partial_probs(sent_probs, sent_inc_labels)
                sent_preds, sent_confidences = BIO_inference(probs)
                predictions.extend(sent_preds)
                confidences.extend(sent_confidences)
            predictions = np.array(predictions)
            confidences = np.array(confidences)
            assert len(predictions) == len(inc_labels)
            assert len(confidences) == len(inc_labels)
    else:
        predictions = np.argmax(pred_np, axis=1)
        confidences = np.max(pred_np, axis=1)

    return predictions, confidences


def bootstrap(gold_data, inc_data, dev_data, model, loss_function, optimizer, ner_to_ix, config):
    # initialize the model with labeled data
    print('init the model with the gold data')
    config['epoch_num'] = config['init_epoch_num']
    train(gold_data, dev_data, model, loss_function, optimizer, ner_to_ix, config)
    # iteration
    for i in range(config['iter_num']):
        print('bootstrap iter num', i)
        inc_pred_np = predict(inc_data['inputs'], model, config)
        inc_predictions, inc_confidences = inference(inc_pred_np, inc_data['sent_ids'], inc_data['labels'], config)
        # print('inc predictions', np.min(inc_predictions), np.max(inc_predictions), np.mean(inc_predictions))
        # print('inc confidences', np.min(inc_confidences), np.max(inc_confidences), np.mean(inc_confidences))
        silver_data = {'inputs': inc_data['inputs'], 'sent_ids': inc_data['sent_ids'], 'labels': inc_predictions,
                       'confidences': inc_confidences}
        extend_data = combine_two_datasets(gold_data, silver_data)
        if i == 0:
            config['epoch_num'] = config['init_epoch_num']
        else:
            config['epoch_num'] = config['iter_epoch_num']
        train(extend_data, dev_data, model, loss_function, optimizer, ner_to_ix, config)
        # train on extended data


def build_model(config):
    model = MLPNet(config['input_size'], config['hidden_size'], config['output_size'])
    loss_function = nn.NLLLoss(reduce=False)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    return model, loss_function, optimizer


def run_experiments(config):
    dir_path = '/path/to/working/dir'
    train_file = dir_path + '/data/ontonotes.development.ner'
    dev_file = dir_path + '/data/ontonotes.development.ner'
    test_file = dir_path + '/data/ontonotes.test.ner'
    # test_file = dir_path + '/data/ontonotes.development.ner'
    # train_input_features_path = dir_path + '/data/train_input_features.pickle'
    # dev_input_features_path = dir_path + '/data/dev_input_features.pickle'
    # test_input_features_path = dir_path + '/data/test_input_features.pickle'
    model_path = dir_path + '/models/MLPNet_' + config['para_option'] + '.pt'
    print('load data')
    train_data = get_data(train_file)
    gold_data, inc_data = split_data(train_data, config)
    dev_data = get_data(dev_file)
    test_data = get_data(test_file)
    print('get vocabulary and embeddings')
    word_to_ix, pos_to_ix, ner_to_ix = get_vocabulary(train_data, config)
    # ner_to_ix = {'O': 0, 'B-LOC': 1, 'I-LOC': 2, 'B-GPE': 3, 'I-GPE': 4, 'B-QUANTITY': 5, 'I-QUANTITY': 6,
    # 'B-NORP': 7, 'I-NORP': 8, 'B-ORG': 9, 'I-ORG': 10, 'B-PERSON': 11, 'I-PERSON': 12, 'B-DATE': 13, 'I-DATE': 14,
    # 'B-CARDINAL': 15, 'I-CARDINAL': 16, 'B-ORDINAL': 17, 'I-ORDINAL': 18, 'B-PRODUCT': 19, 'I-PRODUCT': 20,
    # 'B-WORK_OF_ART': 21, 'I-WORK_OF_ART': 22, 'B-FAC': 23, 'I-FAC': 24, 'B-EVENT': 25, 'I-EVENT': 26,
    # 'B-LANGUAGE': 27, 'I-LANGUAGE': 28, 'B-PERCENT': 29, 'I-PERCENT': 30, 'B-MONEY': 31, 'I-MONEY': 32, 'B-TIME': 33,
    # 'I-TIME': 34, 'B-LAW': 35, 'I-LAW': 36}
    config['ner_to_ix'] = ner_to_ix
    config['pos_to_ix'] = pos_to_ix
    config['word_to_ix'] = word_to_ix
    config['output_size'] = len(ner_to_ix)
    print('ner_to_ix', ner_to_ix)
    vocab_embeddings = get_vocab_embeddings(word_to_ix)
    print('process data')
    # train_input_ids, train_sent_ids, train_pos_ids, train_ner_ids = process_data(train_data, word_to_ix, pos_to_ix,
    #                                                                             ner_to_ix)
    gold_input_ids, gold_sent_ids, gold_pos_ids, gold_ner_ids = process_data(gold_data, word_to_ix, pos_to_ix,
                                                                             ner_to_ix)
    inc_input_ids, inc_sent_ids, inc_pos_ids, inc_ner_ids = process_data(inc_data, word_to_ix, pos_to_ix, ner_to_ix)
    dev_input_ids, dev_sent_ids, dev_pos_ids, dev_ner_ids = process_data(dev_data, word_to_ix, pos_to_ix, ner_to_ix)
    test_input_ids, test_sent_ids, test_pos_ids, test_ner_ids = process_data(test_data, word_to_ix, pos_to_ix,
                                                                             ner_to_ix)
    # print('get train input features')
    # train_input_features = get_word_features(train_input_ids, train_sent_ids, vocab_embeddings)
    print('get gold input features')
    gold_input_features = get_word_features(gold_input_ids, gold_sent_ids, vocab_embeddings)
    print('get inc input features')
    inc_input_features = get_word_features(inc_input_ids, inc_sent_ids, vocab_embeddings)
    print('get dev input features')
    dev_input_features = get_word_features(dev_input_ids, dev_sent_ids, vocab_embeddings)
    print('get test input features')
    test_input_features = get_word_features(test_input_ids, test_sent_ids, vocab_embeddings)
    # print('save input features')
    # pickle.dump(train_input_features, open(train_input_features_path, 'wb'))
    # pickle.dump(dev_input_features, open(dev_input_features_path, 'wb'))
    # pickle.dump(test_input_features, open(test_input_features_path, 'wb'))

    # print('load input features')
    # train_input_features = pickle.load(open(train_input_features_path, 'rb'))
    # dev_input_features = pickle.load(open(dev_input_features_path, 'rb'))
    # test_input_features = pickle.load(open(test_input_features_path, 'rb'))

    # print('split data')
    # gold_data, inc_data = split_data(train_input_features, train_sent_ids, train_pos_ids, train_ner_ids)
    gold_data = {'inputs': gold_input_features, 'sent_ids': gold_sent_ids, 'labels': gold_ner_ids, 'confidences':
        [1.0] * len(gold_input_features)}
    dev_data = {'inputs': dev_input_features, 'sent_ids': dev_sent_ids, 'labels': dev_ner_ids, 'confidences':
        [1.0] * len(dev_input_features)}
    test_data = {'inputs': test_input_features, 'sent_ids': test_sent_ids, 'labels': test_ner_ids, 'confidences':
        [1.0] * len(test_input_features)}
    print('gold words', len(gold_input_features))
    print('inc words', len(inc_input_features))
    print('dev words', len(dev_input_features))
    print('test words', len(test_input_features))
    if config['inc_option'] == 'labeled':
        # inc_data = {'inputs': inc_data[0], 'sent_ids': inc_data[1], 'labels': inc_data[3], 'confidences':
        #     [1.0] * len(inc_data[0])}
        inc_data = {'inputs': inc_input_features, 'sent_ids': inc_sent_ids, 'labels': inc_ner_ids, 'confidences':
            [1.0] * len(inc_input_features)}
        train_data = combine_two_datasets(gold_data, inc_data)
    elif config['inc_option'] == 'none':
        train_data = gold_data
    elif config['inc_option'] == 'unlabeled' or config['inc_option'] == 'constraints':
        inc_data = {'inputs': inc_input_features, 'sent_ids': inc_sent_ids, 'labels': None, 'confidences': None}
    else:
        inc_ner_ids = get_incidental_data(inc_sent_ids, inc_input_ids, inc_pos_ids, inc_ner_ids, config)
        print('inc ner ids', inc_ner_ids[:100])
        inc_data = {'inputs': inc_input_features, 'sent_ids': inc_sent_ids, 'labels': inc_ner_ids, 'confidences': None}
    # elif config['inc_option'] == 'partial':
    #     inc_ner_ids = get_incidental_data(inc_ner_ids, config)
    #     print('inc ner ids', inc_ner_ids[:100])
    #     inc_data = {'inputs': inc_input_features, 'sent_ids': inc_sent_ids, 'labels': inc_ner_ids, 'confidences': None}
    # elif config['inc_option'] == 'noisy':
    #     inc_ner_ids = get_incidental_data(inc_ner_ids, config)
    #     print('inc ner ids', inc_ner_ids[:100])
    #     inc_data = {'inputs': inc_input_features, 'sent_ids': inc_sent_ids, 'labels': inc_ner_ids, 'confidences': None}
    # elif config['inc_option'] == 'partial+noisy':
    #     inc_ner_ids = get_incidental_data(inc_ner_ids, config)
    #     print('inc ner ids', inc_ner_ids[:100])
    #     inc_data = {'inputs': inc_input_features, 'sent_ids': inc_sent_ids, 'labels': inc_ner_ids, 'confidences': None}
    print('build model')
    model, loss_function, optimizer = build_model(config)
    if config['inc_option'] == 'labeled' or config['inc_option'] == 'none':
        print('train model')
        print('train words', len(train_data['inputs']))
        config['epoch_num'] = config['init_epoch_num']
        train(train_data, dev_data, model, loss_function, optimizer, ner_to_ix, config)
    else:
        print('bootstrap')
        bootstrap(gold_data, inc_data, dev_data, model, loss_function, optimizer, ner_to_ix, config)
    print('save model')
    torch.save(model.state_dict(), model_path)
    print('load model')
    model.load_state_dict(torch.load(model_path))
    print('test model')
    test_accuracy = evaluate(test_data, model, ner_to_ix, config)
    print('test accuracy', test_accuracy)


def run_test_experiments(config):
    dir_path = '/path/to/working/dir'
    train_file = dir_path + '/data/ontonotes.development.ner'
    test_file = dir_path + '/data/ontonotes.test.ner'
    model_path = dir_path + '/models/MLPNet_' + config['para_option'] + '.pt'
    print('load data')
    train_data = get_data(train_file)
    test_data = get_data(test_file)
    print('get vocabulary and embeddings')
    word_to_ix, pos_to_ix, ner_to_ix = get_vocabulary(train_data, config)
    config['ner_to_ix'] = ner_to_ix
    config['pos_to_ix'] = pos_to_ix
    config['word_to_ix'] = word_to_ix
    config['output_size'] = len(ner_to_ix)
    print('ner_to_ix', ner_to_ix)
    vocab_embeddings = get_vocab_embeddings(word_to_ix)
    print('process data')
    test_input_ids, test_sent_ids, test_pos_ids, test_ner_ids = process_data(test_data, word_to_ix, pos_to_ix,
                                                                             ner_to_ix)
    print('get test input features')
    test_input_features = get_word_features(test_input_ids, test_sent_ids, vocab_embeddings)
    test_data = {'inputs': test_input_features, 'sent_ids': test_sent_ids, 'labels': test_ner_ids, 'confidences':
        [1.0] * len(test_input_features)}
    print('test words', len(test_input_features))
    print('build model')
    model, loss_function, optimizer = build_model(config)
    print('load model')
    model.load_state_dict(torch.load(model_path))
    print('test model')
    test_accuracy = evaluate(test_data, model, ner_to_ix, config)
    print('test accuracy', test_accuracy)


if __name__ == '__main__':
    para_option = sys.argv[1].split('=')[1]
    inc_option = para_option.split('-')[0]
    config = {'input_size': 1500, 'hidden_size': 4096, 'train_batch_size': 10000, 'test_batch_size': 10000,
              'epoch_num': -1, 'lr': 3e-4, 'unknown_freq': 2, 'inc_option': inc_option, 'gold_ratio': 0.1,
              'iter_num': 5, 'init_epoch_num': 20, 'iter_epoch_num': 1, 'seed': 66, 'para_option': para_option}
    if len(para_option.split('-')) > 1:
        if inc_option == 'partial':
            config['partial_unk_rate'] = float(para_option.split('-')[1])
            print('partial_unk_rate', config['partial_unk_rate'])
        elif inc_option == 'noisy':
            config['noisy_diff_rate'] = float(para_option.split('-')[1])
            print('noisy_diff_rate', config['noisy_diff_rate'])
            # config['noisy_lambda'] = 1 - eta
            config['noisy_lambda'] = 1.0
            config['para_option'] = para_option + '-1.0'
        elif inc_option == 'partial+noisy':
            config['partial_unk_rate'] = float(para_option.split('-')[1].split('+')[0])
            config['noisy_diff_rate'] = float(para_option.split('-')[1].split('+')[1])
            print('partial_unk_rate', config['partial_unk_rate'])
            print('noisy_diff_rate', config['noisy_diff_rate'])
            config['noisy_lambda'] = 1.0
            config['para_option'] = para_option + '-1.0'
        elif inc_option == 'auxiliary':
            config['auxiliary_option'] = para_option.split('-')[1]
            config['k-gram'] = 5
            config['k-gram-freq-gate'] = 2
            config['inc_lambda'] = 1.0
        elif inc_option == 'knowledge':
            config['k-gram'] = int(para_option.split('-')[1])
            config['k-gram-freq-gate'] = 2
            config['inc_lambda'] = 1.0
        elif inc_option == 'constraints':
            config['constraint_option'] = para_option.split('-')[1]
        elif inc_option == 'partial+constraints':
            config['constraint_option'] = para_option.split('-')[1].split('+')[0]
            config['partial_unk_rate'] = float(para_option.split('-')[1].split('+')[1])

    print('config', config)
    set_random_seed(config['seed'])
    print('incidental option', config['inc_option'])
    run_test_experiments(config)

