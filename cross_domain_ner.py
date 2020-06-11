import numpy as np

from utils import set_random_seed


def get_original_stats(file):
    fin = open(file)
    lines = fin.readlines()
    labels = []
    sent_num = 0
    no_entity_num = 0
    entity_num = 0
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            if np.sum(np.array(labels) == 'O') == len(labels):
                no_entity_num += 1
            for label in labels:
                if label[0] == 'B':
                    entity_num += 1
            sent_num += 1
            labels = []
            continue
        labels.append(line.split()[-1])
    fin.close()
    print('no entity sentence percentage', no_entity_num / sent_num)
    print('average entity per sentence', entity_num / sent_num)


def get_stats(file):
    fin = open(file)
    lines = fin.readlines()
    labels = []
    sent_num = 0
    no_entity_num = 0
    entity_num = 0
    word_num = 0
    B_num = 0
    I_num = 0
    O_num = 0
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            if np.sum(np.array(labels) == 'O') == len(labels):
                no_entity_num += 1
            for label in labels:
                if label[0] == 'B':
                    entity_num += 1
                    B_num += 1
                elif label[0] == 'I':
                    I_num += 1
                else:
                    O_num += 1
            sent_num += 1
            word_num += len(labels)
            labels = []
            continue
        labels.append(line.split()[-1])
    fin.close()
    print('no entity sentence percentage', no_entity_num / sent_num)
    print('average entity per sentence', entity_num / sent_num)
    print('sentence length', word_num / sent_num)
    print('BIO proportion', B_num / word_num, I_num / word_num, O_num / word_num)
    print('entity length', (B_num + I_num) / B_num)
    print('sentence num', sent_num)
    print('word num', word_num)


def get_data(file):
    fin = open(file)
    lines = fin.readlines()
    words = []
    labels = []
    with_entities = []
    no_entities = []
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            if len(labels) == 0:
                continue
            if np.sum(np.array(labels) == 'O') == len(labels):
                no_entities.append((words, labels))
            else:
                with_entities.append((words, labels))
            words = []
            labels = []
            continue
        words.append(line.split()[0])
        labels.append(line.split()[-1])
    fin.close()
    return with_entities, no_entities


def write_data(data, file):
    print('data size', len(data))
    fout = open(file, 'w')
    for words, labels in data:
        for x in range(len(words)):
            fout.write(words[x] + '\t' + labels[x] + '\n')
        fout.write('\n')
    fout.close()


def generate_person_data(no_entity_percent=0.5):
    ontonotes_file = 'ontonotes-ner/ontonotes_PER.txt'
    conll_file = 'conll/conll_PER.txt'
    twitter_file = 'twitter/twitter_PER.txt'
    GMB_file = 'GMB/GMB_PER.txt'
    small_twitter_file = 'xdomain-person/small_twitter.txt'
    large_twitter_file = 'xdomain-person/large_twitter.txt'
    small_large_twitter_file = 'xdomain-person/small_large_twitter.txt'
    test_twitter_file = 'xdomain-person/test_twitter.txt'
    large_ontonotes_file = 'xdomain-person/large_ontonotes.txt'
    small_large_ontonotes_file = 'xdomain-person/small_large_ontonotes.txt'
    test_ontonotes_file = 'xdomain-person/test_ontonotes.txt'
    large_conll_file = 'xdomain-person/large_conll.txt'
    small_large_conll_file = 'xdomain-person/small_large_conll.txt'
    test_conll_file = 'xdomain-person/test_conll.txt'
    large_GMB_file = 'xdomain-person/large_GMB.txt'
    small_large_GMB_file = 'xdomain-person/small_large_GMB.txt'
    test_GMB_file = 'xdomain-person/test_GMB.txt'
    ontonotes_with, ontonotes_no = get_data(ontonotes_file)
    conll_with, conll_no = get_data(conll_file)
    twitter_with, twitter_no = get_data(twitter_file)
    GMB_with, GMB_no = get_data(GMB_file)
    with_sent_num = np.min([len(ontonotes_with), len(conll_with), len(twitter_with), len(GMB_with)])
    np.random.shuffle(ontonotes_with)
    np.random.shuffle(ontonotes_no)
    np.random.shuffle(conll_with)
    np.random.shuffle(conll_no)
    np.random.shuffle(twitter_with)
    np.random.shuffle(twitter_no)
    np.random.shuffle(GMB_with)
    np.random.shuffle(GMB_no)
    ontonotes = []
    conll = []
    twitter = []
    GMB = []
    twitter.extend(twitter_with[: with_sent_num])
    twitter.extend(twitter_no[: with_sent_num])
    ontonotes.extend(ontonotes_with[: with_sent_num])
    ontonotes.extend(ontonotes_no[: with_sent_num])
    conll.extend(conll_with[: with_sent_num])
    conll.extend(conll_no[: with_sent_num])
    GMB.extend(GMB_with[: with_sent_num])
    GMB.extend(GMB_no[: with_sent_num])
    np.random.shuffle(ontonotes)
    np.random.shuffle(conll)
    np.random.shuffle(twitter)
    np.random.shuffle(GMB)
    sent_num = with_sent_num * 2
    small_twitter = twitter[: int(sent_num * 0.05)]
    large_twitter = twitter[int(sent_num * 0.05): int(sent_num * 0.5)]
    test_twitter = twitter[int(sent_num * 0.5):]
    large_ontonotes = ontonotes[int(sent_num * 0.05): int(sent_num * 0.5)]
    test_ontonotes = ontonotes[int(sent_num * 0.5):]
    large_conll = conll[int(sent_num * 0.05): int(sent_num * 0.5)]
    test_conll = conll[int(sent_num * 0.5):]
    large_GMB = GMB[int(sent_num * 0.05): int(sent_num * 0.5)]
    test_GMB = GMB[int(sent_num * 0.5):]
    small_large_ontonotes = []
    small_large_ontonotes.extend(small_twitter)
    small_large_ontonotes.extend(large_ontonotes)
    small_large_conll = []
    small_large_conll.extend(small_twitter)
    small_large_conll.extend(large_conll)
    small_large_twitter = []
    small_large_twitter.extend(small_twitter)
    small_large_twitter.extend(large_twitter)
    small_large_GMB = []
    small_large_GMB.extend(small_twitter)
    small_large_GMB.extend(large_GMB)
    write_data(small_twitter, small_twitter_file)
    write_data(large_twitter, large_twitter_file)
    write_data(small_large_twitter, small_large_twitter_file)
    write_data(test_twitter, test_twitter_file)
    write_data(large_ontonotes, large_ontonotes_file)
    write_data(small_large_ontonotes, small_large_ontonotes_file)
    write_data(test_ontonotes, test_ontonotes_file)
    write_data(large_conll, large_conll_file)
    write_data(small_large_conll, small_large_conll_file)
    write_data(test_conll, test_conll_file)
    write_data(large_GMB, large_GMB_file)
    write_data(small_large_GMB, small_large_GMB_file)
    write_data(test_GMB, test_GMB_file)


if __name__ == '__main__':
    set_random_seed(666)
    # orig_file = 'ontonotes-ner/ontonotes.train.ner'
    # orig_file = 'conll/eng.bio.train'
    # orig_file = 'twitter/train'
    # orig_file = 'GMB/GMB.txt'
    # get_original_stats(orig_file)
    # input_file = 'ontonotes-ner/ontonotes_PER.txt'
    # input_file = 'conll/conll_PER.txt'
    # input_file = 'twitter/twitter_PER.txt'
    # input_file = 'GMB/GMB_PER.txt'
    # get_stats(input_file)
    generate_person_data()
