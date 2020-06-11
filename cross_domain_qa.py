import json
import numpy as np
from scipy.stats import bernoulli

from utils import set_random_seed


def get_stats(input_file):
    para_num = 0
    qa_num = 0
    answer_num = 0
    words_num = 0
    ques_words_num = 0
    ans_words_num = 0
    max_word_num = 0
    max_ques_length = 0
    max_ans_length = 0
    min_word_num = 100000
    min_ques_length = 100000
    min_ans_length = 100000
    with open(input_file) as f:
        data = json.load(f)
    for article in data['data']:
        paragraphs = article['paragraphs']
        title = article['title']
        for paragraph in paragraphs:
            para = paragraph['context']
            words_num += len(para.split())
            if len(para.split()) > max_word_num:
                max_word_num = len(para.split())
            if len(para.split()) < min_word_num:
                min_word_num = len(para.split())
            para_num += 1
            qas = paragraph['qas']
            qa_num += len(qas)
            for qa in qas:
                ques_words_num += len(qa['question'].split())
                ans_words_num += len(qa['answers'][0]['text'].split())
                answer_num += len(qa['answers'])
                if len(qa['answers'][0]['text'].split()) > max_ans_length:
                    max_ans_length = len(qa['answers'][0]['text'].split())
                if len(qa['answers'][0]['text'].split()) < min_ans_length:
                    min_ans_length = len(qa['answers'][0]['text'].split())
                if len(qa['question'].split()) > max_ques_length:
                    max_ques_length = len(qa['question'].split())
                if len(qa['question'].split()) < min_ques_length:
                    min_ques_length = len(qa['question'].split())
    print('para num', para_num)
    print('para length', words_num/para_num)
    print('min word num', min_word_num)
    print('max word num', max_word_num)
    print('qa num', qa_num)
    print('qa num per para', qa_num/para_num)
    print('ques length', ques_words_num/qa_num)
    print('min ques length', min_ques_length)
    print('max ques length', max_ques_length)
    print('answer num', answer_num)
    print('answer num per question', answer_num/qa_num)
    print('ans length', ans_words_num/qa_num)
    print('min ans length', min_ans_length)
    print('max ans length', max_ans_length)


def sample_data(input_file, output_file, sample_rate):
    np.random.seed(seed=66)
    with open(input_file) as f:
        data = json.load(f)
    new_articles = []
    for article in data['data']:
        paragraphs = article['paragraphs']
        title = article['title']
        paragraphs_new = []
        for paragraph in paragraphs:
            flag = bernoulli.rvs(sample_rate)
            if flag == 1:
                paragraphs_new.append(paragraph)
        new_articles.append({'title': title, 'paragraphs': paragraphs_new})
    data = {'data': new_articles}
    with open(output_file, 'w') as f:
        json.dump(data, f)


def get_unique_answer_data(input_file, output_file):
    with open(input_file) as f:
        data = json.load(f)
    new_articles = []
    for article in data['data']:
        paragraphs = article['paragraphs']
        title = article['title']
        paragraphs_new = []
        for paragraph in paragraphs:
            qas = paragraph['qas']
            for qa in qas:
                qa['answers'] = [qa['answers'][0]]
            paragraphs_new.append(paragraph)
        new_articles.append({'title': title, 'paragraphs': paragraphs_new})
    data = {'data': new_articles}
    with open(output_file, 'w') as f:
        json.dump(data, f)


def combine_data(input_file_list, output_file, option):
    total_paragraphs = []
    for input_file in input_file_list:
        with open(input_file) as f:
            cur_data = json.load(f)
        for article in cur_data['data']:
            paragraphs = article['paragraphs']
            total_paragraphs.extend(paragraphs)
    articles = [{'title': option, 'paragraphs': total_paragraphs}]
    data = {'data': articles}
    with open(output_file, 'w') as f:
        json.dump(data, f)


def get_paragraphs(input_file):
    with open(input_file) as f:
        data = json.load(f)
    total_paragraphs = []
    for article in data['data']:
        paragraphs = article['paragraphs']
        total_paragraphs.extend(paragraphs)
    return total_paragraphs


def write_data(paragraphs, output_file, option):
    articles = [{'title': option, 'paragraphs': paragraphs}]
    data = {'data': articles}
    with open(output_file, 'w') as f:
        json.dump(data, f)


def generate_qa_data():
    squad_file = 'QA-data/SQuAD/squad.all.json'
    qamr_file = 'QA-data/QAMR/qamr.all.json'
    qasrl_file = 'QA-data/Large-QA-SRL/qasrl.all.json'
    qare_file = 'QA-data/QA-RE/qare.all.json'
    newsqa_file = 'QA-data/NewsQA/newsqa.all.json'
    triviaqa_file = 'QA-data/TriviaQA/triviaqa.all.json'

    small_file = 'QA-data/xdomain-QA/small_newsqa.json'

    large_squad_file = 'QA-data/xdomain-QA/large_squad.json'
    small_large_squad_file = 'QA-data/xdomain-QA/small_large_squad.json'
    test_squad_file = 'QA-data/xdomain-QA/test_squad.json'
    large_qamr_file = 'QA-data/xdomain-QA/large_qamr.json'
    small_large_qamr_file = 'QA-data/xdomain-QA/small_large_qamr.json'
    test_qamr_file = 'QA-data/xdomain-QA/test_qamr.json'
    large_qasrl_file = 'QA-data/xdomain-QA/large_qasrl.json'
    small_large_qasrl_file = 'QA-data/xdomain-QA/small_large_qasrl.json'
    test_qasrl_file = 'QA-data/xdomain-QA/test_qasrl.json'
    large_qare_file = 'QA-data/xdomain-QA/large_qare.json'
    small_large_qare_file = 'QA-data/xdomain-QA/small_large_qare.json'
    test_qare_file = 'QA-data/xdomain-QA/test_qare.json'
    large_newsqa_file = 'QA-data/xdomain-QA/large_newsqa.json'
    small_large_newsqa_file = 'QA-data/xdomain-QA/small_large_newsqa.json'
    test_newsqa_file = 'QA-data/xdomain-QA/test_newsqa.json'
    large_triviaqa_file = 'QA-data/xdomain-QA/large_triviaqa.json'
    small_large_triviaqa_file = 'QA-data/xdomain-QA/small_large_triviaqa.json'
    test_triviaqa_file = 'QA-data/xdomain-QA/test_triviaqa.json'

    squad_paragraphs = get_paragraphs(squad_file)
    qamr_paragraphs = get_paragraphs(qamr_file)
    qasrl_paragraphs = get_paragraphs(qasrl_file)
    qare_paragraphs = get_paragraphs(qare_file)
    newsqa_paragraphs = get_paragraphs(newsqa_file)
    triviaqa_paragraphs = get_paragraphs(triviaqa_file)
    np.random.shuffle(squad_paragraphs)
    np.random.shuffle(qamr_paragraphs)
    np.random.shuffle(qasrl_paragraphs)
    np.random.shuffle(qare_paragraphs)
    np.random.shuffle(newsqa_paragraphs)
    np.random.shuffle(triviaqa_paragraphs)
    squad_paragraphs = squad_paragraphs[:int(len(squad_paragraphs) * 69274 / 98169)]
    qamr_paragraphs = qamr_paragraphs[:int(len(qamr_paragraphs) * 69274 / 87877)]
    qasrl_paragraphs = qasrl_paragraphs[:int(len(qasrl_paragraphs) * 69274 / 299308)]
    qare_paragraphs = qare_paragraphs[:int(len(qare_paragraphs) * 69274 / 121753)]
    newsqa_paragraphs = newsqa_paragraphs[:int(len(newsqa_paragraphs) * 69274 / 78372)]
    triviaqa_paragraphs = triviaqa_paragraphs[:int(len(triviaqa_paragraphs) * 69274 / 69274)]
    squad_num = len(squad_paragraphs)
    qamr_num = len(qamr_paragraphs)
    qasrl_num = len(qasrl_paragraphs)
    qare_num = len(qare_paragraphs)
    newsqa_num = len(newsqa_paragraphs)
    triviaqa_num = len(triviaqa_paragraphs)
    train_ratio = 0.01
    dev_ratio = 0.1
    test_ratio = 0.4

    small_data = newsqa_paragraphs[: int(newsqa_num * train_ratio)]

    large_squad = squad_paragraphs[int(squad_num * train_ratio): int(squad_num * dev_ratio)]
    test_squad = squad_paragraphs[int(squad_num * dev_ratio):int(squad_num * test_ratio)]
    large_qamr = qamr_paragraphs[int(qamr_num * train_ratio): int(qamr_num * dev_ratio)]
    test_qamr = qamr_paragraphs[int(qamr_num * dev_ratio):int(qamr_num * test_ratio)]
    large_qasrl = qasrl_paragraphs[int(qasrl_num * train_ratio): int(qasrl_num * dev_ratio)]
    test_qasrl = qasrl_paragraphs[int(qasrl_num * dev_ratio):int(qasrl_num * test_ratio)]
    large_qare = qare_paragraphs[int(qare_num * train_ratio): int(qare_num * dev_ratio)]
    test_qare = qare_paragraphs[int(qare_num * dev_ratio):int(qare_num * test_ratio)]
    large_newsqa = newsqa_paragraphs[int(newsqa_num * train_ratio): int(newsqa_num * dev_ratio)]
    test_newsqa = newsqa_paragraphs[int(newsqa_num * dev_ratio):int(newsqa_num * test_ratio)]
    large_triviaqa = triviaqa_paragraphs[int(triviaqa_num * train_ratio): int(triviaqa_num * dev_ratio)]
    test_triviaqa = triviaqa_paragraphs[int(triviaqa_num * dev_ratio):int(triviaqa_num * test_ratio)]

    small_large_squad = []
    small_large_squad.extend(small_data)
    small_large_squad.extend(large_squad)
    small_large_qamr = []
    small_large_qamr.extend(small_data)
    small_large_qamr.extend(large_qamr)
    small_large_qasrl = []
    small_large_qasrl.extend(small_data)
    small_large_qasrl.extend(large_qasrl)
    small_large_qare = []
    small_large_qare.extend(small_data)
    small_large_qare.extend(large_qare)
    small_large_newsqa = []
    small_large_newsqa.extend(small_data)
    small_large_newsqa.extend(large_newsqa)
    small_large_triviaqa = []
    small_large_triviaqa.extend(small_data)
    small_large_triviaqa.extend(large_triviaqa)

    write_data(small_data, small_file, 'small_newsqa')

    write_data(large_squad, large_squad_file, 'large_squad')
    write_data(small_large_squad, small_large_squad_file, 'small_large_squad')
    write_data(test_squad, test_squad_file, 'test_squad')
    write_data(large_qamr, large_qamr_file, 'larg_qamr')
    write_data(small_large_qamr, small_large_qamr_file, 'small_large_qamr')
    write_data(test_qamr, test_qamr_file, 'test_qamr')
    write_data(large_qasrl, large_qasrl_file, 'large_qasrl')
    write_data(small_large_qasrl, small_large_qasrl_file, 'small_large_qasrl')
    write_data(test_qasrl, test_qasrl_file, 'test_qasrl')
    write_data(large_qare, large_qare_file, 'large_qare')
    write_data(small_large_qare, small_large_qare_file, 'small_large_qare')
    write_data(test_qare, test_qare_file, 'test_qare')
    write_data(large_newsqa, large_newsqa_file, 'large_newsqa')
    write_data(small_large_newsqa, small_large_newsqa_file, 'small_large_newsqa')
    write_data(test_newsqa, test_newsqa_file, 'test_newsqa')
    write_data(large_triviaqa, large_triviaqa_file, 'large_triviaqa')
    write_data(small_large_triviaqa, small_large_triviaqa_file, 'small_large_triviaqa')
    write_data(test_triviaqa, test_triviaqa_file, 'test_triviaqa')


if __name__ == '__main__':
    set_random_seed(666)
    # dev_file = 'QA-data/TriviaQA/TriviaQA_squad_dev.json'
    # unique_dev_file = 'QA-data/TriviaQA/TriviaQA_squad_dev.unique.json'
    # get_unique_answer_data(dev_file, unique_dev_file)
    # input_file_list = ['QA-data/TriviaQA/TriviaQA_squad_train.json', 'QA-data/TriviaQA/TriviaQA_squad_dev.unique.json']
    # output_file = 'QA-data/TriviaQA/triviaqa.all.json'
    # option = 'triviaqa'
    # combine_data(input_file_list, output_file, option)
    generate_qa_data()
    # input_file = 'QA-data/xdomain-QA/small_large_triviaqa.json'
    # get_stats(input_file)
