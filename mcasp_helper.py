import re
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from math import log
import json


class Find_Words:
    def __init__(self, min_count=10, max_count=10000000, min_pmi=0):
        self.min_count = min_count
        self.min_pmi = min_pmi
        self.chars, self.pairs = defaultdict(int), defaultdict(int)
        self.total = 0.
        self.max_count = max_count

    def text_filter(self, texts):
        for a in tqdm(texts):
            for t in re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', ''.join(a)):
                if t:
                    yield t

    def count(self, texts):
        mi_list = []
        for text in self.text_filter(texts):
            self.chars[text[0]] += 1
            for i in range(len(text)-1):
                self.chars[text[i+1]] += 1
                self.pairs[text[i:i+2]] += 1
                self.total += 1
        self.chars = {i:j for i,j in self.chars.items() if 100 * self.max_count > j > self.min_count}
        self.pairs = {i:j for i,j in self.pairs.items() if self.max_count > j > self.min_count}

        self.strong_segments = set()
        for i,j in self.pairs.items():
            if i[0] in self.chars and i[1] in self.chars:
                mi = log(self.total*j/(self.chars[i[0]]*self.chars[i[1]]))
                mi_list.append(mi)
                if mi >= self.min_pmi:
                    self.strong_segments.add(i)
        print('min mi: %.4f' % min(mi_list))
        print('max mi: %.4f' % max(mi_list))
        print('remaining: %d / %d (%.4f)' % (len(self.strong_segments), len(mi_list), len(self.strong_segments)/len(mi_list)))

    def find_words(self, texts, n):
        self.words = defaultdict(int)
        for text in self.text_filter(texts):
            s = text[0]
            for i in range(len(text)-1):
                if text[i:i+2] in self.strong_segments:
                    s += text[i+1]
                else:
                    self.words[s] += 1
                    s = text[i+1]
        self.words = {i:j for i,j in self.words.items() if j > self.min_count and n+1 > len(i) > 0}


def read_tsv(file_path):
    sentence_list = []
    label_list = []
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentence = []
        labels = []
        for line in lines:
            line = line.strip()
            if line == '':
                if len(sentence) > 0:
                    sentence_list.append(sentence)
                    label_list.append(labels)
                    sentence = []
                    labels = []
                continue
            items = re.split('\\s+', line)
            character = items[0]
            label = items[-1]
            sentence.append(character)
            labels.append(label)

    return sentence_list, label_list


def get_word2id(train_path):
    word2id = {'<PAD>': 0, '<UNK>': 1}
    word = ''
    index = 2
    with open(train_path, 'r', encoding='utf8') as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split('\t')
            character = splits[0]
            label = splits[-1][0]
            word += character
            if label in ['S', 'E']:
                if word not in word2id:
                    word2id[word] = index
                    index += 1
                word = ''
    return word2id


def dlg(data_path_list, ngram_length, renew_freq):

    all_sentences = []
    for file_name in data_path_list:
        sentences, _ = read_tsv(file_name)
        all_sentences.extend(sentences)

    n_gram_dict = extract_ngram(all_sentences, 0, ngram_length)
    corpus_size = 0
    for gram, count in n_gram_dict.items():
        if len(gram) == 1:
            corpus_size += count

    min_dlg = np.inf
    max_dlg = -np.inf

    min_dlg_2 = np.inf
    max_dlg_2 = -np.inf

    n_gram_dlg_dict = {}
    num_small_dlg = 0
    skip_num = 0

    for gram, c_gram in tqdm(n_gram_dict.items()):
        if len(gram) == 1 or c_gram < 2:
            skip_num += 1
            continue
        new_corpus_size = corpus_size - c_gram * (len(gram) - 1) + len(gram) + 1
        dlg = c_gram * np.log10(c_gram) + corpus_size * np.log10(corpus_size) - new_corpus_size * np.log10(new_corpus_size)
        if dlg > max_dlg_2:
            max_dlg_2 = dlg
        if dlg < min_dlg_2:
            min_dlg_2 = dlg
        # if dlg > 200000:
        #     print('%s %d' % (gram, c_gram))
        char_in_gram = list(set(gram))
        for character in char_in_gram:
            c_character = n_gram_dict[character]
            new_c_character = c_character - (c_gram - 1) * gram.count(character)
            # if not new_c_character > 0:
            #     print('gram: %s' % gram)
            #     print('# of new c character: %d' % new_c_character)
            #     raise ValueError()
            new_character_item = new_c_character * np.log10(new_c_character) if new_c_character > 0 else 0
            dlg += new_character_item - c_character * np.log10(c_character)
        adlg = dlg / c_gram
        if dlg > 0:
            n_gram_dlg_dict[gram] = dlg / c_gram
        else:
            num_small_dlg += 1
        if adlg > max_dlg:
            max_dlg = adlg
        if adlg < min_dlg:
            min_dlg = adlg

    new_dlg_dict = {}
    new_all_sentences = []
    for sen in all_sentences:
        str_sen = ''.join(sen)
        new_sen = re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', str_sen)
        for s in new_sen:
            if len(s) > 0:
                new_all_sentences.append(s)
    for sentence in tqdm(new_all_sentences):
        n_gram_list = vitbi(sentence, n_gram_dlg_dict)
        for gram in n_gram_list:
            if gram not in new_dlg_dict:
                new_dlg_dict[gram] = 1
            else:
                new_dlg_dict[gram] += 1

    new_dlg_dict_2 = {gram: c for gram, c in new_dlg_dict.items() if c >= renew_freq and len(gram) < ngram_length+1}

    new_dlg_dict_2 = renew_ngram_by_freq(all_sentences, new_dlg_dict_2, renew_freq, ngram_length)

    return new_dlg_dict_2


def vitbi(sentence, ngram_dict):
    score = [0 for i in range(len(sentence))]
    n_gram = [[] for i in range(len(sentence))]
    word = sentence[0]
    n_gram[0].append(word)
    for i in range(1, len(score)):
        tmp_score_list = [score[i-1], -1, -1, -1, -1]
        for n in range(2, 6):
            if i - n < -1:
                break
            word = ''.join(sentence[i - n + 1: i + 1])
            if word in ngram_dict:
                tmp_score_list[n-1] = score[i-n] + ngram_dict[word] if i-n >= 0 else ngram_dict[word]
        max_score = max(tmp_score_list)
        max_score_index = tmp_score_list.index(max(tmp_score_list))
        word = ''.join(sentence[i-max_score_index: i+1])
        score[i] = max_score
        if i-(max_score_index+1) >= 0:
            n_gram[i].extend(n_gram[i - (max_score_index + 1)])
        n_gram[i].append(word)
    return n_gram[-1]


def pmi(data_path_list, ngram_length, renew_freq):
    all_sentences = []
    for file_name in data_path_list:
        sentences, _ = read_tsv(file_name)
        all_sentences.extend(sentences)

    fw = Find_Words(0, 1000000000000, 0)

    fw.count(all_sentences)
    fw.find_words(all_sentences, ngram_length)
    words = fw.words

    words = renew_ngram_by_freq(all_sentences, words, renew_freq, ngram_length)

    return words


def av(data_path_list, av_threshold, ngram_length, renew_freq):

    all_sentences = []
    for file_name in data_path_list:
        sentences, _ = read_tsv(file_name)
        all_sentences.extend(sentences)

    n_gram_dict = {}
    new_all_sentences = []

    ngram2av = {}

    for sen in all_sentences:
        str_sen = ''.join(sen)
        new_sen = re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', str_sen)
        for s in new_sen:
            if len(s) > 0:
                new_all_sentences.append(s)

    for sentence in tqdm(new_all_sentences):
        for i in range(len(sentence)):
            for n in range(1, ngram_length+1):
                if i + n > len(sentence):
                    break
                left_index = i - 1
                right_index = i + n
                n_gram = ''.join(sentence[i: i + n])
                if n_gram not in n_gram_dict:
                    n_gram_dict[n_gram] = 1
                    ngram2av[n_gram] = {'l': {}, 'r': {}}
                else:
                    n_gram_dict[n_gram] += 1
                if left_index >= 0:
                    ngram2av[n_gram]['l'][sentence[left_index]] = 1
                if right_index < len(sentence):
                    ngram2av[n_gram]['r'][sentence[right_index]] = 1
    remaining_ngram = {}
    for ngram, av_dict in ngram2av.items():
        avl = len(av_dict['l'])
        avr = len(av_dict['r'])
        av = min(avl, avr)
        if av > av_threshold and n_gram_dict[ngram] > 0:
            remaining_ngram[ngram] = n_gram_dict[ngram]

    remaining_ngram = renew_ngram_by_freq(all_sentences, remaining_ngram, renew_freq, ngram_length)

    return remaining_ngram


def extract_ngram(all_sentences, min_feq=0, ngram_len=10):
    n_gram_dict = {}

    new_all_sentences = []

    for sen in all_sentences:
        str_sen = ''.join(sen)
        new_sen = re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', str_sen)
        for s in new_sen:
            if len(s) > 0:
                new_all_sentences.append(s)

    for sentence in new_all_sentences:
        for i in range(len(sentence)):
            for n in range(1, ngram_len+1):
                if i + n > len(sentence):
                    break
                n_gram = ''.join(sentence[i: i + n])
                if n_gram not in n_gram_dict:
                    n_gram_dict[n_gram] = 1
                else:
                    n_gram_dict[n_gram] += 1
    new_ngram_dict = {gram: c for gram, c in n_gram_dict.items() if c > min_feq}
    return new_ngram_dict


def renew_ngram_by_freq(all_sentences, ngram2count, min_feq, ngram_len=10):
    new_ngram2count = {}

    new_all_sentences = []

    for sen in all_sentences:
        str_sen = ''.join(sen)
        new_sen = re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', str_sen)
        for s in new_sen:
            if len(s) > 0:
                new_all_sentences.append(s)

    for sentence in new_all_sentences:
        for i in range(len(sentence)):
            for n in range(1, ngram_len+1):
                if i + n > len(sentence):
                    break
                n_gram = ''.join(sentence[i: i + n])
                if n_gram not in ngram2count:
                    continue
                if n_gram not in new_ngram2count:
                    new_ngram2count[n_gram] = 1
                else:
                    new_ngram2count[n_gram] += 1
    new_ngram_dict = {gram: c for gram, c in new_ngram2count.items() if c >= min_feq}
    return new_ngram_dict


def get_gram2id(data_path_list, ngram_type, ngram_len, av_threshold, ngram_threshold):

    if ngram_type == 'pmi':
        word2count = pmi(data_path_list=data_path_list, ngram_length=ngram_len, renew_freq=ngram_threshold)
    elif ngram_type == 'av':
        word2count = av(data_path_list=data_path_list, av_threshold=av_threshold,
                        ngram_length=ngram_len, renew_freq=ngram_threshold)
    elif ngram_type == 'dlg':
        word2count = dlg(data_path_list=data_path_list, ngram_length=ngram_len, renew_freq=ngram_threshold)
    else:
        raise ValueError()

    gram2id = {'<PAD>': 0}
    index = 1
    for word, count in word2count.items():
        # if count > threshold and count < upper_threshold:
        gram2id[word] = index
        index += 1
    return gram2id, word2count


def load_json(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        line = f.readline()
    return json.loads(line)


def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf8') as f:
        json.dump(data, f)
        f.write('\n')


def get_labels(train_path):
    label_list = ['<PAD>', '<UNK>']

    with open(train_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split()
            joint_label = splits[1]
            if joint_label not in label_list:
                label_list.append(joint_label)

    label_list.extend(['[CLS]', '[SEP]'])
    return label_list
