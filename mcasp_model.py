from __future__ import absolute_import, division, print_function

import math
import os
import numpy as np
import torch
import subprocess
from torch import nn
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer

import pytorch_pretrained_zen as zen

from pytorch_pretrained_bert.crf import CRF

from mcasp_helper import load_json, save_json, read_tsv

DEFAULT_HPARA = {
    'max_seq_length': 128,
    'max_ngram_size': 128,
    'use_bert': False,
    'use_zen': False,
    'do_lower_case': False,
    'use_attention': False,
    'cat_type': 'length',
    'cat_num': 10,
    'ngram_length': 10,
}


class MultiChannelAttention(nn.Module):
    def __init__(self, ngram_size, hidden_size, cat_num):
        super(MultiChannelAttention, self).__init__()
        self.word_embedding = nn.Embedding(ngram_size, hidden_size, padding_idx=0)
        self.channel_weight = nn.Embedding(cat_num, 1)
        self.temper = hidden_size ** 0.5

    def forward(self, word_seq, hidden_state, char_word_mask_matrix, channel_ids):
        # word_seq: (batch_size, channel, word_seq_len)
        # hidden_state: (batch_size, character_seq_len, hidden_size)
        # mask_matrix: (batch_size, channel, character_seq_len, word_seq_len)

        # embedding (batch_size, channel, word_seq_len, word_embedding_dim)
        batch_size, character_seq_len, hidden_size = hidden_state.shape
        channel = char_word_mask_matrix.shape[1]
        word_seq_length = word_seq.shape[2]

        embedding = self.word_embedding(word_seq)

        tmp = embedding.permute(0, 1, 3, 2)

        tmp_hidden_state = torch.stack([hidden_state] * channel, 1)

        # u (batch_size, channel, character_seq_len, word_seq_len)
        u = torch.matmul(tmp_hidden_state, tmp) / self.temper

        # attention (batch_size, channel, character_seq_len, word_seq_len)
        tmp_word_mask_metrix = torch.clamp(char_word_mask_matrix, 0, 1)

        exp_u = torch.exp(u)
        delta_exp_u = torch.mul(exp_u, tmp_word_mask_metrix)

        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 3)] * delta_exp_u.shape[3], 3)

        attention = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10)

        attention = attention.view(batch_size * channel, character_seq_len, word_seq_length)
        embedding = embedding.view(batch_size * channel, word_seq_length, hidden_size)

        character_attention = torch.bmm(attention, embedding)

        character_attention = character_attention.view(batch_size, channel, character_seq_len, hidden_size)

        channel_w = self.channel_weight(channel_ids)
        channel_w = nn.Softmax(dim=1)(channel_w)

        channel_w = channel_w.view(batch_size, -1, 1, 1)

        character_attention = torch.mul(character_attention, channel_w)

        character_attention = character_attention.permute(0, 2, 1, 3)
        character_attention = character_attention.flatten(start_dim=2)

        return character_attention


class McASP(nn.Module):

    def __init__(self, word2id, gram2id, gram2count, labelmap, hpara, model_path):
        super().__init__()

        self.word2id = word2id
        self.gram2id = gram2id
        self.gram2count = gram2count
        self.labelmap = labelmap
        self.hpara = hpara
        self.num_labels = len(self.labelmap)
        self.max_seq_length = self.hpara['max_seq_length']
        self.use_attention = self.hpara['use_attention']
        self.cat_type = self.hpara['cat_type']
        self.cat_num = self.hpara['cat_num']
        self.ngram_length = self.hpara['ngram_length']

        if self.cat_type == 'length':
            assert self.cat_num == self.ngram_length

        self.max_ngram_size = self.hpara['max_ngram_size']

        self.bert_tokenizer = None
        self.bert = None
        self.zen_tokenizer = None
        self.zen = None
        self.zen_ngram_dict = None

        if self.hpara['use_bert']:
            self.bert_tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            self.bert = BertModel.from_pretrained(model_path, cache_dir='')
            hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        elif self.hpara['use_zen']:
            self.zen_tokenizer = zen.BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            self.zen_ngram_dict = zen.ZenNgramDict(model_path, tokenizer=self.zen_tokenizer)
            self.zen = zen.modeling.ZenModel.from_pretrained(model_path, cache_dir='')
            hidden_size = self.zen.config.hidden_size
            self.dropout = nn.Dropout(self.zen.config.hidden_dropout_prob)
        else:
            raise ValueError()

        if self.use_attention:
            self.multi_attention = MultiChannelAttention(len(self.gram2id), hidden_size, self.cat_num)
            self.classifier = nn.Linear(hidden_size * (1 + self.cat_num), self.num_labels, bias=False)
        else:
            self.multi_attention = None
            self.classifier = nn.Linear(hidden_size, self.num_labels, bias=False)

        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None,
                word_seq=None, label_value_matrix=None, word_mask=None, channel_ids=None,
                input_ngram_ids=None, ngram_position_matrix=None):

        if self.bert is not None:
            sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        elif self.zen is not None:
            sequence_output, _ = self.zen(input_ids, input_ngram_ids=input_ngram_ids,
                                          ngram_position_matrix=ngram_position_matrix,
                                          token_type_ids=token_type_ids, attention_mask=attention_mask,
                                          output_all_encoded_layers=False)
        else:
            raise ValueError()

        if self.multi_attention is not None:
            attention_output = self.multi_attention(word_seq, sequence_output, word_mask, channel_ids)
            sequence_output = torch.cat([sequence_output, attention_output], dim=2)

        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        if labels is not None:
            return -1 * self.crf(emissions=logits, tags=labels, mask=attention_mask_label)
        else:
            return self.crf.decode(logits, attention_mask_label)[0]

    @staticmethod
    def init_hyper_parameters(args):
        hyper_parameters = DEFAULT_HPARA.copy()
        hyper_parameters['max_seq_length'] = args.max_seq_length
        hyper_parameters['max_ngram_size'] = args.max_ngram_size
        hyper_parameters['use_bert'] = args.use_bert
        hyper_parameters['use_zen'] = args.use_zen
        hyper_parameters['do_lower_case'] = args.do_lower_case
        hyper_parameters['use_attention'] = args.use_attention
        hyper_parameters['cat_type'] = args.cat_type
        hyper_parameters['cat_num'] = args.cat_num
        hyper_parameters['ngram_length'] = args.ngram_length
        return hyper_parameters

    @classmethod
    def load_model(cls, model_path):
        label_map = load_json(os.path.join(model_path, 'label_map.json'))
        hpara = load_json(os.path.join(model_path, 'hpara.json'))

        gram2id_path = os.path.join(model_path, 'gram2id.json')
        gram2id = load_json(gram2id_path) if os.path.exists(gram2id_path) else None

        if gram2id is not None:
            gram2id = {tuple(k.split('`')): v for k, v in gram2id.items()}

        word2id_path = os.path.join(model_path, 'word2id.json')
        word2id = load_json(word2id_path) if os.path.exists(word2id_path) else None

        gram2count_path = os.path.join(model_path, 'gram2count.json')
        gram2count = load_json(gram2count_path) if os.path.exists(gram2count_path) else None

        res = cls(model_path=model_path, labelmap=label_map, hpara=hpara,
                  gram2id=gram2id, word2id=word2id, gram2count=gram2count)

        res.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin')))
        return res

    def save_model(self, output_dir, vocab_dir):
        output_model_path = os.path.join(output_dir, 'pytorch_model.bin')
        torch.save(self.state_dict(), output_model_path)

        label_map_file = os.path.join(output_dir, 'label_map.json')

        if not os.path.exists(label_map_file):
            save_json(label_map_file, self.labelmap)

            save_json(os.path.join(output_dir, 'hpara.json'), self.hpara)
            if self.gram2id is not None:
                gram2save = {'`'.join(list(k)): v for k, v in self.gram2id.items()}
                save_json(os.path.join(output_dir, 'gram2id.json'), gram2save)
            if self.word2id is not None:
                save_json(os.path.join(output_dir, 'word2id.json'), self.word2id)
            if self.gram2count is not None:
                save_json(os.path.join(output_dir, 'gram2count.json'), self.gram2count)

            output_config_file = os.path.join(output_dir, 'config.json')
            with open(output_config_file, "w", encoding='utf-8') as writer:
                if self.bert:
                    writer.write(self.bert.config.to_json_string())
                elif self.zen:
                    writer.write(self.zen.config.to_json_string())
                else:
                    raise ValueError()
            output_bert_config_file = os.path.join(output_dir, 'bert_config.json')
            command = 'cp ' + str(output_config_file) + ' ' + str(output_bert_config_file)
            subprocess.run(command, shell=True)

            if self.bert or self.zen:
                vocab_name = 'vocab.txt'
            else:
                raise ValueError()
            vocab_path = os.path.join(vocab_dir, vocab_name)
            command = 'cp ' + str(vocab_path) + ' ' + str(os.path.join(output_dir, vocab_name))
            subprocess.run(command, shell=True)

            if self.zen:
                ngram_name = 'ngram.txt'
                ngram_path = os.path.join(vocab_dir, ngram_name)
                command = 'cp ' + str(ngram_path) + ' ' + str(os.path.join(output_dir, ngram_name))
                subprocess.run(command, shell=True)

    def load_sentence(self, data_path):

        flag = 'pred'
        sentence_list = []
        label_list = []
        with open(data_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            sent = [c for c in line]
            label = ['S-NN' for _ in line]
            sentence_list.append(sent)
            label_list.append(label)

        data = []
        for sentence, label in zip(sentence_list, label_list):
            if self.multi_attention is not None:
                ngram_list = []
                matching_position = []
                ngram_list_len = []
                for i in range(self.cat_num):
                    ngram_list.append([])
                    matching_position.append([])
                    ngram_list_len.append(0)
                for i in range(len(sentence)):
                    for j in range(0, self.ngram_length):
                        if i + j + 1 > len(sentence):
                            break
                        ngram = ''.join(sentence[i: i + j + 1])
                        if ngram in self.gram2id:
                            channel_index = self._ngram_category(ngram)
                            try:
                                index = ngram_list[channel_index].index(ngram)
                            except ValueError:
                                ngram_list[channel_index].append(ngram)
                                index = len(ngram_list[channel_index]) - 1
                                ngram_list_len[channel_index] += 1
                            for k in range(j + 1):
                                matching_position[channel_index].append((i + k, index))
            else:
                ngram_list = None
                matching_position = None
                ngram_list_len = None
            max_ngram_len = max(ngram_list_len) if ngram_list_len is not None else None
            data.append((sentence, label, ngram_list, matching_position, max_ngram_len))

        examples = []
        for i, (sentence, label, word_list, matching_position, word_list_len) in enumerate(data):
            guid = "%s-%s" % (flag, i)
            text_a = ' '.join(sentence)
            text_b = None
            word = word_list
            label = label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, word=word, matrix=matching_position,
                             sent_len=len(sentence), word_list_len=word_list_len))
        return examples


    def load_tsv_data(self, data_path):

        flag = data_path[data_path.rfind('/')+1: data_path.rfind('.')]
        sentence_list, label_list = read_tsv(data_path)

        data = []
        for sentence, label in zip(sentence_list, label_list):
            if self.multi_attention is not None:
                ngram_list = []
                matching_position = []
                ngram_list_len = []
                for i in range(self.cat_num):
                    ngram_list.append([])
                    matching_position.append([])
                    ngram_list_len.append(0)
                for i in range(len(sentence)):
                    for j in range(0, self.ngram_length):
                        if i + j + 1 > len(sentence):
                            break
                        ngram = ''.join(sentence[i: i + j + 1])
                        if ngram in self.gram2id:
                            channel_index = self._ngram_category(ngram)
                            try:
                                index = ngram_list[channel_index].index(ngram)
                            except ValueError:
                                ngram_list[channel_index].append(ngram)
                                index = len(ngram_list[channel_index]) - 1
                                ngram_list_len[channel_index] += 1
                            for k in range(j + 1):
                                matching_position[channel_index].append((i + k, index))
            else:
                ngram_list = None
                matching_position = None
                ngram_list_len = None
            max_ngram_len = max(ngram_list_len) if ngram_list_len is not None else None
            data.append((sentence, label, ngram_list, matching_position, max_ngram_len))

        examples = []
        for i, (sentence, label, word_list, matching_position, word_list_len) in enumerate(data):
            guid = "%s-%s" % (flag, i)
            text_a = ' '.join(sentence)
            text_b = None
            word = word_list
            label = label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, word=word, matrix=matching_position,
                             sent_len=len(sentence), word_list_len=word_list_len))
        return examples

    def _ngram_category(self, ngram):
        if self.cat_type == 'length':
            index = int(min(self.cat_num, len(ngram))) - 1
            assert 0 <= index < self.cat_num
            return index
        elif self.cat_type == 'freq':
            index = int(min(self.cat_num, math.log2(self.gram2count[ngram]))) - 1
            assert 0 <= index < self.cat_num
            return index
        else:
            raise ValueError()

    def convert_examples_to_features(self, examples):

        max_seq_length = min(int(max([len(e.text_a.split(' ')) for e in examples]) * 1.1 + 2), self.max_seq_length)

        if self.multi_attention is not None:
            max_word_size = max(max([e.word_list_len for e in examples]), 1)
        else:
            max_word_size = 1

        features = []

        tokenizer = self.bert_tokenizer if self.bert_tokenizer is not None else self.zen_tokenizer

        for (ex_index, example) in enumerate(examples):
            textlist = example.text_a.split(' ')
            labellist = example.label
            tokens = []
            labels = []
            valid = []
            label_mask = []

            for i, word in enumerate(textlist):
                token = tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = labellist[i]
                for m in range(len(token)):
                    if m == 0:
                        valid.append(1)
                        labels.append(label_1)
                        label_mask.append(1)
                    else:
                        valid.append(0)

            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
                valid = valid[0:(max_seq_length - 2)]
                label_mask = label_mask[0:(max_seq_length - 2)]

            ntokens = []
            segment_ids = []
            label_ids = []

            ntokens.append("[CLS]")
            segment_ids.append(0)

            valid.insert(0, 1)
            label_mask.insert(0, 1)
            label_ids.append(self.labelmap["[CLS]"])
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                if len(labels) > i:
                    label = labels[i] if labels[i] in self.labelmap else '<UNK>'
                    label_ids.append(self.labelmap[label])
            ntokens.append("[SEP]")

            segment_ids.append(0)
            valid.append(1)
            label_mask.append(1)
            label_ids.append(self.labelmap["[SEP]"])

            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            label_mask = [1] * len(label_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append(0)
                valid.append(1)
                label_mask.append(0)
            while len(label_ids) < max_seq_length:
                label_ids.append(0)
                label_mask.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(valid) == max_seq_length
            assert len(label_mask) == max_seq_length

            if self.multi_attention is not None:
                wordlist = example.word
                matching_position = example.matrix
                channel_ids = []
                word_ids = []
                for i in range(self.cat_num):
                    channel_ids.append(i)
                    word_ids.append([])

                matching_matrix = np.zeros((self.cat_num, max_seq_length, max_word_size), dtype=np.int)
                for i in range(len(wordlist)):
                    if len(wordlist[i]) > max_word_size:
                        wordlist[i] = wordlist[i][:max_word_size]

                for i in range(len(wordlist)):
                    for word in wordlist[i]:
                        if word == '':
                            continue
                        try:
                            word_ids[i].append(self.gram2id[word])
                        except KeyError:
                            print(word)
                            print(wordlist)
                            print(textlist)
                            raise KeyError()

                for i in range(len(word_ids)):
                    while len(word_ids[i]) < max_word_size:
                        word_ids[i].append(0)

                for i in range(len(matching_position)):
                    for position in matching_position[i]:
                        char_p = position[0] + 1
                        word_p = position[1]
                        if char_p > max_seq_length - 2 or word_p > max_word_size - 1:
                            continue
                        else:
                            matching_matrix[i][char_p][word_p] = 1

                assert len(word_ids) == self.cat_num
                assert len(word_ids[0]) == max_word_size
            else:
                word_ids = None
                matching_matrix = None
                channel_ids = None

            if self.zen_ngram_dict is not None:
                ngram_matches = []
                #  Filter the ngram segment from 2 to 7 to check whether there is a ngram
                for p in range(2, 8):
                    for q in range(0, len(tokens) - p + 1):
                        character_segment = tokens[q:q + p]
                        # j is the starting position of the ngram
                        # i is the length of the current ngram
                        character_segment = tuple(character_segment)
                        if character_segment in self.zen_ngram_dict.ngram_to_id_dict:
                            ngram_index = self.zen_ngram_dict.ngram_to_id_dict[character_segment]
                            ngram_matches.append([ngram_index, q, p, character_segment])

                # random.shuffle(ngram_matches)
                ngram_matches = sorted(ngram_matches, key=lambda s: s[0])

                max_ngram_in_seq_proportion = math.ceil(
                    (len(tokens) / max_seq_length) * self.zen_ngram_dict.max_ngram_in_seq)
                if len(ngram_matches) > max_ngram_in_seq_proportion:
                    ngram_matches = ngram_matches[:max_ngram_in_seq_proportion]

                ngram_ids = [ngram[0] for ngram in ngram_matches]
                ngram_positions = [ngram[1] for ngram in ngram_matches]
                ngram_lengths = [ngram[2] for ngram in ngram_matches]
                ngram_tuples = [ngram[3] for ngram in ngram_matches]
                ngram_seg_ids = [0 if position < (len(tokens) + 2) else 1 for position in ngram_positions]

                ngram_mask_array = np.zeros(self.zen_ngram_dict.max_ngram_in_seq, dtype=np.bool)
                ngram_mask_array[:len(ngram_ids)] = 1

                # record the masked positions
                ngram_positions_matrix = np.zeros(shape=(max_seq_length, self.zen_ngram_dict.max_ngram_in_seq),
                                                  dtype=np.int32)
                for i in range(len(ngram_ids)):
                    ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

                # Zero-pad up to the max ngram in seq length.
                padding = [0] * (self.zen_ngram_dict.max_ngram_in_seq - len(ngram_ids))
                ngram_ids += padding
                ngram_lengths += padding
                ngram_seg_ids += padding
            else:
                ngram_ids = None
                ngram_positions_matrix = None
                ngram_lengths = None
                ngram_tuples = None
                ngram_seg_ids = None
                ngram_mask_array = None

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask,
                              word_ids=word_ids,
                              matching_matrix=matching_matrix,
                              channel_ids=channel_ids,
                              ngram_ids=ngram_ids,
                              ngram_positions=ngram_positions_matrix,
                              ngram_lengths=ngram_lengths,
                              ngram_tuples=ngram_tuples,
                              ngram_seg_ids=ngram_seg_ids,
                              ngram_masks=ngram_mask_array
                              ))
        return features

    def feature2input(self, device, feature):
        all_input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in feature], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in feature], dtype=torch.long)
        input_ids = all_input_ids.to(device)
        input_mask = all_input_mask.to(device)
        segment_ids = all_segment_ids.to(device)
        label_ids = all_label_ids.to(device)
        valid_ids = all_valid_ids.to(device)
        l_mask = all_lmask_ids.to(device)
        if self.multi_attention is not None:
            all_word_ids = torch.tensor([f.word_ids for f in feature], dtype=torch.long)
            all_matching_matrix = torch.tensor([f.matching_matrix for f in feature], dtype=torch.long)
            all_word_mask = torch.tensor([f.matching_matrix for f in feature], dtype=torch.float)
            all_channel_ids = torch.tensor([f.channel_ids for f in feature], dtype=torch.long)

            word_ids = all_word_ids.to(device)
            matching_matrix = all_matching_matrix.to(device)
            word_mask = all_word_mask.to(device)
            channel_ids = all_channel_ids.to(device)
        else:
            word_ids = None
            matching_matrix = None
            word_mask = None
            channel_ids = None
        if self.zen is not None:
            all_ngram_ids = torch.tensor([f.ngram_ids for f in feature], dtype=torch.long)
            all_ngram_positions = torch.tensor([f.ngram_positions for f in feature], dtype=torch.long)
            # all_ngram_lengths = torch.tensor([f.ngram_lengths for f in train_features], dtype=torch.long)
            # all_ngram_seg_ids = torch.tensor([f.ngram_seg_ids for f in train_features], dtype=torch.long)
            # all_ngram_masks = torch.tensor([f.ngram_masks for f in train_features], dtype=torch.long)

            ngram_ids = all_ngram_ids.to(device)
            ngram_positions = all_ngram_positions.to(device)
        else:
            ngram_ids = None
            ngram_positions = None
        return channel_ids, input_ids, input_mask, l_mask, label_ids, matching_matrix, ngram_ids, ngram_positions, segment_ids, valid_ids, word_ids, word_mask


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, word=None, matrix=None, sent_len=None, word_list_len=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.word = word
        self.matrix = matrix
        self.sent_len = sent_len
        self.word_list_len = word_list_len


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None,
                 word_ids=None, matching_matrix=None, channel_ids=None,
                 ngram_ids=None, ngram_positions=None, ngram_lengths=None,
                 ngram_tuples=None, ngram_seg_ids=None, ngram_masks=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.word_ids = word_ids
        self.matching_matrix = matching_matrix
        self.channel_ids = channel_ids

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks


def readsentence(filename):
    data = []

    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            label_list = ['S' for _ in range(len(line))]
            data.append((line, label_list))
    return data

