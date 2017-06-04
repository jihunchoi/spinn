import random

import torch
import jsonlines
from nltk import word_tokenize


class SNLIReader(object):

    def __init__(self, data_path, token_vocab, trans_vocab, label_vocab,
                 max_length=None):
        self.token_vocab = token_vocab
        self.trans_vocab = trans_vocab
        self.label_vocab = label_vocab
        self._max_length = max_length
        self._data = []
        with jsonlines.open(data_path) as reader:
            for obj in reader:
                converted = self._convert_obj(obj)
                if converted:
                    self._data.append(converted)

    @staticmethod
    def _convert_parse(parse):
        tokens = parse.split()
        transition = []
        for token in tokens:
            if token == ')':
                transition.append('REDUCE')
            elif token == '(':
                continue
            else:
                transition.append('SHIFT')
        return transition

    def _convert_obj(self, obj):
        pre = obj['sentence1']
        pre_parse = obj['sentence1_binary_parse']
        hyp = obj['sentence2']
        hyp_parse = obj['sentence2_binary_parse']
        label = obj['gold_label']
        if label == '-':
            return None
        pre_tokens = [self.token_vocab.word_to_id(w)
                      for w in word_tokenize(pre.lower())]
        pre_trans = [self.trans_vocab.word_to_id(w)
                     for w in self._convert_parse(pre_parse)]
        hyp_tokens = [self.token_vocab.word_to_id(w)
                      for w in word_tokenize(hyp.lower())]
        hyp_trans = [self.trans_vocab.word_to_id(w)
                     for w in self._convert_parse(hyp_parse)]
        label = self.label_vocab.word_to_id(label)
        if (len(pre_tokens) > self._max_length
                or len(hyp_tokens) > self._max_length):
            return None
        return {'pre_tokens': pre_tokens, 'pre_trans': pre_trans,
                'hyp_tokens': hyp_tokens, 'hyp_trans': hyp_trans,
                'label': label}

    def _preprocess_batch(self, batch):
        max_trans_length = self._max_length*2 - 1
        shift_id = self.trans_vocab.word_to_id('SHIFT')
        pad_id = self.token_vocab.pad_id

        def pad(tokens, trans):
            num_shift_added = max_trans_length - len(trans)
            left_trans_pad = [shift_id] * num_shift_added
            trans = left_trans_pad + trans
            left_tokens_pad = [pad_id] * num_shift_added
            right_tokens_pad = (
                [pad_id] * (max_trans_length - len(tokens) - num_shift_added))
            tokens = left_tokens_pad + tokens + right_tokens_pad
            return tokens, trans

        preprocessed = {'pre_tokens': [], 'pre_trans': [], 'pre_num_trans': [],
                        'hyp_tokens': [], 'hyp_trans': [], 'hyp_num_trans': [],
                        'label': []}
        for d in batch:
            pre_tokens = d['pre_tokens']
            pre_trans = d['pre_trans']
            pre_num_trans = len(pre_trans)
            hyp_tokens = d['hyp_tokens']
            hyp_trans = d['hyp_trans']
            hyp_num_trans = len(hyp_trans)
            label = d['label']
            pre_tokens, pre_trans = pad(tokens=pre_tokens, trans=pre_trans)
            hyp_tokens, hyp_trans = pad(tokens=hyp_tokens, trans=hyp_trans)
            preprocessed['pre_tokens'].append(pre_tokens)
            preprocessed['pre_trans'].append(pre_trans)
            preprocessed['pre_num_trans'].append(pre_num_trans)
            preprocessed['hyp_tokens'].append(hyp_tokens)
            preprocessed['hyp_trans'].append(hyp_trans)
            preprocessed['hyp_num_trans'].append(hyp_num_trans)
            preprocessed['label'].append(label)

        for k in preprocessed:
            preprocessed[k] = torch.LongTensor(preprocessed[k])
            preprocessed[k].pin_memory()
        return preprocessed

    def shuffle(self):
        random.shuffle(self._data)

    def batch_iterator(self, batch_size):
        num_data = len(self._data)
        for i in range(0, num_data, batch_size):
            batch = self._data[i:i+batch_size]
            batch = self._preprocess_batch(batch)
            yield batch
