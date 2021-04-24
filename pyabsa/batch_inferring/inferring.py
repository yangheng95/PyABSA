# -*- coding: utf-8 -*-
# file: functional.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.

import os
import pickle
import random

import numpy
import torch
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

from ..models.bert_base import BERT_BASE
from ..models.bert_spc import BERT_SPC
from ..models.lcf_bert import LCF_BERT
from ..models.slide_lcf_bert import SLIDE_LCF_BERT
from ..utils.data_utils_for_inferring import Tokenizer4Bert, ABSADataset


class INFER_MODEL:
    def __init__(self, trained_model_path=None, from_train_model=None):
        '''
            from_train_model: load inferring model from trained model
        '''

        self.model_class = {
            'bert_base': BERT_BASE,
            'bert_spc': BERT_SPC,
            'lcf_bert': LCF_BERT,
            'lcfs_bert': LCF_BERT,
            'slide_lcf_bert': SLIDE_LCF_BERT,
            'slide_lcfs_bert': SLIDE_LCF_BERT,
        }

        self.initializers = {
            'xavier_uniform_': torch.nn.init.xavier_uniform_,
            'xavier_normal_': torch.nn.init.xavier_normal,
            'orthogonal_': torch.nn.init.orthogonal_
        }

        if from_train_model:
            self.model = from_train_model[0]
            self.opt = from_train_model[1]
        else:
            try:
                state_dict_path = trained_model_path + '/' + \
                                  [p for p in os.listdir(trained_model_path) if '.state_dict' in p.lower()][0]
                config_path = trained_model_path + '/' + \
                              [p for p in os.listdir(trained_model_path) if '.config' in p.lower()][0]
                self.opt = pickle.load(open(config_path, 'rb'))
            except:
                raise FileNotFoundError('Can not find model from ' + trained_model_path)
            self.bert = BertModel.from_pretrained(self.opt.pretrained_bert_name)
            self.model = self.model_class[self.opt.model_name](self.bert, self.opt).to(self.opt.device)
            self.model.load_state_dict(torch.load(state_dict_path))

        self.bert_tokenizer = BertTokenizer.from_pretrained(self.opt.pretrained_bert_name, do_lower_case=True)
        self.tokenizer = Tokenizer4Bert(self.bert_tokenizer, self.opt.max_seq_len)
        self.dataset = ABSADataset(tokenizer=self.tokenizer, opt=self.opt)

        if self.opt.seed is not None:
            random.seed(self.opt.seed)
            numpy.random.seed(self.opt.seed)
            torch.manual_seed(self.opt.seed)
            torch.cuda.manual_seed(self.opt.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.opt.inputs_cols = self.dataset.input_colses[self.opt.model_name]
        self.opt.initializer = self.opt.initializer

    def batch_infer(self, test_dataset_path=None, save_result=True, clear_input_samples=True, ignore_error=True):
        try:
            test_dataset_path += [p for p in os.listdir(test_dataset_path) if 'infer' in p.lower()][0]
        except:
            raise RuntimeError('Can not find inference dataset!')
        if clear_input_samples:
            self.clear_input_samples()
        if test_dataset_path:
            self.dataset.prepare_infer_dataset(test_dataset_path, ignore_error=ignore_error)
        else:
            raise RuntimeError('Please specify your dataset path!')
        # load training set
        self.infer_dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False)
        return self._infer(save_path=test_dataset_path if save_result else None)

    def infer(self, text: str = None, clear_input_samples=True):
        if clear_input_samples:
            self.clear_input_samples()
        if text:
            self.dataset.prepare_infer_sample(text)
        else:
            raise RuntimeError('Please specify your dataset path!')
            # load training set
        self.infer_dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False)
        return self._infer(print_result=True)

    def _infer(self, save_path=None, print_result=True):

        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        sentiments = {0: 'Negative', 1: "Neutral", 2: 'Positive', -999: ''}
        Correct = {True: 'Correct', False: 'Wrong'}
        results = []
        if save_path:
            fout = open(save_path + '.infer.results', 'w', encoding='utf8')
        with torch.no_grad():
            self.model.eval()
            for _, sample in enumerate(self.infer_dataloader):
                result = {}
                inputs = [sample[col].to(self.opt.device) for col in self.opt.inputs_cols]
                self.model.eval()
                outputs = self.model(inputs)
                sen_logits = outputs
                t_probs = torch.softmax(sen_logits, dim=-1).cpu().numpy()
                sent = int(t_probs.argmax(axis=-1))
                real_sent = int(sample['polarity'])
                aspect = sample['aspect'][0]

                result['text'] = sample['text_raw'][0]
                result['aspect'] = sample['aspect'][0]
                result['sentiment'] = int(t_probs.argmax(axis=-1))
                result['ref_sentiment'] = sentiments[real_sent]
                result['infer result'] = Correct[sent == real_sent]
                results.append(result)
                line1 = sample['text_raw'][0]
                if real_sent == -999:
                    line2 = '{} --> {}'.format(aspect, sentiments[sent])
                else:
                    line2 = '{} --> {}  Real Polarity: {} ({})'.format(aspect,
                                                                       sentiments[sent],
                                                                       sentiments[real_sent],
                                                                       Correct[sent == real_sent]
                                                                       )
                if save_path:
                    fout.write(line1 + '\n')
                    fout.write(line2 + '\n')
                if print_result:
                    print(line1)
                    print(line2)
        return results

    def clear_input_samples(self):
        self.dataset.all_data = []
