# -*- coding: utf-8 -*-
# file: functional.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.

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
    def __init__(self, opt, trained_model_path):
        self.opt = opt
        # opt.learning_rate = 2e-5
        # Use any type of BERT to initialize your model.
        # The weights of loaded BERT will be covered after loading state_dict
        # self.bert = BertModel.from_pretrained('bert-base-uncased')

        if opt.seed is not None:
            random.seed(opt.seed)
            numpy.random.seed(opt.seed)
            torch.manual_seed(opt.seed)
            torch.cuda.manual_seed(opt.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

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

        self.bert_tokenizer = BertTokenizer.from_pretrained(opt.pretrained_bert_name, do_lower_case=True)
        self.tokenizer = Tokenizer4Bert(self.bert_tokenizer, opt.max_seq_len)
        self.dataset = ABSADataset(tokenizer=self.tokenizer, opt=opt)
        self.opt.inputs_cols = self.dataset.input_colses[self.opt.model_name]
        self.opt.initializer = self.initializers[self.opt.initializer]
        self.bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = self.model_class[opt.model_name](self.bert, opt).to(opt.device)
        self.model.load_state_dict(torch.load(trained_model_path))

    def batch_infer(self, test_dataset_path=None):
        if test_dataset_path:
            self.dataset.prepare_infer_dataset(test_dataset_path)
        else:
            raise RuntimeError('Please specify your dataset path!')

        # load training set
        self.test_data_loader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False)

        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        sentiments = {0: 'Negative', 1: "Neutral", 2: 'Positive', -999: ''}
        Correct = {True: 'Correct', False: 'Wrong'}
        results = []
        with torch.no_grad():
            self.model.eval()
            for _, sample in enumerate(self.test_data_loader):
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
                result['sentiment'] = sentiments[int(t_probs.argmax(axis=-1))]
                result['ref_sentiment'] = sentiments[real_sent]
                result['infer result'] = Correct[sent == real_sent]
                results.append(result)
                print(sample['text_raw'][0])
                print('{} --> {}'.format(aspect, sentiments[sent])) if real_sent == -999 \
                    else print('{} --> {}  Real Polarity: {} ({})'.format(aspect, sentiments[sent],
                                                                          sentiments[real_sent],
                                                                          Correct[sent == real_sent]))

        return results

    def infer(self, text=None):
        if text:
            self.dataset.prepare_infer_sample(text)
        else:
            raise RuntimeError('Please specify your dataset path!')
            # load training set
        self.test_data_loader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False)
        _params = filter(lambda p: p.requires_grad, self.model.parameters())

        sentiments = {0: 'Negative', 1: "Neutral", 2: 'Positive', -999: ''}
        Correct = {True: 'Correct', False: 'Wrong'}
        results = []
        with torch.no_grad():
            self.model.eval()
            for _, sample in enumerate(self.test_data_loader):
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
                print(sample['text_raw'][0])
                print('{} --> {}'.format(aspect, sentiments[sent])) if real_sent == -999 \
                    else print('{} --> {}  Real Polarity: {} ({})'.format(aspect, sentiments[sent],
                                                                          sentiments[real_sent],
                                                                          Correct[sent == real_sent]))

        return results
