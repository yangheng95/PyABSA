# -*- coding: utf-8 -*-
# file: ensembler.py
# time: 2021/11/17
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import copy
import os
import pickle

from torch import nn
from torch.nn import ModuleList

import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel

from pyabsa.functional.dataset import ABSADatasetList
from pyabsa.utils.pyabsa_utils import TransformerConnectionError

from ..models import BERTBaselineAPCModelList, GloVeAPCModelList, APCModelList
from ..classic.__bert__.dataset_utils.data_utils_for_training import (Tokenizer4Pretraining,
                                                                      BERTBaselineABSADataset)
from ..classic.__glove__.dataset_utils.data_utils_for_training import (build_tokenizer,
                                                                       build_embedding_matrix,
                                                                       GloVeABSADataset)
from ..dataset_utils.data_utils_for_training import ABSADataset


def model_pool_check(models):
    set1 = set([model for model in models if hasattr(APCModelList, model.__name__)])
    set2 = set([model for model in models if hasattr(BERTBaselineAPCModelList, model.__name__)])
    set3 = set([model for model in models if hasattr(GloVeAPCModelList, model.__name__)])
    if set1 and set2 or set1 and set3 or set2 and set3:
        raise RuntimeError('The APCEnsembler only support the models in same type. ')


class APCEnsembler(nn.Module):
    def __init__(self, opt, load_dataset=True):
        super(APCEnsembler, self).__init__()
        self.opt = opt

        models = [opt.model] if not isinstance(opt.model, list) else opt.model
        model_pool_check(models)

        self.opt.inputs_cols = set()
        for model in models:
            self.opt.inputs_cols |= set(model.inputs)
        self.inputs_cols = self.opt.inputs_cols

        self.models = ModuleList()

        self.tokenizer = None
        self.bert = None
        self.embedding_matrix = None
        self.train_set = None
        self.test_set = None
        self.test_dataloader = None

        for i in range(len(models)):

            # init BERT-based model and dataset
            train_set_cache_path = '{}.{}.train_set.cache'.format(self.opt.model_name, self.opt.dataset_name)
            test_set_cache_path = '{}.{}.test_set.cache'.format(self.opt.model_name, self.opt.dataset_name)

            if hasattr(APCModelList, models[i].__name__):
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.opt.pretrained_bert, do_lower_case='uncased' in self.opt.pretrained_bert) if not self.tokenizer else self.tokenizer
                    self.bert = AutoModel.from_pretrained(self.opt.pretrained_bert) if not self.bert else self.bert  # share the underlying bert between models
                except ValueError as e:
                    print('Init pretrained model failed, exception: {}'.format(e))
                    raise TransformerConnectionError()
                if load_dataset:
                    if os.path.exists(train_set_cache_path) and os.path.exists(test_set_cache_path):
                        print('Loading APC dataset cache:', train_set_cache_path, test_set_cache_path)
                        self.train_set = pickle.load(open(train_set_cache_path, mode='rb'))
                        self.test_set = pickle.load(open(test_set_cache_path, mode='rb'))
                    else:
                        self.train_set = ABSADataset(self.opt.dataset_file['train'], self.tokenizer, self.opt) if not self.train_set else self.train_set
                        if self.opt.dataset_file['test']:
                            self.test_set = ABSADataset(self.opt.dataset_file['test'], self.tokenizer, self.opt) if not self.test_set else self.test_set

                # init the model behind the construction of apc_datasets in case of updating polarities_dim
                self.models.append(models[i](self.bert, self.opt))

            elif hasattr(BERTBaselineAPCModelList, models[i].__name__):
                self.tokenizer = Tokenizer4Pretraining(self.opt.max_seq_len, self.opt.pretrained_bert) if not self.tokenizer else self.tokenizer
                self.bert = AutoModel.from_pretrained(self.opt.pretrained_bert) if not self.bert else self.bert

                if load_dataset:
                    if os.path.exists(train_set_cache_path) and os.path.exists(test_set_cache_path):
                        print('Loading APC dataset cache:', train_set_cache_path, test_set_cache_path)
                        self.train_set = pickle.load(open(train_set_cache_path, mode='rb'))
                        self.test_set = pickle.load(open(test_set_cache_path, mode='rb'))
                    else:
                        self.train_set = BERTBaselineABSADataset(self.opt.dataset_file['train'], self.tokenizer, self.opt) if not self.train_set else self.train_set
                        if self.opt.dataset_file['test']:
                            self.test_set = BERTBaselineABSADataset(self.opt.dataset_file['test'], self.tokenizer, self.opt) if not self.test_set else self.test_set

                self.models.append(models[i](copy.deepcopy(self.bert) if self.opt.deep_ensemble else self.bert, self.opt))

            elif hasattr(GloVeAPCModelList, models[i].__name__):
                # init GloVe-based model and dataset
                if hasattr(ABSADatasetList, opt.dataset_name):
                    opt.dataset_name = os.path.join(os.getcwd(), opt.dataset_name)
                    if not os.path.exists(os.path.join(os.getcwd(), opt.dataset_name)):
                        os.mkdir(os.path.join(os.getcwd(), opt.dataset_name))

                self.tokenizer = build_tokenizer(
                    dataset_list=opt.dataset_file,
                    max_seq_len=opt.max_seq_len,
                    dat_fname='{0}_tokenizer.dat'.format(os.path.basename(opt.dataset_name)),
                    opt=self.opt
                ) if not self.tokenizer else self.tokenizer
                self.embedding_matrix = build_embedding_matrix(
                    word2idx=self.tokenizer.word2idx,
                    embed_dim=opt.embed_dim,
                    dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), os.path.basename(opt.dataset_name)),
                    opt=self.opt
                ) if not self.embedding_matrix else self.embedding_matrix

                if load_dataset:
                    if os.path.exists(train_set_cache_path) and os.path.exists(test_set_cache_path):
                        print('Loading APC dataset cache:', train_set_cache_path, test_set_cache_path)
                        self.train_set = pickle.load(open(train_set_cache_path, mode='rb'))
                        self.train_set = pickle.load(open(test_set_cache_path, mode='rb'))
                    else:
                        self.train_set = GloVeABSADataset(self.opt.dataset_file['train'], self.tokenizer, self.opt) if not self.train_set else self.train_set
                        if self.opt.dataset_file['test']:
                            self.test_set = GloVeABSADataset(self.opt.dataset_file['test'], self.tokenizer, self.opt) if not self.test_set else self.test_set

                self.models.append(models[i](copy.deepcopy(self.embedding_matrix) if self.opt.deep_ensemble else self.embedding_matrix, self.opt))

            if self.opt.cache_dataset:
                print('Caching dataset... please remove cached dataset if change model or dataset')
                pickle.dump(self.train_set, open(train_set_cache_path, mode='wb'))
                pickle.dump(self.test_set, open(test_set_cache_path, mode='wb'))

            train_sampler = DistributedSampler(self.train_set if not self.train_set else self.train_set)
            test_sampler = DistributedSampler(self.test_set if not self.test_set else self.test_set)
            self.train_dataloader = DataLoader(self.train_set, batch_size=self.opt.batch_size, sampler=train_sampler)
            self.test_dataloader = DataLoader(self.test_set, batch_size=self.opt.batch_size, sampler=test_sampler)

        self.dense = nn.Linear(opt.polarities_dim * len(models), opt.polarities_dim)

    def forward(self, inputs):
        outputs = [self.models[i](inputs) for i in range(len(self.models))]
        loss = torch.tensor(0., requires_grad=True)
        if 'ensemble_mode' not in self.opt:
            self.opt.ensemble_mode = 'cat'
        if len(outputs) > 1:
            for i, out in enumerate(outputs):
                if self.opt.ensemble_mode == 'cat':
                    logits = torch.cat((logits, out['logits']), dim=-1) if i != 0 else out['logits']
                elif self.opt.ensemble_mode == 'mean':
                    logits = logits + out['logits'] if i != 0 else out['logits']
                else:
                    raise KeyError('Invalid ensemble_mode!')
                if 'loss' in out:
                    loss = loss + out['loss'] if i != 0 else out['loss']

            if 'ensemble_mode' not in self.opt or self.opt.ensemble_mode == 'cat':
                logits = self.dense(logits)
            elif self.opt.ensemble_mode == 'mean':
                logits = logits / len(self.models)
        else:
            logits = outputs[0]['logits']
            loss = outputs[0]['loss'] if 'loss' in outputs[0] else loss
        return {'logits': logits, 'loss': loss.to(logits.device)}
