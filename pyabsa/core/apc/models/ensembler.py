# -*- coding: utf-8 -*-
# file: ensembler.py
# time: 2021/11/17
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import copy
import os

from torch import nn
from torch.nn import ModuleList

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, BertModel

from pyabsa.functional.dataset import ABSADatasetList

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

        self.opt.inputs = set()
        for model in models:
            self.opt.inputs |= set(model.inputs)
        self.inputs = self.opt.inputs

        self.models = ModuleList()

        self.tokenizer = None
        self.bert = None
        self.embedding_matrix = None
        self.train_set = None
        self.test_set = None
        self.test_dataloader = None

        for i in range(len(models)):

            # init BERT-based model and dataset
            if hasattr(APCModelList, models[i].__name__):
                self.tokenizer = AutoTokenizer.from_pretrained(self.opt.pretrained_bert, do_lower_case=True) if not self.tokenizer else self.tokenizer
                # self.bert = AutoModel.from_pretrained(self.opt.pretrained_bert) if not self.bert else self.bert
                self.bert = AutoModel.from_pretrained(self.opt.pretrained_bert) if not self.bert else copy.deepcopy(self.bert)

                if load_dataset:
                    self.train_set = ABSADataset(self.opt.dataset_file['train'], self.tokenizer, self.opt) if not self.train_set else self.train_set
                    if self.opt.dataset_file['test']:
                        self.test_set = ABSADataset(self.opt.dataset_file['test'], self.tokenizer, self.opt) if not self.test_set else self.test_set
                        self.test_dataloader = DataLoader(dataset=self.test_set, batch_size=self.opt.batch_size, shuffle=False) if not self.test_dataloader else self.test_dataloader

                # init the model behind the construction of apc_datasets in case of updating polarities_dim
                self.models.append(models[i](self.bert, self.opt))

            elif hasattr(BERTBaselineAPCModelList, models[i].__name__):
                self.tokenizer = Tokenizer4Pretraining(self.opt.max_seq_len, self.opt.pretrained_bert) if not self.tokenizer else self.tokenizer
                self.bert = AutoModel.from_pretrained(self.opt.pretrained_bert) if not self.bert else copy.deepcopy(self.bert)

                if load_dataset:
                    self.train_set = BERTBaselineABSADataset(self.opt.dataset_file['train'], self.tokenizer, self.opt) if not self.train_set else self.train_set
                    if self.opt.dataset_file['test']:
                        self.test_set = BERTBaselineABSADataset(self.opt.dataset_file['test'], self.tokenizer, self.opt) if not self.test_set else self.test_set
                        self.test_dataloader = DataLoader(dataset=self.test_set, batch_size=self.opt.batch_size, shuffle=False) if not self.test_dataloader else self.test_dataloader
                # init the model behind the construction of apc_datasets in case of updating polarities_dim
                self.models.append(models[i](self.bert, self.opt))

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
                ) if not self.embedding_matrix else copy.deepcopy(self.embedding_matrix)

                if load_dataset:
                    self.train_set = GloVeABSADataset(self.opt.dataset_file['train'], self.tokenizer, self.opt) if not self.train_set else self.train_set
                    if self.opt.dataset_file['test']:
                        self.test_set = GloVeABSADataset(self.opt.dataset_file['test'], self.tokenizer, self.opt) if not self.test_set else self.test_set
                        self.test_dataloader = DataLoader(dataset=self.test_set, batch_size=self.opt.batch_size, shuffle=False) if not self.test_dataloader else self.test_dataloader

                self.models.append(models[i](self.embedding_matrix, opt))

        self.dense = nn.Linear(opt.polarities_dim * len(models), opt.polarities_dim)

    def forward(self, inputs):
        outputs = [self.models[i](inputs) for i in range(len(self.models))]
        loss = torch.tensor(0., requires_grad=True)
        for i, out in enumerate(outputs):
            if 'ensemble_mode' not in self.opt or self.opt.ensemble_mode == 'cat':
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

        return {'logits': logits, 'loss': loss.to(logits.device)}
