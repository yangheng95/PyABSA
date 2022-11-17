# -*- coding: utf-8 -*-
# file: ensembler.py
# time: 2021/11/17
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import copy
import os
import pickle
import re
from hashlib import sha256

from findfile import find_cwd_dir
from termcolor import colored
from torch import nn
from torch.nn import ModuleList

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModel

from pyabsa.tasks.AspectPolarityClassification.models.__classic__ import GloVeAPCModelList
from pyabsa.tasks.AspectPolarityClassification.models.__lcf__ import APCModelList
from pyabsa.tasks.AspectPolarityClassification.models.__plm__ import BERTBaselineAPCModelList
from pyabsa.tasks.AspectPolarityClassification.dataset_utils.__classic__.data_utils_for_training import GloVeABSADataset
from pyabsa.tasks.AspectPolarityClassification.dataset_utils.__lcf__.data_utils_for_training import ABSADataset
from pyabsa.tasks.AspectPolarityClassification.dataset_utils.__plm__.data_utils_for_training import BERTBaselineABSADataset
from pyabsa.framework.tokenizer_class.tokenizer_class import PretrainedTokenizer, Tokenizer, build_embedding_matrix


def model_pool_check(models):
    set1 = set([model for model in models if hasattr(APCModelList, model.__name__)])
    set2 = set([model for model in models if hasattr(BERTBaselineAPCModelList, model.__name__)])
    set3 = set([model for model in models if hasattr(GloVeAPCModelList, model.__name__)])
    if set1 and set2 or set1 and set3 or set2 and set3:
        raise RuntimeError('The APCEnsembler only support the models in same type. ')


class APCEnsembler(nn.Module):
    def __init__(self, config, load_dataset=True, **kwargs):
        super(APCEnsembler, self).__init__()
        self.config = config

        models = [config.model] if not isinstance(config.model, list) else config.model
        model_pool_check(models)

        self.config.inputs_cols = set()
        for model in models:
            self.config.inputs_cols |= set(model.inputs)
        self.config.inputs_cols = sorted(self.config.inputs_cols)
        self.inputs_cols = self.config.inputs_cols

        self.models = ModuleList()

        self.tokenizer = None
        self.bert = None
        self.embedding_matrix = None
        self.train_set = None
        self.test_set = None
        self.valid_set = None
        self.test_dataloader = None
        self.valid_dataloader = None

        for i in range(len(models)):

            config_str = re.sub(r'<.*?>', '', str(sorted([str(self.config.args[k]) for k in self.config.args if k != 'seed'])))
            hash_tag = sha256(config_str.encode()).hexdigest()
            cache_path = '{}.{}.dataset.{}.cache'.format(self.config.model_name, self.config.dataset_name, hash_tag)

            if load_dataset and os.path.exists(cache_path) and not self.config.overwrite_cache:
                print(colored('Loading dataset cache: {}'.format(cache_path), 'green'))
                with open(cache_path, mode='rb') as f_cache:
                    self.train_set, self.valid_set, self.test_set, self.config = pickle.load(f_cache)
                    config.update(self.config)
                    config.args_call_count.update(self.config.args_call_count)
            if hasattr(APCModelList, models[i].__name__):
                try:

                    if kwargs.get('offline', False):
                        self.tokenizer = AutoTokenizer.from_pretrained(find_cwd_dir(self.config.pretrained_bert.split('/')[-1]), do_lower_case='uncased' in self.config.pretrained_bert)
                        self.bert = AutoModel.from_pretrained(find_cwd_dir(self.config.pretrained_bert.split('/')[-1])) if not self.bert else self.bert  # share the underlying bert between models
                    else:
                        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_bert, do_lower_case='uncased' in self.config.pretrained_bert)
                        self.bert = AutoModel.from_pretrained(self.config.pretrained_bert) if not self.bert else self.bert
                except ValueError as e:
                    print('Init pretrained model failed, exception: {}'.format(e))
                    exit(-1)

                if load_dataset and not os.path.exists(cache_path) or self.config.overwrite_cache:
                    self.train_set = ABSADataset(self.config, self.tokenizer, dataset_type='train') if not self.train_set else self.train_set
                    self.test_set = ABSADataset(self.config, self.tokenizer, dataset_type='test') if not self.test_set else self.test_set
                    self.valid_set = ABSADataset(self.config, self.tokenizer, dataset_type='valid') if not self.valid_set else self.valid_set
                self.models.append(models[i](self.bert, self.config))

            elif hasattr(BERTBaselineAPCModelList, models[i].__name__):
                self.tokenizer = PretrainedTokenizer(self.config) if not self.tokenizer else self.tokenizer
                self.bert = AutoModel.from_pretrained(self.config.pretrained_bert) if not self.bert else self.bert

                if load_dataset and not os.path.exists(cache_path) or self.config.overwrite_cache:
                    self.train_set = BERTBaselineABSADataset(self.config, self.tokenizer, dataset_type='train') if not self.train_set else self.train_set
                    self.test_set = BERTBaselineABSADataset(self.config, self.tokenizer, dataset_type='test') if not self.test_set else self.test_set
                    self.valid_set = BERTBaselineABSADataset(self.config, self.tokenizer, dataset_type='valid') if not self.valid_set else self.valid_set
                self.models.append(models[i](copy.deepcopy(self.bert) if self.config.deep_ensemble else self.bert, self.config))

            elif hasattr(GloVeAPCModelList, models[i].__name__):
                self.tokenizer = Tokenizer.build_tokenizer(
                    config=self.config,
                    cache_path='{0}_tokenizer.dat'.format(os.path.basename(config.dataset_name)),
                ) if not self.tokenizer else self.tokenizer
                self.embedding_matrix = build_embedding_matrix(
                    config=self.config,
                    tokenizer=self.tokenizer,
                    cache_path='{0}_{1}_embedding_matrix.dat'.format(str(config.embed_dim), os.path.basename(config.dataset_name)),
                ) if not self.embedding_matrix else self.embedding_matrix

                if load_dataset and not os.path.exists(cache_path) or self.config.overwrite_cache:
                    self.train_set = GloVeABSADataset(self.config, self.tokenizer, dataset_type='train') if not self.train_set else self.train_set
                    self.test_set = GloVeABSADataset(self.config, self.tokenizer, dataset_type='test') if not self.test_set else self.test_set
                    self.valid_set = GloVeABSADataset(self.config, self.tokenizer, dataset_type='valid') if not self.valid_set else self.valid_set

                self.models.append(models[i](copy.deepcopy(self.embedding_matrix) if self.config.deep_ensemble else self.embedding_matrix, self.config))
                self.config.tokenizer = self.tokenizer
                self.config.embedding_matrix = self.embedding_matrix

            if self.config.cache_dataset and not os.path.exists(cache_path) and not self.config.overwrite_cache:
                print(colored('Caching dataset... please remove cached dataset if any problem happens.', 'red'))
                with open(cache_path, mode='wb') as f_cache:
                    pickle.dump((self.train_set, self.valid_set, self.test_set, self.config), f_cache)

            if load_dataset:
                train_sampler = RandomSampler(self.train_set if not self.train_set else self.train_set)
                self.train_dataloader = DataLoader(self.train_set, batch_size=self.config.batch_size, pin_memory=True, sampler=train_sampler)
                if self.test_set:
                    test_sampler = SequentialSampler(self.test_set if not self.test_set else self.test_set)
                    self.test_dataloader = DataLoader(self.test_set, batch_size=self.config.batch_size, pin_memory=True, sampler=test_sampler)
                if self.valid_set:
                    valid_sampler = SequentialSampler(self.valid_set if not self.valid_set else self.valid_set)
                    self.valid_dataloader = DataLoader(self.valid_set, batch_size=self.config.batch_size, pin_memory=True, sampler=valid_sampler)

        self.dense = nn.Linear(config.output_dim * len(models), config.output_dim)

    def forward(self, inputs):
        outputs = [self.models[i](inputs) for i in range(len(self.models))]
        loss = torch.tensor(0., requires_grad=True)
        if 'ensemble_mode' not in self.config:
            self.config.ensemble_mode = 'cat'
        if len(outputs) > 1:
            for i, out in enumerate(outputs):
                if self.config.ensemble_mode == 'cat':
                    logits = torch.cat((logits, out['logits']), dim=-1) if i != 0 else out['logits']
                elif self.config.ensemble_mode == 'mean':
                    logits = logits + out['logits'] if i != 0 else out['logits']
                else:
                    raise KeyError('Invalid ensemble_mode!')
                if 'loss' in out:
                    loss = loss + out['loss'] if i != 0 else out['loss']

            if 'ensemble_mode' not in self.config or self.config.ensemble_mode == 'cat':
                logits = self.dense(logits)
            elif self.config.ensemble_mode == 'mean':
                logits = logits / len(self.models)
        else:
            logits = outputs[0]['logits']
            loss = outputs[0]['loss'] if 'loss' in outputs[0] else loss
        return {'logits': logits, 'loss': loss.to(logits.device)}
