# -*- coding: utf-8 -*-
# file: functional.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.

import math
import os
import random

import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from ..models.bert_base import BERT_BASE
from ..models.bert_spc import BERT_SPC
from ..models.lcf_bert import LCF_BERT
from ..models.slide_lcf_bert import SLIDE_LCF_BERT
from ..utils.data_utils_for_training import Tokenizer4Bert, ABSADataset
import pickle


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        # opt.learning_rate = 2e-5
        self.bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(opt.pretrained_bert_name, do_lower_case=True)
        tokenizer = Tokenizer4Bert(self.bert_tokenizer, opt.max_seq_len)
        self.model = opt.model_class(self.bert, opt).to(opt.device)

        trainset = ABSADataset(opt.train_dataset_path, tokenizer, opt)
        self.train_data_loader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True, pin_memory=True)

        if opt.device.type == 'cuda':
            print("cuda memory allocated:{}".format(torch.cuda.memory_allocated(device=opt.device.index)))

        self._log_write_args()

    def _log_write_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params (with unfreezed bert)
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _save_model(self, model, save_path, mode=0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        if mode == 0 or 'bert' not in self.opt.model_name:
            save_path = save_path + '/' + self.opt.model_name + '_trained/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.model.state_dict(), save_path + self.opt.model_name + '.state_dict')  # save the state dict
            pickle.dump(self.opt, open(save_path + 'model.config', 'wb'))
        else:
            # save the fine-tuned bert model
            model_output_dir = save_path + '_fine-tuned'
            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)
            output_model_file = os.path.join(model_output_dir, 'pytorch_model.bin')
            output_config_file = os.path.join(model_output_dir, 'bert_config.json')

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            self.bert_tokenizer.save_vocabulary(model_output_dir)

    def _train(self, criterion, lca_criterion, optimizer):
        global_step = 0
        for epoch in range(self.opt.num_epoch):
            for i_batch, sample_batched in tqdm(enumerate(self.train_data_loader),
                                                postfix='training... epoch:'
                                                + str(epoch)
                                                ):

                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)
                if 'lca' in self.opt.model_name:
                    sen_logits, lca_logits, lca_ids = outputs
                    sen_loss = criterion(sen_logits, targets)
                    lcp_loss = lca_criterion(lca_logits, lca_ids)
                    loss = (1 - self.opt.sigma) * sen_loss + self.opt.sigma * lcp_loss
                else:
                    sen_logits = outputs
                    loss = criterion(sen_logits, targets)

                loss.backward()
                optimizer.step()

        self._save_model(self.model, self.opt.model_path_to_save, mode=0)
        print('trained model saved in: {}'.format(self.opt.model_path_to_save))

    def run(self):

        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        lca_criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        self._reset_params()
        self._train(criterion, lca_criterion, optimizer)

        return self.model, self.opt


def train_by_single_config(opt):
    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'bert_base': BERT_BASE,
        'bert_spc': BERT_SPC,
        'lcf_bert': LCF_BERT,
        'slide_lcf_bert': SLIDE_LCF_BERT,
        'slide_lcfs_bert': SLIDE_LCF_BERT,
        'lcfs_bert': LCF_BERT,
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_
    }

    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = ABSADataset.input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]

    opt.device = torch.device(opt.device)
    ins = Instructor(opt)
    return ins.run()  # _reset_params in every repeat
