# -*- coding: utf-8 -*-
# file: lcf_template_atepc.py
# time: 2021/6/22
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


import numpy as np
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertForTokenClassification


class LCF_TEMPLATE_ATEPC(nn.Module):

    def __init__(self, bert_base_model, opt):
        super(LCF_TEMPLATE_ATEPC, self).__init__()
        config = bert_base_model.config
        self.bert4global = bert_base_model
        self.opt = opt
        self.bert4local = self.bert4global
        self.dropout = nn.Dropout(self.opt.dropout)

        self.num_labels = opt.num_labels
        self.classifier = nn.Linear(opt.hidden_dim, opt.num_labels)

    def get_batch_token_labels_bert_base_indices(self, labels):
        if labels is None:
            return
        # convert tags of BERT-SPC input to BERT-BASE format
        labels = labels.detach().cpu().numpy()
        for text_i in range(len(labels)):
            sep_index = np.argmax((labels[text_i] == 5))
            labels[text_i][sep_index + 1:] = 0
        return torch.tensor(labels).to(self.opt.device)

    def get_ids_for_local_context_extractor(self, text_indices):
        # convert BERT-SPC input to BERT-BASE format
        text_ids = text_indices.detach().cpu().numpy()
        for text_i in range(len(text_ids)):
            sep_index = np.argmax((text_ids[text_i] == self.opt.sep_indices))
            text_ids[text_i][sep_index + 1:] = 0
        return torch.tensor(text_ids).to(self.opt.device)

    def forward(self, input_ids_spc,
                token_type_ids=None,
                attention_mask=None,
                labels=None,
                polarity=None,
                valid_ids=None,
                attention_mask_label=None,
                lcf_cdm_vec=None,
                lcf_cdw_vec=None
                ):
        lcf_cdm_vec = lcf_cdm_vec.unsqueeze(2) if lcf_cdm_vec is not None else None
        lcf_cdw_vec = lcf_cdw_vec.unsqueeze(2) if lcf_cdw_vec is not None else None
        raise NotImplementedError('This is a template ATEPC model based on LCF, '
                                  'please implement your model use this template.')
