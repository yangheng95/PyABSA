# -*- coding: utf-8 -*-
# file: lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import torch
import torch.nn as nn
from transformers import RobertaModel, T5ForConditionalGeneration, BartForConditionalGeneration, AutoTokenizer, \
    AutoModel

from pyabsa.networks.losses.ClassImblanceCE import ClassBalanceCrossEntropyLoss
from pyabsa.networks.losses.FocalLoss import FocalLoss
from pyabsa.networks.losses.LDAMLoss import LDAMLoss
from pyabsa.utils.pyabsa_utils import fprint


class BERT_MLP(nn.Module):
    MODEL_CLASSES = {
        't5-base': T5ForConditionalGeneration,
        'facebook/bart-base': BartForConditionalGeneration,
        'Salesforce/codet5-small': T5ForConditionalGeneration,
        'Salesforce/codet5-base': T5ForConditionalGeneration,
    }
    #
    inputs = ['source_ids', 'label', 'corrupt_label']

    def __init__(self, bert, config):
        super(BERT_MLP, self).__init__()
        self.config = config
        self.encoder = self.MODEL_CLASSES.get(self.config.pretrained_bert, AutoModel) \
            .from_pretrained(self.config.pretrained_bert)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_bert)
        self.classifier1 = nn.Linear(config.hidden_dim, 2)
        self.classifier2 = nn.Linear(config.hidden_dim, 2)
        if self.config.get('loss_fn', 'CrossEntropyLoss') == 'FocalLoss':
            fprint('Using FocalLoss')
            self.loss_fct1 = FocalLoss()
            self.loss_fct2 = nn.CrossEntropyLoss()
        elif self.config.get('loss_fn', 'CrossEntropyLoss') == 'CrossEntropyLoss':
            fprint('Using CrossEntropyLoss')
            self.loss_fct1 = nn.CrossEntropyLoss()
            self.loss_fct2 = nn.CrossEntropyLoss()
        elif self.config.get('loss_fn', 'CrossEntropyLoss') == 'ClassBalanceCrossEntropyLoss':
            fprint('Using ClassBalanceCrossEntropyLoss')
            self.loss_fct1 = ClassBalanceCrossEntropyLoss()
            self.loss_fct2 = nn.CrossEntropyLoss()
        elif self.config.get('loss_fn', 'CrossEntropyLoss') == 'LDMALoss':
            fprint('Using LDMALoss')
            self.loss_fct1 = LDAMLoss(list(range(2)), max_m=0.5, s=30)
            self.loss_fct2 = nn.CrossEntropyLoss()

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.encoder.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]
        return vec

    def get_bart_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.encoder.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_roberta_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        vec = self.encoder(input_ids=source_ids, attention_mask=attention_mask)[0][:, 0, :]
        return vec

    def forward(self, inputs):
        source_ids, labels, corrupt_labels = inputs

        if 't5' in self.config.pretrained_bert:
            vec = self.get_t5_vec(source_ids)
        elif 'bart' in self.config.pretrained_bert:
            vec = self.get_bart_vec(source_ids)
        else:
            vec = self.get_roberta_vec(source_ids)

        logits1 = self.classifier1(vec)
        logits2 = self.classifier2(vec)
        prob = nn.functional.softmax(logits1, -1)
        c_prob = nn.functional.softmax(logits2, -1)

        if labels is not None:
            loss_fct1 = nn.CrossEntropyLoss()
            loss_fct2 = nn.CrossEntropyLoss()

            # loss_fct1 = FocalLoss()
            # loss_fct2 = FocalLoss()

            loss = loss_fct1(logits1, labels) + loss_fct2(logits2, corrupt_labels)
            # loss = loss_fct1(logits1, labels)
            # loss = loss_fct1(logits2, corrupt_labels)
            return {'loss': loss, 'logits': prob, 'c_logits': c_prob}
        else:
            return {'logits': prob, 'c_logits': c_prob}
