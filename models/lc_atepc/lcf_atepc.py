# -*- coding: utf-8 -*-
# file: lcf_atepc.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.


from pytorch_transformers.modeling_bert import BertForTokenClassification, BertPooler, BertSelfAttention

from torch.nn import Linear, CrossEntropyLoss
import torch
import torch.nn as nn
import copy
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_vec = np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_length))
        zero_tensor = torch.tensor(zero_vec).float().to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class LCF_ATEPC(BertForTokenClassification):

    def __init__(self, bert_base_model, args):
        super(LCF_ATEPC, self).__init__(config=bert_base_model.config)
        config = bert_base_model.config
        self.bert4global = bert_base_model
        self.args = args
        # do not init lcf layer if BERT-SPC or BERT-BASE specified
        # if not self.args.use_bert_spc:
        if not self.args.use_dual_bert:
            self.bert4local = copy.deepcopy(self.bert4global)
        else:
            self.bert4local = self.bert4global
        self.pooler = BertPooler(config)
        if args.dataset in {'camera', 'car', 'phone', 'notebook'}:
            self.dense = torch.nn.Linear(768, 2)
        else:
            self.dense = torch.nn.Linear(768, 3)
        self.bert_global_focus = self.bert4global
        self.dropout = nn.Dropout(self.args.dropout)
        self.SA1 = SelfAttention(config, args)
        self.SA2 = SelfAttention(config, args)
        self.linear_double = nn.Linear(768 * 2, 768)
        self.linear_triple = nn.Linear(768 * 3, 768)

    def get_ate_labels(self, labels):
        if labels is None:
            return
        # convert tags of BERT-SPC input to BERT-BASE format
        labels = labels.detach().cpu().numpy()
        for text_i in range(len(labels)):
            sep_index = np.argmax((labels[text_i] == 5))
            labels[text_i][sep_index + 1:] = 0
        return torch.tensor(labels).to(self.args.device)

    def get_apc_polarities(self, b_polarities):
        b_polarities = b_polarities.detach().cpu().numpy()
        shape = b_polarities.shape
        polarities = np.zeros((shape[0]))
        i = 0
        for polarity in b_polarities:
            polarity_idx = np.flatnonzero(polarity + 1)
            try:
                polarities[i] = polarity[polarity_idx[0]]
            except:
                pass
            i += 1
        polarities = torch.from_numpy(polarities).long().to(self.args.device)
        return polarities

    def get_cdw_vecs(self, text_local_indices, polarities):
        text_ids = text_local_indices.detach().cpu().numpy()
        asp_ids = polarities.detach().cpu().numpy()
        weighted_text_raw_indices = np.ones((text_local_indices.size(0), text_local_indices.size(1), 768),
                                            dtype=np.float32)
        SRD = self.args.SRD
        for text_i, asp_i in zip(range(len(text_ids)), range(len(asp_ids))):
            a_ids = np.flatnonzero(asp_ids[asp_i] + 1)
            text_len = np.flatnonzero(text_ids[text_i])[-1] + 1
            asp_len = len(a_ids)
            try:
                asp_begin = a_ids[0]
            except:
                asp_begin = 0
            asp_avg_index = (asp_begin * 2 + asp_len) / 2
            # a_ids[-1] + asp_len + 1 is the position of the last token_i [SEP]
            distances = np.zeros((text_len), dtype=np.float32)
            for i in range(len(distances)):
                if abs(i - asp_avg_index) + asp_len / 2 > SRD:
                    distances[i] = 1 - (abs(i - asp_avg_index) + asp_len / 2
                                        - SRD) / len(distances)
                else:
                    distances[i] = 1
            for i in range(len(distances)):
                weighted_text_raw_indices[text_i][i] = weighted_text_raw_indices[text_i][i] * distances[i]
        weighted_text_raw_indices = torch.from_numpy(weighted_text_raw_indices)
        return weighted_text_raw_indices.to(self.args.device)

    def get_cdm_vecs(self, text_local_indices, polarities):
        text_ids = text_local_indices.detach().cpu().numpy()
        asp_ids = polarities.detach().cpu().numpy()
        SRD = self.args.SRD
        masked_text_raw_indices = np.ones((text_local_indices.size(0), text_local_indices.size(1), 768),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(text_ids)), range(len(asp_ids))):
            a_ids = np.flatnonzero(asp_ids[asp_i] + 1)
            try:
                asp_begin = a_ids[0]
            except:
                asp_begin = 0
            asp_len = len(a_ids)
            if asp_begin >= SRD:
                mask_begin = asp_begin - SRD
            else:
                mask_begin = 0
            for i in range(mask_begin):
                masked_text_raw_indices[text_i][i] = np.zeros((768), dtype=np.float)
            for j in range(asp_begin + asp_len + SRD - 1, self.args.max_seq_length):
                masked_text_raw_indices[text_i][j] = np.zeros((768), dtype=np.float)
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.args.device)

    def get_bert_base_ids(self, text_indices):
        # convert BERT-SPC input to BERT-BASE format
        text_ids = text_indices.detach().cpu().numpy()
        for text_i in range(len(text_ids)):
            sep_index = np.argmax((text_ids[text_i] == 102))
            text_ids[text_i][sep_index + 1:] = 0
        return torch.tensor(text_ids).to(self.args.device)

    def forward(self, bert_spc_ids, token_type_ids=None, attention_mask=None, ate_labels=None, polarities=None,
                valid_ids=None, attention_mask_label=None):
        if not self.args.use_bert_spc:
            bert_spc_ids = self.get_bert_base_ids(bert_spc_ids)
            ate_labels = self.get_ate_labels(ate_labels)
        global_context_out, _ = self.bert4global(bert_spc_ids, token_type_ids, attention_mask)
        apc_polarities = self.get_apc_polarities(polarities)

        batch_size, max_len, feat_dim = global_context_out.shape
        global_valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).to(self.args.device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    global_valid_output[i][jj] = global_context_out[i][j]
        global_context_out = self.dropout(global_valid_output)
        ate_logits = self.classifier(global_context_out)

        if self.args.lcf is not None:

            if self.args.use_bert_spc:
                local_context_ids = self.get_bert_base_ids(bert_spc_ids)
            else:
                local_context_ids = bert_spc_ids

            local_context_out, _ = self.bert4local(bert_spc_ids)
            batch_size, max_len, feat_dim = local_context_out.shape
            local_valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).to(self.args.device)
            # for i in range(batch_size):
            #     jj = -1
            #     for j in range(max_len):
            #         if valid_ids[i][j].item() == 1:
            #             jj += 1
            #             local_valid_output[i][jj] = local_context_out[i][j]
            local_context_out = self.dropout(local_valid_output)

            if 'cdm' in self.args.lcf:
                cdm_vec = self.get_cdm_vecs(local_context_ids, polarities)
                cdm_context_out = torch.mul(local_context_out, cdm_vec)
                cdm_context_out = self.SA1(cdm_context_out)
                cat_out = torch.cat((global_context_out, cdm_context_out), dim=-1)
                cat_out = self.linear_double(cat_out)
            elif 'cdw' in self.args.lcf:
                cdw_vec = self.get_cdw_vecs(local_context_ids, polarities)
                cdw_context_out = torch.mul(local_context_out, cdw_vec)
                cdw_context_out = self.SA1(cdw_context_out)
                cat_out = torch.cat((global_context_out, cdw_context_out), dim=-1)
                cat_out = self.linear_double(cat_out)
            elif 'fusion' in self.args.lcf:
                cdm_vec = self.get_cdm_vecs(local_context_ids, polarities)
                cdm_context_out = torch.mul(local_context_out, cdm_vec)
                cdw_vec = self.get_cdw_vecs(local_context_ids, polarities)
                cdw_context_out = torch.mul(local_context_out, cdw_vec)
                cat_out = torch.cat((global_context_out, cdw_context_out, cdm_context_out), dim=-1)
                cat_out = self.linear_triple(cat_out)
            # sa_out = self.SA2(cat_out)
            pooled_out = self.pooler(cat_out)
        else:
            pooled_out = self.pooler(global_context_out)
        pooled_out = self.dropout(pooled_out)
        apc_logits = self.dense(pooled_out)

        if ate_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            loss_sen = CrossEntropyLoss()
            loss_ate = loss_fct(ate_logits.view(-1, self.num_labels), ate_labels.view(-1))
            loss_apc = loss_sen(apc_logits, apc_polarities)

            return loss_ate, loss_apc
        else:
            return ate_logits, apc_logits
