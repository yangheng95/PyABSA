# -*- coding: utf-8 -*-
# file: FAST_LSA_S_V2.py
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Copyright (C) 2021. All Rights Reserved.
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.networks.lsa import LSA
from pyabsa.networks.sa_encoder import Encoder
from pyabsa.utils.pyabsa_utils import fprint


class FAST_LSA_S_V2(nn.Module):
    inputs = [
        "text_indices",
        "spc_mask_vec",
        "lcfs_cdw_vec",
        "left_lcfs_cdw_vec",
        "right_lcfs_cdw_vec",
        "lcfs_cdm_vec",
        "left_lcfs_cdm_vec",
        "right_lcfs_cdm_vec",
    ]

    def __init__(self, bert, config):
        super(FAST_LSA_S_V2, self).__init__()
        self.bert4global = bert
        self.config = config
        self.dropout = nn.Dropout(config.dropout)

        self.post_encoder = Encoder(bert.config, config)
        self.post_encoder_ = Encoder(bert.config, config)
        self.bert_pooler = BertPooler(bert.config)

        if self.config.lcf == "cdw":
            self.CDW_LSA = LSA(self.bert4global, self.config)
        if self.config.lcf == "cdm":
            self.CDM_LSA = LSA(self.bert4global, self.config)
        elif self.config.lcf == "fusion":
            self.CDW_LSA = LSA(self.bert4global, self.config)
            self.CDM_LSA = LSA(self.bert4global, self.config)
            self.fusion_linear = nn.Linear(config.embed_dim * 3, config.embed_dim)

        self.post_linear = nn.Linear(config.embed_dim * 2, config.embed_dim)
        self.dense = nn.Linear(config.embed_dim, config.output_dim)

    def forward(self, inputs):
        text_indices = inputs["text_indices"]
        spc_mask_vec = inputs["spc_mask_vec"].unsqueeze(2)
        lcfs_cdw_matrix = inputs["lcfs_cdw_vec"].unsqueeze(2)
        left_lcfs_cdw_matrix = inputs["left_lcfs_cdw_vec"].unsqueeze(2)
        right_lcfs_cdw_matrix = inputs["right_lcfs_cdw_vec"].unsqueeze(2)
        lcfs_cdm_matrix = inputs["lcfs_cdm_vec"].unsqueeze(2)
        left_lcfs_cdm_matrix = inputs["left_lcfs_cdm_vec"].unsqueeze(2)
        right_lcfs_cdm_matrix = inputs["right_lcfs_cdm_vec"].unsqueeze(2)

        global_context_features = self.bert4global(text_indices)["last_hidden_state"]

        if self.config.lcf == "cdw":
            sent_out = self.CDW_LSA(
                global_context_features,
                spc_mask_vec=spc_mask_vec,
                lcf_matrix=lcfs_cdw_matrix,
                left_lcf_matrix=left_lcfs_cdw_matrix,
                right_lcf_matrix=right_lcfs_cdw_matrix,
            )
            sent_out = torch.cat((global_context_features, sent_out), -1)
            sent_out = self.post_linear(sent_out)

        elif self.config.lcf == "cdm":
            sent_out = self.CDM_LSA(
                global_context_features,
                spc_mask_vec=spc_mask_vec,
                lcf_matrix=lcfs_cdm_matrix,
                left_lcf_matrix=left_lcfs_cdm_matrix,
                right_lcf_matrix=right_lcfs_cdm_matrix,
            )
            sent_out = torch.cat((global_context_features, sent_out), -1)
            sent_out = self.post_linear(sent_out)

        elif self.config.lcf == "fusion":
            cdw_sent_out = self.CDW_LSA(
                global_context_features,
                spc_mask_vec=spc_mask_vec,
                lcf_matrix=lcfs_cdw_matrix,
                left_lcf_matrix=left_lcfs_cdw_matrix,
                right_lcf_matrix=right_lcfs_cdw_matrix,
            )
            cdm_sent_out = self.CDM_LSA(
                global_context_features,
                spc_mask_vec=spc_mask_vec,
                lcfs_matrix=lcfs_cdm_matrix,
                left_lcf_matrix=left_lcfs_cdm_matrix,
                right_lcf_matrix=right_lcfs_cdm_matrix,
            )
            sent_out = self.fusion_linear(
                torch.cat((global_context_features, cdw_sent_out, cdm_sent_out), -1)
            )

        else:
            fprint("Invalid LCF mode: {}".format(self.config.lcf))
            sent_out = global_context_features

        sent_out = self.dropout(sent_out)
        sent_out = self.post_encoder_(sent_out)
        sent_out = self.bert_pooler(sent_out)
        dense_out = self.dense(sent_out)

        return {"logits": dense_out, "hidden_state": sent_out}
