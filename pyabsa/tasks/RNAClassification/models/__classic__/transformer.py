# -*- coding: utf-8 -*-
# file: transformer.py
# time: 01/11/2022 12:58
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
from torch import nn


class Transformer(nn.Module):
    def __init__(self, embedding_matrix, config):
        super(Transformer, self).__init__()
        self.config = self.config
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.dropout = nn.Dropout(self.config.dropout)
        self.transformer = nn.Transformer(d_model=self.config.hidden_dim,
                                          # nhead=self.config.num_attention_heads,
                                          # num_encoder_layers=self.config.num_hidden_layers,
                                          # num_decoder_layers=self.config.num_hidden_layers,
                                          # dim_feedforward=self.config.intermediate_size,
                                          dropout=self.config.dropout,
                                          activation=self.config.hidden_act,
                                          custom_encoder=None,
                                          custom_decoder=None)
        self.classifier = nn.Linear(self.config.hidden_dim, self.config.output_dim)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None):
        transformer_outputs = self.transformer(input_ids=input_ids,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask,
                                               inputs_embeds=inputs_embeds)
        sequence_output = transformer_outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        logits = logits.squeeze(-1)
        return logits
