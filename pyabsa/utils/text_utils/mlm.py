# -*- coding: utf-8 -*-
# file: mlm.py
# time: 03/11/2022 15:37
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
from transformers import (
    BertForMaskedLM,
    RobertaForMaskedLM,
    DebertaV2ForMaskedLM,
    AutoConfig,
)
from transformers import AutoTokenizer


def get_mlm_and_tokenizer(model, config):
    base_model = None
    for child in model.children():
        if hasattr(child, "base_model"):
            base_model = child.base_model
            break

    pretrained_config = AutoConfig.from_pretrained(config.pretrained_bert)
    if "deberta-v3" in config.pretrained_bert:
        MLM = DebertaV2ForMaskedLM(pretrained_config)
        MLM.deberta = base_model
    elif "roberta" in config.pretrained_bert:
        MLM = RobertaForMaskedLM(pretrained_config)
        MLM.roberta = base_model
    else:
        MLM = BertForMaskedLM(pretrained_config)
        MLM.bert = base_model
    return MLM, AutoTokenizer.from_pretrained(config.pretrained_bert)
