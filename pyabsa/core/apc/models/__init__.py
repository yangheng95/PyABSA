# -*- coding: utf-8 -*-
# file: __init__.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2021. All Rights Reserved.

import pyabsa.core.apc.classic.__glove__.models
import pyabsa.core.apc.classic.__bert__.models


class APCModelList:
    from .bert_base import BERT_BASE
    from .bert_spc import BERT_SPC
    from .dlcf_dca_bert import DLCF_DCA_BERT
    from .dlcfs_dca_bert import DLCFS_DCA_BERT
    from .fast_lcf_bert import FAST_LCF_BERT
    from .fast_lcf_bert_att import FAST_LCF_BERT_ATT
    from .fast_lcfs_bert import FAST_LCFS_BERT
    from .lca_bert import LCA_BERT
    from .lcf_bert import LCF_BERT
    from .lcf_dual_bert import LCF_DUAL_BERT
    from .lcf_template_apc import LCF_TEMPLATE_BERT
    from .lcfs_bert import LCFS_BERT
    from .lcfs_dual_bert import LCFS_DUAL_BERT
    from .fast_lsa_t import FAST_LSA_T
    from .fast_lsa_s import FAST_LSA_S
    from .lsa_t import LSA_T
    from .lsa_s import LSA_S
    from .ssw_s import SSW_S
    from .ssw_t import SSW_T

    SLIDE_LCF_BERT = FAST_LSA_T
    SLIDE_LCFS_BERT = FAST_LSA_S
    LSA_T = LSA_T
    LSA_S = LSA_S
    FAST_LSA_T = FAST_LSA_T
    FAST_LSA_S = FAST_LSA_S

    DLCF_DCA_BERT = DLCF_DCA_BERT
    DLCFS_DCA_BERT = DLCFS_DCA_BERT

    LCF_BERT = LCF_BERT
    FAST_LCF_BERT = FAST_LCF_BERT
    LCF_DUAL_BERT = LCF_DUAL_BERT

    LCFS_BERT = LCFS_BERT
    FAST_LCFS_BERT = FAST_LCFS_BERT
    LCFS_DUAL_BERT = LCFS_DUAL_BERT

    LCA_BERT = LCA_BERT

    BERT_BASE = BERT_BASE
    BERT_SPC = BERT_SPC

    FAST_LCF_BERT_ATT = FAST_LCF_BERT_ATT

    SSW_S = SSW_S
    SSW_T = SSW_T

    LCF_TEMPLATE_BERT = LCF_TEMPLATE_BERT


class GloVeAPCModelList:
    LSTM = pyabsa.core.apc.classic.__glove__.models.LSTM
    IAN = pyabsa.core.apc.classic.__glove__.models.IAN
    MemNet = pyabsa.core.apc.classic.__glove__.models.MemNet
    RAM = pyabsa.core.apc.classic.__glove__.models.RAM
    TD_LSTM = pyabsa.core.apc.classic.__glove__.models.TD_LSTM
    TC_LSTM = pyabsa.core.apc.classic.__glove__.models.TC_LSTM
    Cabasc = pyabsa.core.apc.classic.__glove__.models.Cabasc
    ATAE_LSTM = pyabsa.core.apc.classic.__glove__.models.ATAE_LSTM
    TNet_LF = pyabsa.core.apc.classic.__glove__.models.TNet_LF
    AOA = pyabsa.core.apc.classic.__glove__.models.AOA
    MGAN = pyabsa.core.apc.classic.__glove__.models.MGAN
    ASGCN = pyabsa.core.apc.classic.__glove__.models.ASGCN


class BERTBaselineAPCModelList:
    LSTM_BERT = pyabsa.core.apc.classic.__bert__.models.LSTM_BERT
    IAN_BERT = pyabsa.core.apc.classic.__bert__.models.IAN_BERT
    MemNet_BERT = pyabsa.core.apc.classic.__bert__.models.MemNet_BERT
    RAM_BERT = pyabsa.core.apc.classic.__bert__.models.RAM_BERT
    TD_LSTM_BERT = pyabsa.core.apc.classic.__bert__.models.TD_LSTM_BERT
    TC_LSTM_BERT = pyabsa.core.apc.classic.__bert__.models.TC_LSTM_BERT
    Cabasc_BERT = pyabsa.core.apc.classic.__bert__.models.Cabasc_BERT
    ATAE_LSTM_BERT = pyabsa.core.apc.classic.__bert__.models.ATAE_LSTM_BERT
    TNet_LF_BERT = pyabsa.core.apc.classic.__bert__.models.TNet_LF_BERT
    AOA_BERT = pyabsa.core.apc.classic.__bert__.models.AOA_BERT
    MGAN_BERT = pyabsa.core.apc.classic.__bert__.models.MGAN_BERT
    ASGCN_BERT = pyabsa.core.apc.classic.__bert__.models.ASGCN_BERT
