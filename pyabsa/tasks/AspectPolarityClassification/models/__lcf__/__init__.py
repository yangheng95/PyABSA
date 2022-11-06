# -*- coding: utf-8 -*-
# file: __init__.py
# time: 02/11/2022 20:07
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.


class LCFAPCModelList(list):
    from .bert_base import BERT_MLP
    from .bert_spc import BERT_SPC
    from .bert_spc_v2 import BERT_SPC_V2
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
    from .fast_lsa_t_v2 import FAST_LSA_T_V2
    from .fast_lsa_t import FAST_LSA_T
    from .fast_lsa_s_v2 import FAST_LSA_S_V2
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

    FAST_LSA_T_V2 = FAST_LSA_T_V2
    FAST_LSA_S_V2 = FAST_LSA_S_V2

    DLCF_DCA_BERT = DLCF_DCA_BERT
    DLCFS_DCA_BERT = DLCFS_DCA_BERT

    LCF_BERT = LCF_BERT
    FAST_LCF_BERT = FAST_LCF_BERT
    LCF_DUAL_BERT = LCF_DUAL_BERT

    LCFS_BERT = LCFS_BERT
    FAST_LCFS_BERT = FAST_LCFS_BERT
    LCFS_DUAL_BERT = LCFS_DUAL_BERT

    LCA_BERT = LCA_BERT

    BERT_MLP = BERT_MLP
    BERT_SPC = BERT_SPC
    BERT_SPC_V2 = BERT_SPC_V2

    FAST_LCF_BERT_ATT = FAST_LCF_BERT_ATT

    LCF_TEMPLATE_BERT = LCF_TEMPLATE_BERT

    def __init__(self):
        model_list = [
            self.SLIDE_LCF_BERT,
            self.SLIDE_LCFS_BERT,
            self.LSA_T,
            self.LSA_S,
            self.FAST_LSA_T,
            self.FAST_LSA_S,
            self.FAST_LSA_T_V2,
            self.FAST_LSA_S,
            self.FAST_LSA_S_V2,
            self.DLCF_DCA_BERT,
            self.DLCFS_DCA_BERT,
            self.LCF_BERT,
            self.FAST_LCF_BERT,
            self.LCF_DUAL_BERT,
            self.LCFS_BERT,
            self.FAST_LCFS_BERT,
            self.LCFS_DUAL_BERT,
            self.LCA_BERT,
            self.BERT_MLP,
            self.BERT_SPC,
            self.FAST_LCF_BERT_ATT,
        ]
        super().__init__(model_list)


class APCModelList(LCFAPCModelList):
    pass
