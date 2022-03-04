# -*- coding: utf-8 -*-
# file: __init__.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

class ATEPCModelList(list):
    from pyabsa.core.atepc.models.bert_base_atepc import BERT_BASE_ATEPC
    from pyabsa.core.atepc.models.fast_lcf_atepc import FAST_LCF_ATEPC
    from pyabsa.core.atepc.models.fast_lcfs_atepc import FAST_LCFS_ATEPC
    from pyabsa.core.atepc.models.lcf_atepc import LCF_ATEPC
    from pyabsa.core.atepc.models.lcf_atepc_large import LCF_ATEPC_LARGE
    from pyabsa.core.atepc.models.lcf_template_atepc import LCF_TEMPLATE_ATEPC
    from pyabsa.core.atepc.models.lcfs_atepc import LCFS_ATEPC
    from pyabsa.core.atepc.models.lcfs_atepc_large import LCFS_ATEPC_LARGE

    BERT_BASE_ATEPC = BERT_BASE_ATEPC
    FAST_LCF_ATEPC = FAST_LCF_ATEPC
    FAST_LCFS_ATEPC = FAST_LCFS_ATEPC
    LCF_ATEPC = LCF_ATEPC
    LCF_ATEPC_LARGE = LCF_ATEPC_LARGE
    LCFS_ATEPC = LCFS_ATEPC
    LCFS_ATEPC_LARGE = LCFS_ATEPC_LARGE

    LCF_TEMPLATE_ATEPC = LCF_TEMPLATE_ATEPC

    def __init__(self):
        model_list = [self.BERT_BASE_ATEPC,
                      self.FAST_LCF_ATEPC,
                      self.FAST_LCFS_ATEPC,
                      self.LCF_ATEPC,
                      self.LCF_ATEPC_LARGE,
                      self.LCFS_ATEPC,
                      self.LCFS_ATEPC_LARGE]
        super().__init__(model_list)
