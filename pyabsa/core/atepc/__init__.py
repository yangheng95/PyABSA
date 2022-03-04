# -*- coding: utf-8 -*-
# file: __init__.py
# time: 2021/5/21 0021
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa.core.atepc.models import (lcfs_atepc,
                                      lcfs_atepc_large,
                                      lcf_atepc,
                                      fast_lcfs_atepc,
                                      lcf_template_atepc,
                                      lcf_atepc_large,
                                      fast_lcf_atepc,
                                      bert_base_atepc)


class ATEPCModelList:
    BERT_BASE_ATEPC = bert_base_atepc.BERT_BASE_ATEPC

    LCF_ATEPC = lcf_atepc.LCF_ATEPC
    LCF_ATEPC_LARGE = lcf_atepc_large.LCF_ATEPC_LARGE
    FAST_LCF_ATEPC = fast_lcf_atepc.FAST_LCF_ATEPC

    LCFS_ATEPC = lcfs_atepc.LCFS_ATEPC
    LCFS_ATEPC_LARGE = lcfs_atepc_large.LCFS_ATEPC_LARGE
    FAST_LCFS_ATEPC = fast_lcfs_atepc.FAST_LCFS_ATEPC

    LCF_TEMPLATE_ATEPC = lcf_template_atepc.LCF_TEMPLATE_ATEPC
