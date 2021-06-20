# -*- coding: utf-8 -*-
# file: apc_benchmark.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa.research.benchmark.apc_benchmark.slide_lcfs_bert_benchmark import (run_slide_lcfs_bert_cdw,
                                                                               run_slide_lcfs_bert_cdm)
from pyabsa.research.benchmark.apc_benchmark.slide_lcf_bert_benchmark import (run_slide_lcf_bert_cdw,
                                                                              run_slide_lcf_bert_cdm)
from pyabsa.research.benchmark.apc_benchmark.lcf_bert_benchmark import run_lcf_bert_cdw, run_lcf_bert_cdm
from pyabsa.research.benchmark.apc_benchmark.lcfs_bert_benchmark import run_lcfs_bert_cdw, run_lcfs_bert_cdm
from pyabsa.research.benchmark.apc_benchmark.bert_spc_benchmark import run_bert_spc_cdw, run_bert_spc_cdm
from pyabsa.research.benchmark.apc_benchmark.bert_base_benchmark import run_bert_base_cdw, run_bert_base_cdm
