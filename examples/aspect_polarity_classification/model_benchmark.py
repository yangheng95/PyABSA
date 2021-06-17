# -*- coding: utf-8 -*-
# file: model_benchmark.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa.research.benchmark.apc_benchmark import run_slide_lcf_bert_cdw, run_slide_lcf_bert_cdm

from pyabsa.research.benchmark.atepc_benchmark import run_benchmark_for_atepc_models

run_slide_lcf_bert_cdw()
run_slide_lcf_bert_cdm()

from pyabsa.research.benchmark.apc_benchmark import run_slide_lcfs_bert_cdw, run_slide_lcfs_bert_cdm

run_slide_lcfs_bert_cdw()
run_slide_lcfs_bert_cdm()
