# -*- coding: utf-8 -*-
# file: atepc_benchmark.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


from pyabsa.research.atepc.lcf_atepc_benchmark import run_lcf_atepc_cdw, run_lcf_atepc_cdm


def run_benchmark_for_atepc_models():
    run_lcf_atepc_cdw()
    run_lcf_atepc_cdm()
