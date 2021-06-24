# -*- coding: utf-8 -*-
# file: atepc_benchmark.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa.research.benchmark.atepc_benchmark.bert_base_atepc_benchmark import (run_bert_base_atepc_cdw,
                                                                                 run_bert_base_atepc_cdm)

from pyabsa.research.benchmark.atepc_benchmark.lcf_atepc_benchmark import (run_lcf_atepc_cdw,
                                                                           run_lcf_atepc_cdm)

from pyabsa.research.benchmark.atepc_benchmark.lcfs_atepc_benchmark import (run_lcfs_atepc_cdw,
                                                                            run_lcfs_atepc_cdm)

from pyabsa.research.benchmark.atepc_benchmark.fast_lcf_atepc_benchmark import (run_fast_lcf_atepc_cdw,
                                                                                run_fast_lcf_atepc_cdm)

from pyabsa.research.benchmark.atepc_benchmark.fast_lcfs_atepc_benchmark import (run_fast_lcfs_atepc_cdw,
                                                                                 run_fast_lcfs_atepc_cdm)

from pyabsa.research.benchmark.atepc_benchmark.lcf_atepc_large_benchmark import (run_lcf_atepc_large_cdw,
                                                                                 run_lcf_atepc_large_cdm)

from pyabsa.research.benchmark.atepc_benchmark.lcfs_atepc_large_benchmark import (run_lcfs_atepc_large_cdw,
                                                                                  run_lcfs_atepc_large_cdm)


def run_benchmark_for_all_atepc_models():
    run_bert_base_atepc_cdw()
    run_bert_base_atepc_cdm()
    run_lcf_atepc_cdw()
    run_lcf_atepc_cdm()
    run_lcfs_atepc_cdw()
    run_lcfs_atepc_cdm()
    run_fast_lcf_atepc_cdw()
    run_fast_lcf_atepc_cdm()
    run_fast_lcfs_atepc_cdw()
    run_fast_lcfs_atepc_cdm()
    run_lcf_atepc_large_cdw()
    run_lcf_atepc_large_cdm()
    run_lcfs_atepc_large_cdw()
    run_lcfs_atepc_large_cdm()
