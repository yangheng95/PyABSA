# -*- coding: utf-8 -*-
# file: __init__.py.py
# time: 03/02/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

__version__ = '0.4.26'
__name__ = 'metric_visualizer'

from update_checker import UpdateChecker

from .metric_visualizer import MetricVisualizer

checker = UpdateChecker()
check_result = checker.check(__name__, __version__)

if check_result:
    print(check_result)
