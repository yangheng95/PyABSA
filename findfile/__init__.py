# -*- coding: utf-8 -*-
# file: __init__.py.py
# time: 2021/8/4
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
from update_checker import UpdateChecker

from findfile.find import (find_files,
                           find_file,
                           find_dirs,
                           find_dir,
                           find_cwd_dir,
                           find_cwd_file,
                           find_cwd_dirs,
                           find_cwd_files,
                           rm_dirs,
                           rm_files,
                           rm_dir,
                           rm_file,
                           rm_cwd_files,
                           rm_cwd_dirs,
                           )

__name__ = 'findfile'
__version__ = '1.7.9.7'

checker = UpdateChecker()
check_result = checker.check(__name__, __version__)

if check_result:
    print(check_result)
