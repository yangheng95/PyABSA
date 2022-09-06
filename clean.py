# -*- coding: utf-8 -*-
# file: clean.py
# time: 2022/7/3
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
import os

from findfile import rm_files, rm_dirs, rm_cwd_files, rm_cwd_dirs

# rm_cwd_dirs(or_key=['__pycache__'])
# rm_cwd_files(or_key=['pyabsa.egg-info'])
# rm_cwd_dirs(or_key=['pyabsa.egg-info'])
# rm_cwd_dirs(or_key=['dist'])
#
# # rm_dirs(os.getcwd(), or_key=['integrated_datasets'])
# # rm_dirs(os.getcwd(), or_key=['checkpoints'])
# rm_dirs(r'C:\Users\chuan\OneDrive - University of Exeter\Works', or_key=['integrated_datasets', 'checkpoints', 'logs'])
# rm_files(r'C:\Users\chuan\OneDrive - University of Exeter\Works', or_key=['.pyc'])
# rm_dirs(r'/home/yangheng/', or_key=['integrated_datasets', 'checkpoints', 'logs'])
