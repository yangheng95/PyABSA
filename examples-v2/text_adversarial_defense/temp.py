# -*- coding: utf-8 -*-
# file: temp.py
# time: 19:01 2023/2/5
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
import findfile

for f in findfile.find_cwd_files(['integrated_datasets'], exclude_key=['apc', 'atepc']):
    with open(f, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    with open(f, 'w', encoding='utf-8') as fout:
        for line in lines:
            fout.write(line.replace('!ref!', '$LABEL$'))
