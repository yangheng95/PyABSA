# -*- coding: utf-8 -*-
# file: fix_dataset.py
# time: 2021/6/5 0005
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

# convert polarity labels to {0, N-1}

from pyabsa import find_target_file
dataset_path = 'apc_datasets/MAMS'
train_datasets = find_target_file(dataset_path, 'train', exclude_key='infer', find_all=True)
test_datasets = find_target_file(dataset_path, 'test', exclude_key='infer', find_all=True)
for file in train_datasets + test_datasets:

        fin = open(file, 'r', newline='\n', encoding='utf-8')
        lines = fin.readlines()
        fin.close()
        path_to_save = file
        fout = open(path_to_save, 'w', encoding='utf-8', newline='\n', errors='ignore')

        for i in range(0, len(lines), 3):
            fout.write(lines[i])
            fout.write(lines[i+1])
            fout.write(str(int(lines[i + 2].strip())+1) + '\n')
        fout.close()
print('process finished')
