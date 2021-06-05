# -*- coding: utf-8 -*-
# file: convert_dataset_for_inferring.py
# author: yangheng<yangheng@m.scnu.edu.cn>
# Copyright (C) 2021. All Rights Reserved.

from pyabsa.pyabsa_utils import find_target_file


# convert atepc_datasets in this repo for inferring
def generate_inferring_set_for_apc(dataset_path):
    train_datasets = find_target_file(dataset_path, 'train', exclude_key='infer', find_all=True)
    test_datasets = find_target_file(dataset_path, 'test', exclude_key='infer', find_all=True)
    for file in train_datasets + test_datasets:
        try:
            fin = open(file, 'r', newline='\n', encoding='utf-8')
            lines = fin.readlines()
            fin.close()
            path_to_save = file + '.inference'
            fout = open(path_to_save, 'w', encoding='utf-8', newline='\n', errors='ignore')

            for i in range(0, len(lines), 3):
                sample = lines[i].strip().replace('$T$', '[ASP]{}[ASP]'.format(lines[i + 1].strip()))
                fout.write(sample + ' !sent! ' + lines[i + 2].strip() + '\n')
            fout.close()
        except:
            print('Unprocessed file:', file)
    print('process finished')
