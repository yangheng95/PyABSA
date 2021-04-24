# -*- coding: utf-8 -*-
# file: convert_dataset_for_inferring.py
# author: yangheng<yangheng@m.scnu.edu.cn>
# Copyright (C) 2021. All Rights Reserved.

import os


# convert datasets in this repo for inferring
def convert_dataset_for_inference(dataset_path):
    if os.path.isdir(dataset_path + '/'):
        files = os.listdir(dataset_path)
    else:
        files = [dataset_path]
    for file in files:
        if 'train' in file or 'test' in file:
            try:
                fin = open(dataset_path + '/' + file, 'r', newline='\n', encoding='utf-8')
                lines = fin.readlines()
                fin.close()
                path_to_save = dataset_path + '/' + file + '.inferring.dat'
                fout = open(path_to_save, 'w', encoding='utf-8', newline='\n', errors='ignore')

                for i in range(0, len(lines), 3):
                    sample = lines[i].strip().replace('$T$', '[ASP]{}[ASP]$'.format(lines[i + 1].strip()))
                    fout.write(sample + ' !sent! ' + lines[i + 2])
                fout.close()
            except:
                print('Unprocessed file:', file)
