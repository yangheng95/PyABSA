# -*- coding: utf-8 -*-
# file: convert_dataset_for_inferring.py
# author: yangheng<yangheng@m.scnu.edu.cn>
# Copyright (C) 2021. All Rights Reserved.


dataset_files = {
    'twitter': {
        'main': 'acl-14-short-data/main.raw',
        'test': 'acl-14-short-data/test.raw'
    },
    'rest14': {
        'main': 'semeval14/Restaurants_Train.xml.seg',
        'test': 'semeval14/Restaurants_Test_Gold.xml.seg'
    },
    'rest15': {
        'main': 'semeval15/restaurant_train.raw',
        'test': 'semeval15/restaurant_test.raw'
    },
    'rest16': {
        'main': 'semeval16/restaurant_train.raw',
        'test': 'semeval16/restaurant_test.raw'
    },
    'laptop': {
        'main': 'semeval14/Laptops_Train.xml.seg',
        'test': 'semeval14/Laptops_Test_Gold.xml.seg'
    },
    'car': {
        'main': 'Chinese/car/car.main.txt',
        'test': 'Chinese/car/car.test.txt'
    },
    'phone': {
        'main': 'Chinese/camera/camera.main.txt',
        'test': 'Chinese/camera/camera.test.txt'
    },
    'notebook': {
        'main': 'Chinese/notebook/notebook.main.txt',
        'test': 'Chinese/notebook/notebook.test.txt'
    },
    'camera': {
        'main': 'Chinese/phone/phone.main.txt',
        'test': 'Chinese/phone/phone.test.txt'
    },
    'multilingual': {
        'main': 'multilingual/multilingual_train.raw',
        'test': 'multilingual/multilingual_test.raw'
    }
}


# convert datasets in this repo for inferring
def convert(dataset_name='multilingual', dataset_type='test'):
    fin = open(dataset_files[dataset_name][dataset_type], 'r', newline='\n', encoding='utf-8')
    lines = fin.readlines()
    fin.close()

    path_to_save = dataset_name + '_' + dataset_type + '_inferring.dat'
    fout = open(path_to_save, 'w', encoding='utf-8', newline='\n', errors='ignore')

    for i in range(0, len(lines), 3):
        sample = lines[i].strip().replace('$T$', '[ASP]{}[ASP]$'.format(lines[i + 1].strip()))
        fout.write(sample + ' !sent! ' + lines[i + 2])

    fout.close()


if __name__ == '__main__':
    convert('rest14', 'test')
    convert('rest15', 'test')
    convert('rest16', 'test')
    convert('laptop', 'test')
    convert('mams', 'test')
    convert('rest14', 'main')
    convert('rest15', 'main')
    convert('rest16', 'main')
    convert('laptop', 'main')
    convert('mams', 'main')
