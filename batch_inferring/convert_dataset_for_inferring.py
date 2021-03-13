# -*- coding: utf-8 -*-
# file: convert_dataset_for_inferring.py
# author: yangheng<yangheng@m.scnu.edu.cn>
# Copyright (C) 2021. All Rights Reserved.


dataset_files = {
    'twitter': {
        'train': '../datasets/acl-14-short-data/train.raw',
        'test': '../datasets/acl-14-short-data/test.raw'
    },
    'rest14': {
        'train': '../datasets/semeval14/Restaurants_Train.xml.seg',
        'test': '../datasets/semeval14/Restaurants_Test_Gold.xml.seg'
    },
    'rest15': {
        'train': '../datasets/semeval15/restaurant_train.raw',
        'test': '../datasets/semeval15/restaurant_test.raw'
    },
    'rest16': {
        'train': '../datasets/semeval16/restaurant_train.raw',
        'test': '../datasets/semeval17/restaurant_test.raw'
    },
    'laptop': {
        'train': '../datasets/semeval14/Laptops_Train.xml.seg',
        'test': '../datasets/semeval14/Laptops_Test_Gold.xml.seg'
    },
    'car': {
        'train': '../datasets/Chinese/car/car.train.txt',
        'test': '../datasets/Chinese/car/car.test.txt'
    },
    'phone': {
        'train': '../datasets/Chinese/camera/camera.train.txt',
        'test': '../datasets/Chinese/camera/camera.test.txt'
    },
    'notebook': {
        'train': '../datasets/Chinese/notebook/notebook.train.txt',
        'test': '../datasets/Chinese/notebook/notebook.test.txt'
    },
    'camera': {
        'train': '../datasets/Chinese/phone/phone.train.txt',
        'test': '../datasets/Chinese/phone/phone.test.txt'
    },
    'multilingual': {
        'train': '../datasets/multilingual/multilingual_train.raw',
        'test': '../datasets/multilingual/multilingual_test.raw'
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
        sample = lines[i].strip().replace('$T$', '${}$'.format(lines[i + 1].strip()))
        fout.write(sample + ' !sent! '+lines[i + 2])

    fout.close()


if __name__ == '__main__':
    convert('rest14', 'test')
