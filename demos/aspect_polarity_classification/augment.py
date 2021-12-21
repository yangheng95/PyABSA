# -*- coding: utf-8 -*-
# file: augment.py
# time: 2021/12/20
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import tqdm

from pyabsa import ABSADatasetList
from pyabsa.functional.dataset import detect_dataset

# import the CheckListAugmenter
from textattack.augmentation import CheckListAugmenter

# Alter default values if desired
augmenter = CheckListAugmenter(pct_words_to_swap=0.3, transformations_per_example=10)

datasets = detect_dataset(ABSADatasetList.MAMS, 'apc')
train_sets = datasets['train']
for train_set in train_sets:
    print('processing {}'.format(train_set))
    augmented_train_set = []
    fin = open(train_set, encoding='utf8', mode='r', newline='\r\n')
    lines = fin.readlines()
    fin.close()
    for i in tqdm.tqdm(range(0, len(lines), 3)):
        try:
            lines[i] = lines[i].strip()
            lines[i + 1] = lines[i + 1].strip()
            lines[i + 2] = lines[i + 2].strip()
            augmented_train_set.extend([lines[i], lines[i + 1], lines[i + 2]])
            # augmentations = augmenter.augment(lines[i].replace('$T$', lines[i + 1]))
            augmentations = augmenter.augment(lines[i])
            for text in augmentations:
                if '$T$' in text:
                    augmented_train_set.extend([text, lines[i + 1], lines[i + 2]])
                else:
                    print(text)
        except:
            print(lines[i])

    fout = open(train_set + '.augment', encoding='utf8', mode='w')
    for line in augmented_train_set:
        fout.write(line + '\n')
    fout.close()
