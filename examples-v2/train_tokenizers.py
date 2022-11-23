# -*- coding: utf-8 -*-
# file: train_tokenizers.py
# time: 2022/11/19 15:30
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import findfile
from transformers import AutoTokenizer

from pyabsa.utils import train_word2vec, train_bpe_tokenizer

if __name__ == '__main__':
    # train word2vec
    paths = findfile.find_cwd_files('.txt')
    pre_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    # train_word2vec(paths, save_path='word2vec', pre_tokenizer=pre_tokenizer)

    # train bpe tokenizer
    train_bpe_tokenizer(paths, save_path='bpe_tokenizer', base_tokenizer='roberta-base')
