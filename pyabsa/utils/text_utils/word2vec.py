# -*- coding: utf-8 -*-
# file: word2vec.py
# time: 02/11/2022 15:41
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import os
import time

import findfile
from findfile import find_cwd_files
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from transformers import AutoTokenizer


# from gensim.models.word2vec import LineSentence

def train_word2vec(corpus_files: list = None,
                   save_path='word2vec',
                   vector_dim=300,
                   window=5,
                   min_count=1000,
                   skip_gram=1,
                   num_workers=None,
                   epochs=10,
                   pre_tokenizer=None,
                   **kwargs):
    '''
    LineSentence(inp)：格式简单：一句话=一行; 单词已经过预处理并被空格分隔。
    size：是每个词的向量维度；
    window：是词向量训练时的上下文扫描窗口大小，窗口为5就是考虑前5个词和后5个词；
    min-count：设置最低频率，默认是5，如果一个词语在文档中出现的次数小于5，那么就会丢弃；
    workers：是训练的进程数（需要更精准的解释，请指正），默认是当前运行机器的处理器核数。这些参数先记住就可以了。
    sg ({0, 1}, optional) – 模型的训练算法: 1: skip-gram; 0: CBOW
    alpha (float, optional) – 初始学习率
    iter (int, optional) – 迭代次数，默认为5
    '''

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    in_corpus = []
    if not corpus_files:
        corpus_files = find_cwd_files('.txt', exclude_key=['word2vec', 'ignore'])
    elif isinstance(corpus_files, str):
        corpus_files = [corpus_files]
    else:
        assert isinstance(corpus_files, list)
    print('Start loading corpus files:', corpus_files)
    for f in corpus_files:
        with open(f, 'r', encoding='utf-8') as fin:
            for line in fin:
                if pre_tokenizer:
                    res = pre_tokenizer.tokenize(line.strip())
                else:
                    res = line.strip().split()
                in_corpus.append(res)

    print('Start training word2vec model...')
    start = time.time()
    model = Word2Vec(sentences=in_corpus,
                     vector_size=vector_dim,
                     window=window,
                     min_count=min_count,
                     sg=skip_gram,
                     workers=num_workers if num_workers else os.cpu_count() - 1,
                     epochs=epochs,
                     **kwargs)
    print('Time cost: ', time.time() - start)

    model.wv.save_word2vec_format(os.path.join(save_path, 'word2vec768d.txt'), binary=False)  # 不以C语言可以解析的形式存储词向量
    model.save(os.path.join(save_path, 'w2v768d.model'))
    print('Word2vec training done ...')


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    tokenizer = AutoTokenizer.from_pretrained('rna_bpe_tokenizer')
    paths = []

    train_word2vec(paths, 'word2vec', num_workers=12, pre_tokenizer=tokenizer)
