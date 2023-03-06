# -*- coding: utf-8 -*-
# file: word2vec.py
# time: 02/11/2022 15:41
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import os
import time

from findfile import find_cwd_files
from transformers import AutoTokenizer

from pyabsa.utils.pyabsa_utils import fprint

# from gensim.models.word2vec import LineSentence

import os
import time
from typing import List
from transformers import AutoTokenizer


def train_word2vec(
    corpus_files: List[str] = None,  # a list of file paths for the input corpus
    save_path: str = "word2vec",  # the directory where the model and vectors will be saved
    vector_dim: int = 300,  # the dimension of the resulting word vectors
    window: int = 5,  # the size of the window used for context
    min_count: int = 1000,  # the minimum count of a word for it to be included in the model
    skip_gram: int = 1,  # whether to use skip-gram (1) or CBOW (0) algorithm
    num_workers: int = None,  # the number of worker threads to use (default: CPU count - 1)
    epochs: int = 10,  # the number of iterations over the corpus
    pre_tokenizer: str = None,  # the name of a tokenizer to use for preprocessing (optional)
    **kwargs
):
    """
    Train a Word2Vec model on a given corpus and save the resulting model and vectors to disk.

    Args:
    - corpus_files: a list of file paths for the input corpus
    - save_path: the directory where the model and vectors will be saved
    - vector_dim: the dimension of the resulting word vectors
    - window: the size of the window used for context
    - min_count: the minimum count of a word for it to be included in the model
    - skip_gram: whether to use skip-gram (1) or CBOW (0) algorithm
    - num_workers: the number of worker threads to use (default: CPU count - 1)
    - epochs: the number of iterations over the corpus
    - pre_tokenizer: the name of a tokenizer to use for preprocessing (optional)
    """
    from gensim.models import Word2Vec

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    in_corpus = []

    if not corpus_files:
        # if corpus_files not specified, find all .txt files in the current working directory
        corpus_files = find_cwd_files(".txt", exclude_key=["word2vec", "ignore"])
    elif isinstance(corpus_files, str):
        # if only one file path is specified, convert it to a list
        corpus_files = [corpus_files]
    else:
        # ensure that corpus_files is a list
        assert isinstance(corpus_files, list)

    # load the input corpus
    fprint("Start loading corpus files:", corpus_files)
    if isinstance(pre_tokenizer, str):
        pre_tokenizer = AutoTokenizer.from_pretrained(pre_tokenizer)
    for f in corpus_files:
        with open(f, "r", encoding="utf-8") as fin:
            for line in fin:
                if pre_tokenizer:
                    res = pre_tokenizer.tokenize(line.strip())
                else:
                    res = line.strip().split()
                in_corpus.append(res)

    # train the Word2Vec model
    fprint("Start training word2vec model")
    start = time.time()
    model = Word2Vec(
        sentences=in_corpus,
        vector_size=vector_dim,
        window=window,
        min_count=min_count,
        sg=skip_gram,
        workers=num_workers if num_workers else os.cpu_count() - 1,
        epochs=epochs,
        **kwargs
    )
    fprint("Time cost: ", time.time() - start)

    model.wv.save_word2vec_format(
        os.path.join(save_path, "word2vec768d.txt"), binary=False
    )  # 不以C语言可以解析的形式存储词向量
    model.save(os.path.join(save_path, "w2v768d.model"))
    fprint("Word2vec training done ")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained("rna_bpe_tokenizer")
    paths = []

    train_word2vec(paths, "word2vec", num_workers=12, pre_tokenizer=tokenizer)
