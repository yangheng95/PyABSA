# -*- coding: utf-8 -*-
# file: bpe_tokenizer.py
# time: 2022/11/19 15:28
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import os
import time

import findfile
from transformers import AutoTokenizer

from pyabsa.utils.pyabsa_utils import fprint


def train_bpe_tokenizer(corpus_files=None,
                        base_tokenizer='roberta-base',
                        save_path='bpe_tokenizer',
                        vocab_size=60000,
                        min_frequency=1000,
                        special_tokens=None,
                        **kwargs
                        ):
    """
    Train a Byte-Pair Encoding tokenizer.
    Args:
        base_tokenizer: The base tokenizer template from transformer tokenizers.
        e.g., you can pass any name of the pretrained tokenizer from https://huggingface.co/models
        corpus_files: input corpus files organized line by line.
        save_path: save path of the tokenizer.
        vocab_size: the size of the vocabulary.
        min_frequency: the minimum frequency of the tokens.
        special_tokens: special tokens to add to the vocabulary.
        **kwargs:

    Returns:

    """
    from tokenizers import ByteLevelBPETokenizer

    if special_tokens is None:
        special_tokens = [
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ]

    if not corpus_files:
        corpus_files = findfile.find_cwd_files('.txt', exclude_key=['word2vec', 'ignore'])
    elif isinstance(corpus_files, str):
        corpus_files = [corpus_files]
    else:
        assert isinstance(corpus_files, list)
    fprint('Start loading corpus files:', corpus_files)

    tokenizer = ByteLevelBPETokenizer()
    # Customize training
    fprint('Start training BPE tokenizer...')
    start = time.time()
    tokenizer.train(files=corpus_files, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)
    fprint('Time cost: ', time.time() - start)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    assert isinstance(base_tokenizer, str), 'base_tokenizer must be a string of the pretrained tokenizer name.'
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(base_tokenizer)
    config.save_pretrained(f'{save_path}/')

    tokenizer.save(f'{save_path}/tokenizer.json')
    tokenizer.save_model(f'{save_path}/')
    fprint('BPE tokenizer training done ...')


if __name__ == '__main__':
    train_bpe_tokenizer(corpus_files=None,
                        base_tokenizer='roberta-base',
                        save_path='bpe_tokenizer'
                        )

    tokenizer = AutoTokenizer.from_pretrained('rna_bpe_tokenizer2')

    output = tokenizer.tokenize(
        "GTGAGTTTACATATTCCTTTTATATACCGTTATCACTCATGATTAGGTGATCATAATTGGTAGGAAGAACCTTTGGTTAAGTAAGGTAAAAGAA"
        "ATAGGCTAGTTCGTGCCAAATATTTCTTAATGAATACAATTCAGATAGATGTTTACTGCAGAGTTATTTTTTGAGCTTTGGTTGCTGGTAGTAGTC"
        "GCCAATTCCAGAAATTGTGGTTTTAGGATCCTCTCAGTTTTATAAATTCAAGCAGTGATTCTTTCCTTGAAGATTTATGTTCCGTCTAAGTAGTTCAAAGGTTTTGTATATTTATACTGCTCTTATTATCTTTCTTTTTTGAAATTGCAG")
    fprint(output)
