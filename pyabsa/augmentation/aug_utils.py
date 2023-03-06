# -*- coding: utf-8 -*-
# file: utils.py
# time: 15:04 2022/12/31
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
import random
from typing import List


def contextual_code_noise_instance(
    code: str, noise_level: float = 0.15, noise_type: str = "hybrid", **kwargs
) -> str:
    """
    perform contextual noise on code, based on replace, insert, delete operations
    :param code: input code
    :param noise_level: noise level
    :param noise_type: noise type, can be {word, char, token}
    :param kwargs: other arguments
    :return: augmented instance
    """

    def __random_indices(source, percentage):
        assert 0 <= percentage <= 1
        tokens = source.split()
        ids = list(
            set(
                [
                    random.randint(0, len(tokens) - 1)
                    for _ in range(int(len(tokens) * percentage))
                ]
            )
        )
        return ids

    def __replace_token(tokens: list, ids: list):
        for idx in ids:
            tokens[idx] = tokens[random.randint(0, len(tokens) - 1)]
        return tokens

    def __delete_token(tokens: list, ids: list):
        _tokens = []
        for idx, token in enumerate(tokens):
            if idx in ids:
                continue
            _tokens.append(tokens[idx])
        return tokens

    def __add_token(tokens: list, ids: list):
        _tokens = []
        for idx, token in enumerate(tokens):
            if idx in ids:
                _tokens.append(tokens[random.randint(0, len(tokens) - 1)])
            _tokens.append(tokens[idx])
        return tokens

    code_tokens = code.split()
    if noise_type == "replace" or noise_type == "hybrid":
        code_tokens = __replace_token(code_tokens, __random_indices(code, noise_level))

    if noise_type == "delete" or noise_type == "hybrid":
        code_tokens = __delete_token(code_tokens, __random_indices(code, noise_level))

    if noise_type == "insert" or noise_type == "hybrid":
        code_tokens = __add_token(code_tokens, __random_indices(code, noise_level))

    corrupt_code_src = " ".join(code_tokens)

    return corrupt_code_src


def contextual_noise_instance(
    text: str, tokenizer, noise_level: float = 0.15, noise_type: str = "word", **kwargs
):
    """
    :param text: input text
    :param tokenizer: tokenizer
    :param noise_level: noise level
    :param noise_type: noise type, can be {word, char, token}
    :param kwargs: other arguments
    :return: augmented instance
    """
    if noise_type == "word":
        return __word_noise_instance(text, tokenizer, noise_level, **kwargs)
    elif noise_type == "char":
        return __char_noise_instance(text, tokenizer, noise_level, **kwargs)
    elif noise_type == "token":
        return __token_noise_instance(text, tokenizer, noise_level, **kwargs)
    else:
        raise ValueError("Unknown noise type: %s" % noise_type)


def __word_noise_instance(text, tokenizer, noise_level, **kwargs):
    """
    :param text: input text
    :param tokenizer: tokenizer
    :param noise_level: noise level
    :param kwargs: other arguments
    :return: augmented instance
    """
    if len(text) <= 2:
        return text
    words = text.split()
    words = (
        words[0]
        + [
            word if random.random() > noise_level else tokenizer.mask_token
            for word in words[1:-1]
        ]
        + words[-1]
    )
    return " ".join(words)


def __char_noise_instance(text, tokenizer, noise_level, **kwargs):
    """
    :param text: input text
    :param tokenizer: tokenizer
    :param noise_level: noise level
    :param kwargs: other arguments
    :return: augmented instance
    """
    if len(text) <= 2:
        return text
    chars = list(text)
    chars = (
        chars[0]
        + [
            char if random.random() > noise_level else tokenizer.mask_token
            for char in chars[1:-1]
        ]
        + chars[-1]
    )
    return "".join(chars)


def __token_noise_instance(text, tokenizer, noise_level, **kwargs):
    """
    :param text: input text
    :param tokenizer: tokenizer
    :param noise_level: noise level
    :param kwargs: other arguments
    :return: augmented instance
    """
    if len(text) <= 2:
        return text
    tokens = tokenizer.tokenize(text)
    tokens = (
        tokens[0]
        + [
            token if random.random() > noise_level else tokenizer.mask_token
            for token in tokens[1:-1]
        ]
        + tokens[-1]
    )
    return tokenizer.convert_tokens_to_string(tokens)


def contextual_ids_noise_instance(
    ids: List[int],
    tokenizer,
    noise_level: float = 0.15,
    noise_type: str = "mask",
    **kwargs
):
    """
    :param ids: input ids
    :param tokenizer: tokenizer
    :param noise_level: noise level
    :param noise_type: noise type, can be {word, char, token}
    :param kwargs: other arguments
    :return: augmented instance
    """

    if noise_type == "mask":
        return __ids_mask_instance(ids, tokenizer, noise_level, **kwargs)
    elif noise_type == "random":
        return __ids_random__instance(ids, tokenizer, noise_level, **kwargs)
    else:
        raise ValueError("Unknown noise type: %s" % noise_type)


def __ids_mask_instance(ids, tokenizer, noise_level, **kwargs):
    """
    :param ids: input ids
    :param tokenizer: tokenizer
    :param noise_level: noise level
    :param kwargs: other arguments
    :return: augmented instance
    """
    if len(ids) <= 2:
        return ids
    mask_token_id = tokenizer.mask_token_id
    ids = (
        [ids[0]]
        + [
            mask_token_id
            if random.random() < noise_level and _id != tokenizer.eos_token_id
            else _id
            for _id in ids[1:-1]
        ]
        + [ids[-1]]
    )
    return ids


def __ids_random__instance(ids, tokenizer, noise_level, **kwargs):
    """
    :param ids: input ids
    :param tokenizer: tokenizer
    :param noise_level: noise level
    :param kwargs: other arguments
    :return: augmented instance
    """
    if len(ids) <= 2:
        return ids
    ids = (
        [ids[0]]
        + [
            random.randint(0, tokenizer.vocab_size)
            if random.random() < noise_level and _id != tokenizer.eos_token_id
            else _id
            for _id in ids[1:-1]
        ]
        + [ids[-1]]
    )
    return ids
