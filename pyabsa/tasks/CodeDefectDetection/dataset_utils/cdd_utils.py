# -*- coding: utf-8 -*-
# file: cdd_utils.py
# time: 15:32 2022/12/22
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
import json
import random

import numpy as np

from pyabsa.utils.pyabsa_utils import fprint


def random_indices(source, percentage):
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


def _switch_token(tokens: list, ids: list):
    ids = ids[:-1] if len(ids) % 2 == 1 else ids
    for idx1, idx2 in zip(ids[: len(ids) // 2], list(reversed(ids))[: len(ids) // 2]):
        tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]
    return tokens


def _replace_token(tokens: list, ids: list):
    for idx in ids:
        tokens[idx] = tokens[random.randint(0, len(tokens) - 1)]
    return tokens


def _delete_token(tokens: list, ids: list):
    _tokens = []
    for idx, token in enumerate(tokens):
        if idx in ids:
            continue
        _tokens.append(tokens[idx])
    return tokens


def _add_token(tokens: list, ids: list):
    _tokens = []
    for idx, token in enumerate(tokens):
        if idx in ids:
            _tokens.append(tokens[random.randint(0, len(tokens) - 1)])
        _tokens.append(tokens[idx])
    return tokens


def _prepare_corrupt_code(code_src):
    # perform obfuscation

    # perform noising
    code_tokens = code_src.split()

    replace_ids = random_indices(code_src, random.random() / 10)
    code_tokens = _replace_token(code_tokens, replace_ids)

    deletion_ids = random_indices(code_src, random.random() / 10)
    code_tokens = _delete_token(code_tokens, deletion_ids)

    addition_ids = random_indices(code_src, random.random() / 10)
    code_tokens = _add_token(code_tokens, addition_ids)

    corrupt_code_src = " ".join(code_tokens)

    return corrupt_code_src


def remove_comment(code_str):
    source = code_str
    source = source.replace("/*", "\\n/*").replace("*/", "*/\\n").replace("//", "\\n//")
    # source = source.replace('{', ' ').replace('}', ' ')
    lines = source.split("\\n")
    for i, line in enumerate(lines):
        if "//" in line:
            line = line[: line.find("//")]
        if "/*" and "*/" in lines[i]:
            line = line[: line.find("/*")] + line[line.find("*/") + 2 :]
        lines[i] = line
    code_lines = "\\n".join(lines)
    while "\\n\\n" in code_lines:
        code_lines = code_lines.replace("\\n\\n", "\\n")
    while "  " in code_lines:
        code_lines = code_lines.replace("  ", " ")
    while "\n\n" in code_lines:
        code_lines = code_lines.replace("\n\n", "\n")
    code_lines = code_lines.replace("\\n", "\n")
    return "\\n".join(lines)


def read_defect_examples(lines, data_num, remove_comments=True):
    """Read examples from filename."""
    examples = []
    for idx, line in enumerate(lines):
        try:
            js = json.loads(line)
            code = " ".join(s.strip() for s in js["func"].split())
            if remove_comments:
                code = remove_comment(code)
            examples.append(code + "$LABEL$" + str(js["target"]))
        except Exception as e:
            try:
                code = " ".join(s.strip() for s in line.split())
                if remove_comments:
                    code = remove_comment(code)
                examples.append(code)
            except Exception as e:
                print(e)
                print(line)
                continue

        if idx + 1 == data_num:
            break
    return examples


def calc_stats(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.source)))
            avg_trg_len_tokenize.append(len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        fprint(
            "Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
            len(examples),
            np.mean(avg_src_len),
            np.mean(avg_trg_len),
            max(avg_src_len),
            max(avg_trg_len),
        )
        fprint(
            "[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
            np.mean(avg_src_len_tokenize),
            np.mean(avg_trg_len_tokenize),
            max(avg_src_len_tokenize),
            max(avg_trg_len_tokenize),
        )
    else:
        fprint(
            "Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
            len(examples),
            np.mean(avg_src_len),
            np.mean(avg_trg_len),
            max(avg_src_len),
            max(avg_trg_len),
        )
