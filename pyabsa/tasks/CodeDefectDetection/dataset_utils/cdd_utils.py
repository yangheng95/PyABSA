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
import re
import numpy as np

from pyabsa.utils.pyabsa_utils import fprint


class CodeLineIterator(list):
    def __init__(self, code, strip=True):
        self.code = code
        self.lines = code.split("\n")
        if strip:
            self.lines = [line.strip() for line in self.lines if line.strip() != ""]
        else:
            self.lines = [
                line.strip() + "\n" for line in self.lines if line.strip() != ""
            ]
        super().__init__(self.lines)

    def __getitem__(self, item):
        return self.lines[item]

    def __setitem__(self, key, value):
        self.lines[key] = value

    def __iter__(self):
        return self.lines.__iter__()

    def __len__(self):
        return len(self.lines)

    def __str__(self):
        return "\n".join(self.lines)


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


def remove_comment(code_str, tokenizer=None):
    """
    Remove comments from code string,
    :param code_str: code string
    :param tokenizer: tokenizer if passed, will add <mask> token to the code
    """
    code = code_str

    for s in re.findall(r"/\*.*?\*/", code, re.DOTALL):
        code = code.replace(s, "")
    for s in re.findall(r"//.*", code):
        code = code.replace(s, "")
    for s in re.findall(r"\n[\s]*\n", code, re.DOTALL):
        code = code.replace(s, "\n")
    for s in re.findall(r"\t[\s]*\t", code, re.DOTALL):
        code = code.replace(s, "    ")

    if tokenizer:
        # add <mask> noise
        code_lines = CodeLineIterator(code)
        # line_ids = random.choices(range(len(code_lines)), k=len(code_lines) // random.randint(20, 50))
        line_ids = random.choices(range(len(code_lines)), k=0)
        for line_id in line_ids:
            code_lines[line_id] = "".join(
                [
                    tokenizer.tokenizer.mask_token
                    for _ in range(len(tokenizer.tokenize(code_lines[line_id])))
                ]
            )
        code = "\n".join(code_lines)
    return code


def read_defect_examples(lines, data_num, remove_comments=True, tokenizer=None):
    """Read examples from filename."""
    examples = []
    token_len_sum = 0
    for idx, line in enumerate(lines):
        try:
            js = json.loads(line)
            code = js["func"]
            if remove_comments:
                code = remove_comment(code, tokenizer)
            if tokenizer:
                token_len_sum += len(tokenizer.tokenize(code))
            examples.append(
                code + "$FEATURE$" + str(js["feature"]) + "$LABEL$" + str(js["target"])
            )
        except Exception as e:
            try:
                code = " ".join(s.strip() for s in line.split())
                if remove_comments:
                    code = remove_comment(code, tokenizer)
                examples.append(code)
            except Exception as e:
                print(e)
                print(line)
                continue

        if idx + 1 == data_num:
            break
    if tokenizer:
        print("Average code length: ", token_len_sum / len(examples))
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
