# -*- coding: utf-8 -*-
# file: atepc_utils.py
# time: 2021/5/27 0027
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

# from transformers import AutoTokenizer
import re
import string

from pyabsa.core.apc.dataset_utils.apc_utils import (get_syntax_distance,
                                                     get_cdw_vec,
                                                     get_lca_ids_and_cdm_vec)


# It is hard to tokenize multilingual text, I decide to use a pretrained tokenizer, you can alter according to your demands
# tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

def split_text(text):
    # text = ' '.join(tokenizer.tokenize(text)[1:])
    # return text
    text = text.strip()

    Chinese_punctuation = '＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
    punctuation = string.punctuation + Chinese_punctuation
    for p in punctuation:
        text = text.replace('{}'.format(p), ' {} '.format(p))
    # text = ' '.join(re.compile(r'\w+|[{}]'.format(re.escape(punctuation))).findall(text)).replace('$ T $', '$T$')

    # for non-latin Languages
    non_latin_unicode = [
        '[\u4e00-\u9fa5]',  # Chinese
        '[\u0800-\u4e00]',  # Japanese
        '[\uac00-\ud7a3]',  # Korean
        '[\u0e00-\u0e7f]',  # Thai
        '[\u1000-\u109F]',  # Myanmar
    ]
    latin_lan = not any([re.match(lan, text) for lan in non_latin_unicode])
    if latin_lan:
        return text.split()

    s = text
    word_list = []
    while len(s) > 0:
        # match = re.match('^({})'.format(''.join(non_latin_unicode)), s)
        match = re.match(r'^[a-z,A-Z]+', s)
        if match:
            word = match.group(0)
        else:
            word = s[0:1]  # 若非英文单词，直接获取第一个字符
        if word:
            word_list.append(word)
        #   从文本中去掉提取的 word，并去除文本收尾的空格字符
        s = s.replace(word, '', 1).strip(' ')
    return word_list


def process_iob_tags(iob_tags: list) -> list:
    for i in range(len(iob_tags) - 1):

        if iob_tags[i] == 'O' and 'ASP' in iob_tags[i + 1]:
            iob_tags[i + 1] = 'B-ASP'

        if 'ASP' in iob_tags[i] and 'B-ASP' in iob_tags[i + 1]:
            iob_tags[i + 1] = 'I-ASP'

    return iob_tags


def prepare_input_for_atepc(opt, tokenizer, text_left, text_right, aspect):
    if hasattr(opt, 'dynamic_truncate') and opt.dynamic_truncate:
        _max_seq_len = opt.max_seq_len - len(aspect.split())
        text_left = text_left.split(' ')
        text_right = text_right.split(' ')
        if _max_seq_len < len(text_left) + len(text_right):
            cut_len = len(text_left) + len(text_right) - _max_seq_len
            if len(text_left) > len(text_right):
                text_left = text_left[cut_len:]
            else:
                text_right = text_right[:len(text_right) - cut_len]
        text_left = ' '.join(text_left)
        text_right = ' '.join(text_right)

    bos_token = tokenizer.bos_token if tokenizer.bos_token else '[CLS]'
    eos_token = tokenizer.eos_token if tokenizer.eos_token else '[SEP]'

    text_raw = text_left + ' ' + aspect + ' ' + text_right
    text_spc = bos_token + ' ' + text_raw + ' ' + eos_token + ' ' + aspect + ' ' + eos_token

    text_bert_tokens = tokenizer.tokenize(text_spc)
    text_raw_bert_tokens = tokenizer.tokenize(bos_token + ' ' + text_raw + ' ' + eos_token)
    aspect_bert_tokens = tokenizer.tokenize(aspect)

    text_bert_indices = tokenizer.convert_tokens_to_ids(text_bert_tokens)
    text_raw_bert_indices = tokenizer.convert_tokens_to_ids(text_raw_bert_tokens)
    aspect_bert_indices = tokenizer.convert_tokens_to_ids(aspect_bert_tokens)

    aspect_begin = len(tokenizer.tokenize(bos_token + ' ' + text_left))

    if 'lcfs' in opt.model_name or opt.use_syntax_based_SRD:
        syntactical_dist, _ = get_syntax_distance(text_raw, aspect, tokenizer, opt)
    else:
        syntactical_dist = None

    lcf_cdm_vec = get_lca_ids_and_cdm_vec(opt, text_bert_indices, aspect_bert_indices,
                                          aspect_begin, syntactical_dist)

    lcf_cdw_vec = get_cdw_vec(opt, text_bert_indices, aspect_bert_indices,
                              aspect_begin, syntactical_dist)

    inputs = {
        'text_raw': text_raw,
        'text_spc': text_spc,
        'aspect': aspect,
        'text_bert_indices': text_bert_indices,
        'text_raw_bert_indices': text_raw_bert_indices,
        'aspect_bert_indices': aspect_bert_indices,
        'lcf_cdm_vec': lcf_cdm_vec,
        'lcf_cdw_vec': lcf_cdw_vec,
    }

    return inputs


def load_atepc_inference_datasets(fname):
    lines = []
    if isinstance(fname, str):
        fname = [fname]

    for f in fname:
        print('loading: {}'.format(f))
        fin = open(f, 'r', encoding='utf-8')
        lines.extend(fin.readlines())
        fin.close()
    for i in range(len(lines)):
        lines[i] = lines[i][:lines[i].find('!sent!')].replace('[ASP]', '')
    return sorted(set(lines), key=lines.index)
