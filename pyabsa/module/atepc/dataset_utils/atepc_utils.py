# -*- coding: utf-8 -*-
# file: atepc_utils.py
# time: 2021/5/27 0027
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import re


def split_text(text):
    text = text.strip()
    word_list = []
    #   输入小写化
    s = text.lower()
    while len(s) > 0:
        match = re.match(r'[a-z]+', s)
        if match:
            word = match.group(0)
        else:
            word = s[0:1]  # 若非英文单词，直接获取第一个字符
        if word:
            word_list.append(word)
        #   从文本中去掉提取的 word，并去除文本收尾的空格字符
        s = s.replace(word, '', 1).strip(' ')
    return word_list
