# -*- coding: utf-8 -*-
# file: atepc_utils.py
# time: 2021/5/27 0027
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import re


# def split_text(s1):
#     # 把句子按字分开，中文按字分，英文按单词，数字按空格
#     regEx = re.compile('[\\W]*')  # 我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
#     res = re.compile(r"([\u4e00-\u9fa5])")  # [\u4e00-\u9fa5]中文范围
#
#     p1 = regEx.split(s1.lower())
#     str1_list = []
#     for str in p1:
#         if res.split(str) == None:
#             str1_list.append(str)
#         else:
#             ret = res.split(str)
#             for ch in ret:
#                 str1_list.append(ch)
#
#     list_word1 = [w for w in str1_list if len(w.strip()) > 0]  # 去掉为空的字符
#
#     return list_word1


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
