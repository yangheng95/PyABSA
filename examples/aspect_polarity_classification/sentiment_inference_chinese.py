# -*- coding: utf-8 -*-
# file: sentiment_inference_chinese.py
# time: 2021/5/21 0021
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
# Usage: Evaluate on given text or inference dataset

from pyabsa import find_target_file
from pyabsa import load_sentiment_classifier

# Assume the sent_classifier is loaded or obtained using train function
model_path = 'state_dict/bert_spc_cdw_acc_96.48_f1_95.91'   # pending update
sent_classifier = load_sentiment_classifier(trained_model_path=model_path,
                                            auto_device=True  # Use CUDA if available
                                            )

# 由于BERT采用单字分词，中文是否用空格分割不影响BERT的表现。欢迎贡献中文或其它语言数据集
chinese_text = '还 有 就 是 [ASP]笔 画 的 键 盘 分 布[ASP] 我 感 觉 不 合 理 !sent! -1'
sent_classifier.infer(chinese_text, print_result=True)
