# -*- coding: utf-8 -*-
# file: sentiment_inference_chinese.py
# time: 2021/5/21 0021
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
# Usage: Evaluate on given text or inference dataset

from pyabsa import load_sentiment_classifier

# Assume the sent_classifier is loaded or obtained using train function
model_path = '../state_dict/bert_spc_cdw_acc_97.43_f1_96.94'   # please always check update on Google Drive before using
sent_classifier = load_sentiment_classifier(trained_model_path=model_path,
                                            auto_device=True  # Use CUDA if available
                                            )

# 由于BERT采用单字分词，中文是否用空格分割不影响BERT的表现。欢迎贡献中文或其它语言数据集
chinese_text = '还有就是[ASP]笔画的键盘分布[ASP]我感觉不合理!sent! 0'
sent_classifier.infer(chinese_text, print_result=True)

infer_set = '../apc_datasets/Chinese'  # file or dir, automatic load all inference set from infer_set recursively

results = sent_classifier.batch_infer(target_file=infer_set,
                                      print_result=True,
                                      save_result=True,
                                      ignore_error=True,
                                      )
