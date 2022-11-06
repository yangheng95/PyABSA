# -*- coding: utf-8 -*-
# file: sentiment_inference_chinese.py
# time: 2021/5/21 0021
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
# Usage: Evaluate on given text or inference dataset

from pyabsa import APCCheckpointManager, ABSADatasetList, available_checkpoints

checkpoint_map = available_checkpoints(from_local=False)

sent_classifier = APCCheckpointManager.get_sentiment_classifier(checkpoint='chinese')
# sent_classifier = APCCheckpointManager.get_sentiment_classifier(checkpoint_class='checkpoint_class')

# # 由于BERT采用单字分词，中文是否用空格分割不影响BERT的表现。欢迎贡献中文或其它语言数据集
# chinese_text = '还有就是[ASP]笔画的键盘分布[ASP]我感觉不合理 !sent! 0'
# sent_classifier.infer(chinese_text, print_result=True)

infer_set = ABSADatasetList.Chinese

results = sent_classifier.batch_predict(target_file=infer_set,
                                        print_result=True,
                                        save_result=True,
                                        ignore_error=True,
                                        )
# print(results)
