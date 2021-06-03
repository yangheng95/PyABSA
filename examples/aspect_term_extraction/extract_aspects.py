# -*- coding: utf-8 -*-
# file: extract_aspects.py
# time: 2021/5/27 0027
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import load_aspect_extractor

# 本工具提供的所有功能均属于测试功能，供学习所用， 欢迎帮助维护及提出意见
# 仅仅实现了单条文本抽取方面及分类情感， 后面有精力会实现批量抽取方面

# All the functions provided by this tool are experimental and for learning purpose only,
# welcome to help maintain and put forward suggestions
# There might batch extraction function in the future

examples = ['But the staff was so nice to us .',
            'But the staff was so horrible to us .',
            r'Not only was the food outstanding , but the little ` perks \' were great .',
            'It took half an hour to get our check , which was perfect since we could sit , have drinks and talk !',
            'It was pleasantly uncrowded , the service was delightful , the garden adorable , '
            'the food -LRB- from appetizers to entrees -RRB- was delectable .',
            'How pretentious and inappropriate for MJ Grill to claim that it provides power lunch and dinners !'
            ]

# 从Google Drive下载提供的预训练模型
# Download the provided pre-training model from Google Drive
model_path = 'state_dict/lcf_atepc_cdw'  # pending update

# 如果有需要，使用以下方法自定义情感索引到情感标签的词典， 其中-999为必需的填充， e.g.,
sentiment_map = {0: 'Negative', 1: "Neutral", 2: 'Positive', -999: ''}
# aspect_extractor = load_aspect_extractor(trained_model_path=model_path,
#                                          sentiment_map=sentiment_map,
#                                          auto_device=True  # False means load model on CPU
#                                          )
aspect_extractor = load_aspect_extractor(trained_model_path=model_path,
                                         auto_device=True  # False means load model on CPU
                                         )

atepc_result = aspect_extractor.extract_aspect(examples=examples,  # list-support only, for current
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )

# print(atepc_result)
