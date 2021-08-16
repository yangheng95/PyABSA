# -*- coding: utf-8 -*-
# file: extract_aspects_chinese.py
# time: 2021/5/27 0027
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import ATEPCCheckpointManager

# 本工具提供的所有功能均属于测试功能，供学习所用， 欢迎帮助维护及提出意见
# 仅仅实现了列表抽取方面及分类情感， 后面有精力会实现从文件中批量抽取方面

# All the functions provided by this tool are experimental and for learning purpose only,
# welcome to help maintain and put forward suggestions
# There might batch extraction function in the future

examples = ['尤 其 是 照 的 大 尺 寸 照 片 时 效 果 也 是 非 常 不 错 的',
            '照 大 尺 寸 的 照 片 的 时 候 手 机 反 映 速 度 太 慢',
            '关 键 的 时 候 需 要 表 现 持 续 影 像 的 短 片 功 能 还 是 很 有 用 的',
            '相 比 较 原 系 列 锐 度 高 了 不 少 这 一 点 好 与 不 好 大 家 有 争 议',
            '相比较原系列锐度高了不少这一点好与不好大家有争议',
            '这款手机的大小真的很薄，但是颜色不太好看， 总体上我很满意啦。'
            ]

# 从Google Drive下载提供的预训练模型
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='chinese2')
# model_path = 'state_dict/lcf_atepc_cdw_apcacc_96.88_apcf1_96.35_atef1_91.2'

# 如果有需要，使用以下方法自定义情感索引到情感标签的词典， 其中-999为必需的填充， e.g.,
sentiment_map = {0: 'Bad', 1: 'Good', -999: ''}

atepc_result = aspect_extractor.extract_aspect(inference_source=examples,  # list-support only, for now
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )
