# -*- coding: utf-8 -*-
# file: extract_aspects_multilingual.py
# time: 2021/5/27 0027
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import ATEPCCheckpointManager, available_checkpoints

checkpoint_map = available_checkpoints(from_local=False)

examples = ['But the staff was so nice to us .',
            'But the staff was so horrible to us .',
            r'Not only was the food outstanding , but the little ` perks \' were great .',
            'It took half an hour to get our check , which was perfect since we could sit , have drinks and talk !',
            'It was pleasantly uncrowded , the service was delightful , the garden adorable',
            'the food -LRB- from appetizers to entrees -RRB- was delectable .',
            'How pretentious and inappropriate for MJ Grill to claim that it provides power lunch and dinners !',
            '尤 其 是 照 的 大 尺 寸 照 片 时 效 果 也 是 非 常 不 错 的',
            '照 大 尺 寸 的 照 片 的 时 候 手 机 反 映 速 度 太 慢',
            '关 键 的 时 候 需 要 表 现 持 续 影 像 的 短 片 功 能 还 是 很 有 用 的',
            '相 比 较 原 系 列 锐 度 高 了 不 少 这 一 点 好 与 不 好 大 家 有 争 议'
            ]

# 从Google Drive下载提供的预训练模型
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='chinese')

atepc_result = aspect_extractor.extract_aspect(inference_source=examples,  # list-support only, for current
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )

# print(atepc_result)
