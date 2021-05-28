# -*- coding: utf-8 -*-
# file: extract_aspects.py
# time: 2021/5/27 0027
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import load_aspect_extractor

# 本工具提供的所有功能均属于测试功能，供学习所用， 欢迎帮助维护及提出意见
# 仅仅实现了单挑文本抽取方面， 后面有精力会实现批量抽取方面，及抽取方面并预测极性功能

# All the functions provided by this tool are experimental and for learning purpose only,
# welcome to help maintain and put forward suggestions
# only implements the term extraction on single text.
# There might batch extraction function in the future

text = 'But the staff was so nice to us .'
# text = 'But the staff was so horrible to us .'

# 从谷歌下载提供的预训练模型
# Download the provided pre-training model from Google Drive
model_path = 'state_dict/lcf_atepc_cdw_apcacc_84.27_apcf1_77.77_atef1_82.14_rest14'

aspect_extractor = load_aspect_extractor(trained_model_path=model_path,
                                         auto_device=True)

atepc_result = aspect_extractor.extract_aspect([text])
# print(atepc_result)
