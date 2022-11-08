# -*- coding: utf-8 -*-
# file: extract_aspects.py
# time: 2021/5/27 0027
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
from pyabsa import AspectTermExtraction as ATEPC, DeviceTypeOption

# checkpoint_map = available_checkpoints()


aspect_extractor = ATEPC.AspectExtractor('multilingual', auto_device=DeviceTypeOption.AUTO)
# aspect_extractor = ATEPC.AspectExtractor('english', auto_device=DeviceTypeOption.AUTO)
# aspect_extractor = ATEPC.AspectExtractor('chinese', auto_device=DeviceTypeOption.AUTO)

inference_source = ATEPC.ATEPCDatasetList.Chinese
atepc_result = aspect_extractor.batch_predict(inference_source,  #
                                              save_result=False,
                                              print_result=True,  # print the result
                                              pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                              eval_batch_size=32
                                              )

print(atepc_result)
