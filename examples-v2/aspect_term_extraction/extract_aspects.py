# -*- coding: utf-8 -*-
# file: extract_aspects.py
# time: 2021/5/27 0027
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
from pyabsa import (
    AspectTermExtraction as ATEPC,
    DeviceTypeOption,
    available_checkpoints,
)
from pyabsa import TaskCodeOption

checkpoint_map = available_checkpoints(
    TaskCodeOption.Aspect_Polarity_Classification, show_ckpts=True
)
# checkpoint_map = available_checkpoints()


aspect_extractor = ATEPC.AspectExtractor(
    "multilingual",
)
# aspect_extractor = ATEPC.AspectExtractor('english', auto_device=DeviceTypeOption.AUTO)
# aspect_extractor = ATEPC.AspectExtractor('chinese', auto_device=DeviceTypeOption.AUTO)
# aspect_extractor.predict("  * * * HITS BLUNT * * * What a great time I had discovering a live band and tasty food in this place . Went for momo ’ s , ended with not only momo ’ s but a great variety that was delightful . Great friendly service was appreciated . The Himalayan black salt drink ( maybe needs a little side note to let people what to expect ) . Will be back !")


while True:
    aspect_extractor.predict(input("Please input a sentence: "))


# inference_source = ATEPC.ATEPCDatasetList.Multilingual
# atepc_result = aspect_extractor.batch_predict(
#     inference_source,  #
#     save_result=False,
#     print_result=True,  # print the result
#     pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
#     eval_batch_size=32,
# )
#
# while True:
#     aspect_extractor.predict(input("Please input a sentence: "))
