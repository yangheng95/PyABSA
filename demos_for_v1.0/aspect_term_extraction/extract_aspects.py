# -*- coding: utf-8 -*-
# file: extract_aspects.py
# time: 2021/5/27 0027
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
from pyabsa import ABSADatasetList, available_checkpoints
from pyabsa import ATEPCCheckpointManager

# checkpoint_map = available_checkpoints(from_local=False)


aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='english',
                                                               auto_device=True,  # False means load model on CPU
                                                               cal_perplexity=True,
                                                               )

inference_source = ABSADatasetList.SemEval
atepc_result = aspect_extractor.extract_aspect(inference_source=inference_source,  #
                                               save_result=True,
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )

print(atepc_result)
