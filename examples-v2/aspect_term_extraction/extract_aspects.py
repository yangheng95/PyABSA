# -*- coding: utf-8 -*-
# file: extract_aspects.py
# time: 2021/5/27 0027
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
from pyabsa import AspectTermExtraction as ATEPC

# checkpoint_map = available_checkpoints(from_local=False)


aspect_extractor = ATEPC.AspectExtractor('fast_lcf_atepc_Laptop14_cdw_apcacc_68.18_apcf1_51.44_atef1_80.09',
                                         auto_device=True,  # False means load model on CPU
                                         cal_perplexity=True,
                                         )

inference_source = ATEPC.ATEPCDatasetList.SemEval
atepc_result = aspect_extractor.batch_predict(inference_source=inference_source,  #
                                              save_result=True,
                                              print_result=True,  # print the result
                                              pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                              )

print(atepc_result)
