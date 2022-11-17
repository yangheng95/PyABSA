# -*- coding: utf-8 -*-
# file: ATEPC-inference-v1.py
# time: 2022/11/17 17:30
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research

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