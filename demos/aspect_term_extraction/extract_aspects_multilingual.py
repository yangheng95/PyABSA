# -*- coding: utf-8 -*-
# file: extract_aspects_multilingual.py
# time: 2021/5/27 0027
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import ATEPCCheckpointManager, available_checkpoints, ABSADatasetList

available_checkpoint = available_checkpoints()
# Download checkpoint from HuggingFace or GooGle Drive according to the checkpoint name,
# otherwise auto-search locally using the checkpoint name as a keyword.
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='multilingual')

# Load a local checkpoint by specifying the checkpoint path.
# AspectExtractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='./checkpoints/multilingual')

examples = ['But the staff was so nice to us .', '尤其是照的大尺寸照片时效果也是非常不错的']
atepc_result = aspect_extractor.extract_aspect(inference_source=examples, pred_sentiment=True)
