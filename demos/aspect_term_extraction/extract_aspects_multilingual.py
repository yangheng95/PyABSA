# -*- coding: utf-8 -*-
# file: extract_aspects_multilingual.py
# time: 2021/5/27 0027
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import ATEPCCheckpointManager, available_checkpoints, ABSADatasetList

available_checkpoint = available_checkpoints()
# Download checkpoint_class from HuggingFace or GooGle Drive according to the checkpoint_class name,
# otherwise auto-search locally using the checkpoint_class name as a keyword.
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='multilingual')

# Load a local checkpoint_class by specifying the checkpoint_class path.
# AspectExtractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint_class='./checkpoints/multilingual')

examples = ['But the staff was so nice to us .', '尤其是照的大尺寸照片时效果也是非常不错的']
atepc_result = aspect_extractor.extract_aspect(inference_source=examples, pred_sentiment=True)
