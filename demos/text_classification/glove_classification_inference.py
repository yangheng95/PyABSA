# -*- coding: utf-8 -*-
# file: 2_classification_glove.py
# time: 2021/8/5
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import TCDatasetList, TCCheckpointManager

model_path = 'lstm'  # 'lstm' is a keyword to search the checkpoint_class in the folder
text_classifier = TCCheckpointManager.get_text_classifier(checkpoint=model_path)

# batch inference works on the dataset files
inference_sets = TCDatasetList.SST2
results = text_classifier.batch_predict(target_file=inference_sets,
                                        print_result=True,
                                        save_result=True,
                                        ignore_error=True,
                                        )
