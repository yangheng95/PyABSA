# -*- coding: utf-8 -*-
# file: sentiment_inference_multilingual.py
# time: 2021/5/21 0021
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
# Usage: Evaluate on given text or inference dataset

from pyabsa import ABSADatasetList, APCCheckpointManager, available_checkpoints

checkpoint_map = available_checkpoints()

sent_classifier = APCCheckpointManager.get_sentiment_classifier(checkpoint='multilingual')

text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent ,' \
       ' the [ASP]decor[ASP] cool and understated . !sent! Positive, Positive'
sent_classifier.infer(text, print_result=True)

multilingual = ABSADatasetList.Multilingual
sent_classifier.batch_infer(target_file=multilingual,
                            print_result=True,
                            save_result=True,
                            ignore_error=True,
                            )
