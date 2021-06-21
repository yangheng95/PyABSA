# -*- coding: utf-8 -*-
# file: view_latest_checkpoint.py
# time: 2021/6/20
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import update_checkpoints

checkpoint_map = update_checkpoints()
# You can find the latest checkpoints and use them according to their name, e.g.:
# from pyabsa import APCTrainedModelManager
# model_path = APCTrainedModelManager.get_checkpoint(checkpoint_name='Chinese')
# sent_classifier = load_sentiment_classifier(trained_model_path=model_path,
#                                             auto_device=True  # Use CUDA if available
#                                             )
