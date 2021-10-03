# -*- coding: utf-8 -*-
# file: change_loading_device.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


from pyabsa import APCConfigManager, APCCheckpointManager

sent_classifier = APCCheckpointManager.get_sentiment_classifier('checkpoint', auto_device=True)

# The default loading device is CPU， if auto_device=True will load model on CUDA if any
# you can load the model to CPU manually
sent_classifier.cpu()

# load the model to CUDA (0)
sent_classifier.cuda()

# # load the model to CPU or CUDA, like cpu, cuda:0, cuda:1, main.
sent_classifier.to('cuda:0')
