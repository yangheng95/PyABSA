# -*- coding: utf-8 -*-
# file: inferring.py
# time: 2021/5/21 0021
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
# Usage: Evaluate on given text or inference dataset

from pyabsa import find_target_file
from pyabsa import load_sentiment_classifier

# Assume the sent_classifier is loaded or obtained using train function
model_path = 'state_dict/slide_lcfs_bert_cdw_acc87.14'
sent_classifier = load_sentiment_classifier(trained_model_path=model_path,
                                            auto_device=True  # Use CUDA if available
                                            )
# The default loading device is CPU
# load the model to CPU
# sent_classifier.cpu()

# load the model to CUDA (0)
# sent_classifier.cuda()

# # load the model to CPU or CUDA, like cpu, cuda:0, cuda:1, etc.
# sent_classifier.to('cuda:0')

# sent_classifier.to('cuda') if torch.cuda.is_available() else sent_classifier.cpu()

text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent ,' \
       ' the [ASP]decor[ASP] cool and understated . !sent! 1 1'
sent_classifier.infer(text, print_result=True)

# batch inferring returns the results, save the result if necessary using save_result=True
inference_set_path = 'rest16_inferring.dat'  # file or dir
# inference_set_path = 'datasets/restaurant14'  # file or dir
inference_set = find_target_file(inference_set_path, 'infer', exclude_key='result', find_all=True)

for infer_set in inference_set:
    results = sent_classifier.batch_infer(inference_set_path=infer_set,
                                          print_result=True,
                                          save_result=True,
                                          ignore_error=True)
