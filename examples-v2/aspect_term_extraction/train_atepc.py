# -*- coding: utf-8 -*-
# file: train_atepc.py
# time: 2021/5/21 0021
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                                               ATEPC trainer script                                                  #
########################################################################################################################
import random

from pyabsa import AspectTermExtraction as ATEPC

#
# config = ATEPC.ATEPCConfigManager.get_atepc_config_english()
# config.model = ATEPC.ATEPCModelList.BERT_BASE_ATEPC
# config.evaluate_begin = 0
# config.log_step = -1
# config.l2reg = 1e-5
# config.num_epoch = 1
# config.seed = random.randint(1, 100)
# config.use_bert_spc = True
# config.cache_dataset = False
# config.output_dim = 3
# config.num_labels = 6
#
# chinese_sets = ATEPC.ATEPCDatasetList.Laptop14
#
# aspect_extractor = ATEPC.ATEPCTrainer(config=config,
#                                       dataset=chinese_sets,
#                                       checkpoint_save_mode=1,
#                                       auto_device=True
#                                       ).load_trained_model()


atepc_examples = ['But the staff was so nice to us .',
                  'But the staff was so horrible to us .',
                  r'Not only was the food outstanding , but the little ` perks \' were great .',
                  'It took half an hour to get our check , which was perfect since we could sit , have drinks and talk !',
                  'It was pleasantly uncrowded , the service was delightful , the garden adorable , '
                  'the food -LRB- from appetizers to entrees -RRB- was delectable .',
                  'How pretentious and inappropriate for MJ Grill to claim that it provides power lunch and dinners !'
                  ]
from pyabsa import AspectTermExtraction as ATEPC
import torch.cuda
from pyabsa import DeviceTypeOption

# # for dataset in ABSADatasetList():
for dataset in ATEPC.ATEPCDatasetList()[:1]:
    for model in ATEPC.ATEPCModelList():
        config = ATEPC.ATEPCConfigManager.get_atepc_config_english()
        torch.cuda.empty_cache()
        config.model = model
        config.cache_dataset = True
        config.num_epoch = 1
        config.evaluate_begin = 0
        config.max_seq_len = 10
        config.log_step = -1
        config.ate_loss_weight = 5
        config.show_metric = -1
        # config.output_dim = 3
        # config.num_labels = 6
        aspect_extractor = ATEPC.ATEPCTrainer(config=config,
                                              dataset=dataset,
                                              checkpoint_save_mode=1,
                                              auto_device=DeviceTypeOption.ALL_CUDA,
                                              ).load_trained_model()

        aspect_extractor.batch_predict(inference_source=atepc_examples,  #
                                       save_result=True,
                                       print_result=True,  # print the result
                                       pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                       )
