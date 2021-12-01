# -*- coding: utf-8 -*-
# file: train_atepc.py
# time: 2021/5/21 0021
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


########################################################################################################################
#                                               ATEPC training script                                                  #
########################################################################################################################
from pyabsa.functional import ATEPCModelList
from pyabsa.functional import Trainer, ATEPCTrainer
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import ATEPCConfigManager

atepc_config = ATEPCConfigManager.get_atepc_config_english()
atepc_config.num_epoch = 10
atepc_config.evaluate_begin = 4
atepc_config.log_step = 100
atepc_config.dropout = 0.1
atepc_config.l2reg = 0.00005
atepc_config.batch_size = 32
atepc_config.srd_alignment = False

atepc_config.pretrained_bert = 'bert-base-uncased'
atepc_config.model = ATEPCModelList.FAST_LCF_ATEPC
dataset_path = ABSADatasetList.Restaurant14
# or your local dataset: dataset_path = 'your local dataset path'

aspect_extractor = ATEPCTrainer(config=atepc_config,
                                dataset=dataset_path,
                                from_checkpoint='',  # set checkpoint to train on the checkpoint.
                                checkpoint_save_mode=1,
                                auto_device='allcuda'
                                ).load_trained_model()

examples = ['But the staff was so nice to us .',
            'But the staff was so horrible to us .',
            r'Not only was the food outstanding , but the little ` perks \' were great .',
            'It took half an hour to get our check , which was perfect since we could sit , have drinks and talk !',
            'It was pleasantly uncrowded , the service was delightful , the garden adorable , '
            'the food -LRB- from appetizers to entrees -RRB- was delectable .',
            'How pretentious and inappropriate for MJ Grill to claim that it provides power lunch and dinners !'
            ]

inference_source = ABSADatasetList.Laptop14
atepc_result = aspect_extractor.extract_aspect(inference_source=inference_source,
                                               save_result=True,
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )
