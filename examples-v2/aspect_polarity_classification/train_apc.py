# -*- coding: utf-8 -*-
# file: trainer.py
# time: 2021/5/26 0026
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import random

from pyabsa.tasks.AspectPolarityClassification import APCDatasetList

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
########################################################################################################################

from pyabsa import AspectPolarityClassification as APC, ModelSaveOption, DeviceTypeOption
import warnings
warnings.filterwarnings('ignore')

for dataset in [
    APCDatasetList.Laptop14,
    APCDatasetList.Restaurant14,
    APCDatasetList.Restaurant15,
    APCDatasetList.Restaurant16,
    # APCDatasetList.MAMS
]:
    # config = APC.APCConfigManager.get_apc_config_english()
    # config.model = APC.APCModelList.FAST_LSA_T_V2
    # # config.model = APC.APCModelList.FAST_LSA_S_V2
    # # config.model = APC.APCModelList.BERT_SPC_V2
    # config.pretrained_bert = 'microsoft/deberta-v3-base'
    # config.evaluate_begin = 5
    # config.max_seq_len = 80
    # config.num_epoch = 30
    # config.log_step = -1
    # config.dropout = 0
    # config.cache_dataset = False
    # config.l2reg = 1e-8
    # config.lsa = True
    # config.seed = [random.randint(0, 10000) for _ in range(5)]
    #
    # APC.APCTrainer(config=config,
    #                dataset=dataset,
    #                checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
    #                auto_device=DeviceTypeOption.AUTO,
    #                # load_aug=True
    #                ).destroy()
    #
    # config = APC.APCConfigManager.get_apc_config_english()
    # # config.model = APC.APCModelList.FAST_LSA_T_V2
    # config.model = APC.APCModelList.FAST_LSA_S_V2
    # # config.model = APC.APCModelList.BERT_SPC_V2
    # config.pretrained_bert = 'microsoft/deberta-v3-base'
    # config.evaluate_begin = 5
    # config.max_seq_len = 80
    # config.num_epoch = 30
    # config.log_step = -1
    # config.dropout = 0
    # config.cache_dataset = False
    # config.l2reg = 1e-8
    # config.lsa = True
    # config.seed = [random.randint(0, 10000) for _ in range(5)]
    #
    # APC.APCTrainer(config=config,
    #                dataset=dataset,
    #                checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
    #                auto_device=DeviceTypeOption.AUTO,
    #                # load_aug=True
    #                ).destroy()

    config = APC.APCConfigManager.get_apc_config_english()
    # config.model = APC.APCModelList.FAST_LSA_T_V2
    # config.model = APC.APCModelList.FAST_LSA_S_V2
    config.model = APC.APCModelList.BERT_SPC_V2
    config.pretrained_bert = 'microsoft/deberta-v3-base'
    config.evaluate_begin = 5
    config.max_seq_len = 80
    config.num_epoch = 30
    config.log_step = -1
    config.dropout = 0
    config.cache_dataset = False
    config.l2reg = 1e-8
    config.lsa = True
    config.seed = [random.randint(0, 10000) for _ in range(5)]

    APC.APCTrainer(config=config,
                   dataset=dataset,
                   checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
                   auto_device=DeviceTypeOption.AUTO,
                   # load_aug=True
                   ).destroy()

    # text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent , the [ASP]decor[ASP] cool and understated . $LABEL$ Positive, Positive'
    # sent_classifier.predict(text, print_result=True)
    # sent_classifier.batch_predict(dataset, print_result=True)
