# -*- coding: utf-8 -*-
# file: trainer.py
# time: 2021/5/26 0026
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
########################################################################################################################

from pyabsa import (
    ModelSaveOption,
    DeviceTypeOption,
    DatasetItem,
)

from pyabsa import AspectSentimentTripletExtraction as ASTE


if __name__ == "__main__":
    config = ASTE.ASTEConfigManager.get_aste_config_english()
    config.max_seq_len = 120
    config.log_step = -1
    # config.pretrained_bert = "microsoft/mdeberta-v3-base"
    # config.pretrained_bert = "microsoft/deberta-v3-base"
    config.pretrained_bert = "bert-base-chinese"
    config.num_epoch = 100
    config.learning_rate = 2e-5
    # config.cache_dataset = False
    config.use_amp = True
    config.cache_dataset = True
    config.spacy_model = "zh_core_web_sm"

    # dataset = "Laptop14"
    # dataset = "aste"
    # dataset = "semeval"
    dataset = "chinese"
    trainer = ASTE.ASTETrainer(
        config=config,
        dataset=dataset,
        # from_checkpoint='english',
        checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
        # checkpoint_save_mode=ModelSaveOption.DO_NOT_SAVE_MODEL,
        auto_device=True,
    )
    triplet_extractor = trainer.load_trained_model()

    examples = [
        "I love this laptop, it is very good.",
        "I hate this laptop, it is very bad.",
        "I like this laptop, it is very good.",
        "I dislike this laptop, it is very bad.",
    ]
    for example in examples:
        triplet_extractor.predict("I love this laptop, it is very good.")
