# -*- coding: utf-8 -*-
# file: run_adversarial_defense_test.py
# time: 05/11/2022 02:39
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
from pyabsa import DatasetItem
from pyabsa import TextAdversarialDefense as TAD


def test_tad_training():
    config = TAD.TADConfigManager.get_tad_config_english()
    config.model = TAD.BERTTADModelList.TADBERT
    config.num_epoch = 1
    config.pretrained_bert = 'bert-base-uncased'
    config.patience = 5
    config.evaluate_begin = 0
    config.max_seq_len = 10
    config.log_step = -1
    config.dropout = 0.5
    config.learning_rate = 1e-5
    config.cache_dataset = False
    config.seed = [2]
    config.l2reg = 1e-5
    config.cross_validate_fold = -1
    config.data_num = 600

    dataset = DatasetItem("SST2TextFooler")

    text_classifier = TAD.TADTrainer(
        config=config, dataset=dataset, checkpoint_save_mode=1, auto_device=True
    ).load_trained_model()

    text_classifier.batch_predict(dataset)


if __name__ == "__main__":
    test_tad_training()
