# -*- coding: utf-8 -*-
# file: run_rna_classification_test.py
# time: 05/11/2022 02:38
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.


def test_glove_rna_classification():
    from pyabsa import RNAClassification as RNAC
    from pyabsa.utils.data_utils.dataset_item import DatasetItem
    for model in [
        RNAC.GloVeRNACModelList.LSTM,
        RNAC.GloVeRNACModelList.MHSA,
        RNAC.GloVeRNACModelList.CNN,

    ]:
        config = RNAC.RNACConfigManager.get_rnac_config_glove()
        config.model = model
        config.num_epoch = 1
        config.pretrained_bert = 'rna_bpe_tokenizer'
        config.evaluate_begin = 0
        config.max_seq_len = 512
        config.hidden_dim = 768
        config.embed_dim = 768
        config.cache_dataset = False
        config.dropout = 0.5
        config.num_lstm_layer = 1
        config.do_lower_case = False
        config.seed = 61
        config.log_step = -1
        config.l2reg = 0.001
        config.save_last_ckpt_only = True
        config.num_mhsa_layer = 1

        dataset = DatasetItem('sfe')
        sent_classifier = RNAC.RNACTrainer(config=config,
                                           dataset=dataset,
                                           checkpoint_save_mode=1,
                                           auto_device=True
                                           ).load_trained_model()

        results = sent_classifier.batch_predict(target_file=dataset,
                                                print_result=True,
                                                save_result=True,
                                                ignore_error=False,
                                                )

def test_bert_rna_classification():
    from pyabsa import RNAClassification as RNAC
    from pyabsa.utils.data_utils.dataset_item import DatasetItem
    for model in [
        RNAC.BERTRNACModelList.BERT_MLP

    ]:
        config = RNAC.RNACConfigManager.get_rnac_config_english()
        config.model = model
        config.num_epoch = 1
        config.pretrained_bert = 'roberta-base'
        config.evaluate_begin = 0
        config.max_seq_len = 12
        config.hidden_dim = 768
        config.embed_dim = 768
        config.cache_dataset = False
        config.dropout = 0.5
        config.num_lstm_layer = 1
        config.do_lower_case = False
        config.seed = 61
        config.log_step = -1
        config.l2reg = 0.001
        config.save_last_ckpt_only = True
        config.num_mhsa_layer = 1

        dataset = DatasetItem('sfe')
        sent_classifier = RNAC.RNACTrainer(config=config,
                                           dataset=dataset,
                                           checkpoint_save_mode=1,
                                           auto_device=True
                                           ).load_trained_model()

        results = sent_classifier.batch_predict(target_file=dataset,
                                                print_result=True,
                                                save_result=True,
                                                ignore_error=False,
                                                )
