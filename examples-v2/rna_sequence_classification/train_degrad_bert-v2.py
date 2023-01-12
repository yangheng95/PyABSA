# -*- coding: utf-8 -*-
# file: train_degrad.py
# time: 06/11/2022 01:43
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import random

from pyabsa import DatasetItem
from pyabsa.tasks import RNABiClassification as RNAC

# preprocess_rna()
config = RNAC.RNACConfigManager.get_rnac_config_english()
config.model = RNAC.BERTRNACModelList.BERT_MLP
config.num_epoch = 5
config.pretrained_bert = "roberta-base"
config.evaluate_begin = 0
config.max_seq_len = 200
config.hidden_dim = 768
config.embed_dim = 768
# config.cache_dataset = False
config.cache_dataset = False
config.dropout = 0.5
config.l2reg = 0
config.batch_size = 64
config.learning_rate = 2e-5
config.show_metric = True
config.num_lstm_layer = 1
config.do_lower_case = False
config.seed = [random.randint(0, 10000) for _ in range(3)]
config.log_step = -1
config.save_last_ckpt_only = True
config.num_mhsa_layer = 1

dataset = DatasetItem("degrad-v2")

classifier = RNAC.RNACTrainer(
    config=config, dataset=dataset, checkpoint_save_mode=1, auto_device=True
).load_trained_model()

rnas = [
    "ATGGGATAATGGTTTCGTACCAAAAGCTGGTGCGTTCCTTCCTTTTGGTGCTGGAAGCCATCTATGCCCGGGAAATGATCTGGCTAAGCTCGAGATTTCAATTTTTCTTCATCATTTCCTCCTCAAATATCAGGTGAAACGGAGCAACCCCGAATGTCCAGTGATGTATCTGCCTCATACCAGACCAACTGATAATTGCT$LABEL$cds,1",
    "TGTGAGTGAAGAAGATAATGCAGACTCACCTTTTGGTGGGACCTATCCCACTCAAAGGCTACCGTCGATTCTCTTCCTCCTCCTTCTCCGGCGATCTCCTCCCTCCGTCGTCTAACCCTATCGGCCGAGACCTATTCCCTCACCGTCGAAGGCACCGCGACGGCAAATCTCGGAGTTACCGTAATCGCTCGAAAACGACG$LABEL$cds,1",
    "TAAACCGTATTTAAATGGACGATCGATGTATCTTTTGAACAGTTTCCTCGTGAATGCGTTAGGTATGATGGGTTCCGGGAAAACGACTGTAGGGAAGATTATGGCAAGATCGCTTGGTTATACATTCTTTGATTGTGACACTTTGATCGAGCAGGCTATGAAGGGAACTTCTGTAGCTGAGATATTTGAGCATTTCGGTG$LABEL$cds,1",
    "AACCTCAAACCAGAAACACAAGCAACTCTTGTGGACAATATAATGGCCCTAGGATCTGAATGGTTTCAGTCACCCTTGAAGCTTACGACTTTGATTTCTATCTACAAAGTCTTTATTGCACGTAGATACGCCCTCCAGGTGATAAAGGACGTTTTCACGAGGAGGAAAGCGTCCAGAGAAATGTGCGGAGACTTCCTCGA$LABEL$cds,1",
    "CCGTTTGAGTGGAGACGAAGGCGTTTCCGGTTCTCTTCTCTCGTCGGAGTTCTGAGGTAAAAAAAGAATAAGGAGAAGAAGAAAAGCAAAAGCATAAAAGAGAGTAGCAAAGACTGAGAATGGAAAGCTTGGACACTAATTTTCCTGTGCGCCATAGAAAGGTCTCGTTTGAAAGTAAGGGAAACAAGACAGAGATTGTG$LABEL$5utr,1",
]
# classifier = RNAC.RNAClassifier('lstm_degrad_acc_83.03_f1_82.25')
for rna in rnas:
    classifier.predict(rna)

# classifier.batch_predict(dataset)
