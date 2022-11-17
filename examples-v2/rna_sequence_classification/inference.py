# -*- coding: utf-8 -*-
# file: ensemble_classification_inference.py
# time: 23/10/2022 15:10
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import RNAClassification as RNAC

model_path = 'lstm_sfe_acc_71.21_f1_41.59'
rna_classifier = RNAC.RNAClassifier(model_path)

# batch inference works on the dataset files
# inference_sets = DatasetItem('sfe')
inference_sets = 'integrated_datasets/rnac_datasets/sfe/pure_splicing_unuq_events_sequence_intron500.csv.test.tc.inference'
# inference_sets = 'pure_splicing_non-unuq_events_sequence_intron500.csv.inference'
results = rna_classifier.batch_predict(target_file=inference_sets,
                                       print_result=True,
                                       save_result=True,
                                       ignore_error=False,
                                       )
