# -*- coding: utf-8 -*-
# file: ensemble_classification_inference.py
# time: 04/11/2022 16:10
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

from pyabsa import RNARegression as RNAC

regressor = RNAC.RNARegressor("lstm_decay_rate_r2_0.0586")

regressor.batch_predict(
    "integrated_datasets/rnar_datasets/decay_rate/decay_rate.tsv.test.dat.inference"
)
