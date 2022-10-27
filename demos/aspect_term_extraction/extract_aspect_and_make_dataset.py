# -*- coding: utf-8 -*-
# file: make_dataset.py
# time: 2022/7/7
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import make_ABSA_dataset

make_ABSA_dataset(dataset_name_or_path='test.xml.dat.inference', checkpoint='english')
