# -*- coding: utf-8 -*-
# file: cache_utils.py
# time: 02/11/2022 15:56
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import os
import shutil

import findfile


def clean():
    if os.path.exists("checkpoints"):
        shutil.rmtree("checkpoints")

    if os.path.exists("integrated_datasets"):
        shutil.rmtree("integrated_datasets")

    if os.path.exists("source_datasets.backup"):
        shutil.rmtree("source_datasets.backup")

    if os.path.exists("run"):
        shutil.rmtree("run")

    print("Start cleaning...")
    for f in findfile.find_cwd_files(
        or_key=[".zip", ".cache", ".mv", ".json", ".txt"],
        exclude_key="glove",
        recursive=1,
    ):
        os.remove(f)

    print("Cleaned all files in the current directory.")
