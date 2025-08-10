# -*- coding: utf-8 -*-
# file: run_dataset_downloading_test.py
# time: 05/11/2022 21:19
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import os
import shutil

from pyabsa import (
    download_all_available_datasets,
    download_dataset_by_name,
    TaskCodeOption,
)
from pyabsa.tasks.AspectPolarityClassification import APCDatasetList


def test_download_dataset_by_name():
    if os.path.exists("integrated_datasets"):
        shutil.rmtree("integrated_datasets")
    download_dataset_by_name(
        TaskCodeOption.Aspect_Polarity_Classification,
        dataset_name=APCDatasetList.English,
    )


def test_download_all_available_dataset():
    if os.path.exists("integrated_datasets"):
        shutil.rmtree("integrated_datasets")
    download_all_available_datasets()


if __name__ == "__main__":
    test_download_dataset_by_name()
    test_download_all_available_dataset()
