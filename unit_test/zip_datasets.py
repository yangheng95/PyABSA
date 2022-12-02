# -*- coding: utf-8 -*-
# file: zip_datasets.py
# time: 05/11/2022 17:10
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import os
import zipfile
from pathlib import Path

import findfile

from pyabsa.utils.pyabsa_utils import fprint


def cascade_zip_datasets():
    # iterate zip all datasets in the folder
    datasets = findfile.find_dirs('integrated_datasets', 'datasets')

    for dataset in datasets:
        if dataset in ['integrated_datasets', 'integrated_datasets.zip', ]:
            continue
        for d in findfile.find_dirs(dataset, ''):
            dataset_name = Path(d).name

            if dataset_name in [os.path.basename(dataset)]:
                continue

            fprint(f'Zip dataset: {dataset_name}')

            zip_file = zipfile.ZipFile(f'integrated_datasets/{os.path.basename(dataset)}.{dataset_name}.zip'.lower(), 'w', zipfile.ZIP_DEFLATED)

            for root, dirs, files in os.walk(d):

                for file in files:
                    zip_file.write(os.path.join(root, file).lower())

            zip_file.close()


if __name__ == '__main__':
    cascade_zip_datasets()
