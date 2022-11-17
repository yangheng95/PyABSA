# -*- coding: utf-8 -*-
# file: make_absa_dataset.py
# time: 02/11/2022 21:40
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import os

import findfile

from termcolor import colored

from pyabsa import LabelPaddingOption

from pyabsa.tasks.AspectTermExtraction.prediction.aspect_extractor import AspectExtractor


def make_ABSA_dataset(dataset_name_or_path, checkpoint='english'):
    """
    Make APC and ATEPC datasets for PyABSA, using aspect extractor from PyABSA to automatically build datasets. This method WILL NOT give you the best performance but is quite fast and labor-free.
    The names of dataset files to be processed should end with '.raw.ignore'. The files will be processed and saved to the same directory. The files will be overwritten if they already exist.
    The data in the dataset files will be plain text row by row.

    For obtaining the best performance, you should use DPT tool in ABSADatasets to manually annotate the dataset files,
    which can be found in the following link:  https://github.com/yangheng95/ABSADatasets/tree/v1.2/DPT . This tool should be downloaded and run on a browser.

    is much more time-consuming.
    :param dataset_name_or_path: The name of the dataset to be processed. If the name is a directory, all files in the directory will be processed. If it is a file, only the file will be processed.
    If it is a directory name, I use the findfile to find all files in the directory.
    :param checkpoint: Which checkpoint to use. Basically, You can select from {'multilingual', 'english', 'chinese'}, Default is 'english'.
    :return:
    """

    if os.path.isdir(dataset_name_or_path):
        fs = findfile.find_files(dataset_name_or_path, and_key=['.ignore'], exclude_key=['.apc', '.atepc'])
    elif os.path.isfile(dataset_name_or_path):
        fs = [dataset_name_or_path]
    else:
        fs = findfile.find_cwd_files([dataset_name_or_path, '.dat'], exclude_key=['.apc', '.atepc'])
    if fs:
        aspect_extractor = AspectExtractor(checkpoint=checkpoint)
    else:
        print('No files found! Please make sure your dataset names end with ".ignore"')
    for f in fs:

        with open(f, mode='r', encoding='utf8') as f_in:
            lines = f_in.readlines()
        results = aspect_extractor.batch_predict(lines)
        with open(f.replace('.ignore', '') + '.apc', mode='w', encoding='utf-8') as f_apc_out:
            with open(f.replace('.ignore', '') + '.atepc', mode='w', encoding='utf-8') as f_atepc_out:

                for result in results:
                    for j, pos in enumerate(result['position']):
                        for i, (token, IOB) in enumerate(zip(result['tokens'], result['IOB'])):
                            if i + 1 in pos:
                                f_atepc_out.write(
                                    token + ' ' + IOB.replace('[CLS]', 'O').replace('[SEP]', 'O') + ' ' + result['sentiment'][
                                        j - 1] + '\n')
                                result['position'][j].pop(0)
                            else:
                                f_atepc_out.write(
                                    token + ' ' + IOB.replace('[CLS]', 'O').replace('[SEP]', 'O') + ' ' + str(LabelPaddingOption.LABEL_PADDING) + '\n')
                        f_atepc_out.write('\n')

                    for aspect, sentiment in zip(result['aspect'], result['sentiment']):
                        f_apc_out.write(' '.join(result['tokens']) + '\n')
                        f_apc_out.write('{}\n'.format(aspect))
                        f_apc_out.write('{}\n'.format(sentiment))

    print('Datasets built for {}!'.format(' '.join(fs)))
    print(colored(
        'You may need add ID for your dataset, and move the generated datasets to integrated_dataset/apc_datasets and integrated_dataset/atepc_datasets, respectively',
        'red'))
