# -*- coding: utf-8 -*-
# file: setup.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from os import path

from setuptools import setup, find_packages

setup(
    name='pyabsa',
    version='0.6.3.2',
    description='This tool provides the sota models for aspect term extraction (ATE) '
                'and aspect polarity classification (APC)',
    # The project's main homepage.
    url='https://github.com/yangheng95/pyabsa',
    # Author details
    author='Yang Heng',
    python_requires=">=3.6",
    packages=find_packages(),
    include_package_data=True,
    data_files=[('config', ['pyabsa/apc/training/training_configs.json'])],
    # Choose your license
    license='MIT',
    install_requires=['transformers>=4.4.2', 'spacy', 'networkx', 'seqeval', 'tqdm'],
)
