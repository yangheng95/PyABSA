# -*- coding: utf-8 -*-
# file: setup.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from os import path

from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))
setup(
    name='pyabsa',
    version='0.1.3alpha',
    description='Train & infer aspect-based sentiment analysis using state-of-the-art models',
    # long_description=str(open(path.join(here, "README.md"), encoding='utf8').read()),
    # The project's main homepage.
    url='https://github.com/yangheng95',
    # Author details
    author='Yang Heng',
    python_requires=">=3.6",
    packages=find_packages(),
    # Choose your license
    license='MIT',
    install_requires=['transformers>=4.4.2', 'spacy', 'networkx'],
)
