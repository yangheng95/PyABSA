# -*- coding: utf-8 -*-
# file: setup.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from setuptools import setup, find_packages

from pyabsa import __name__, __version__
from pathlib import Path

cwd = Path(__file__).parent
long_description = (cwd / "README.md").read_text(encoding='utf8')

setup(
    name=__name__,
    version=__version__,
    description='This tool provides the state-of-the-art models for aspect term extraction (ATE), aspect polarity classification (APC), and text classification (TC).',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yangheng95/PyABSA',
    # Author details
    author='Yang Heng',
    author_email='hy345@exeter.ac.uk',
    python_requires=">=3.6",
    packages=find_packages(),
    include_package_data=True,
    exclude_package_date={'': ['.gitignore']},
    # Choose your license
    license='MIT',
    install_requires=['findfile>=1.7.9.3',
                      'autocuda>=0.11',
                      'metric-visualizer>=0.4.22',
                      'spacy',
                      'networkx',
                      'seqeval',
                      'update-checker',
                      'typing_extensions',
                      'tqdm',
                      'pytorch_warmup',
                      'termcolor',
                      'gitpython',
                      'gdown>=4.4.0',
                      'transformers>4.5',
                      'torch>1.0.0, <1.11.0',
                      'sentencepiece'
                      ],
)
