# -*- coding: utf-8 -*-
# file: setup.py
# time: 2021/4/22 0022
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from setuptools import setup, find_packages

from pyabsa import __name__, __version__
from pathlib import Path

cwd = Path(__file__).parent
long_description = (cwd / "README.md").read_text(encoding="utf8")

extras = {}
# extras = [
#     'git+https://github.com/yangheng95/TextAttack',
#     'tensorflow',
#     'tensorflow_hub',
#     'gensim',
# ]

setup(
    name=__name__,
    version=__version__,
    description="This tool provides the state-of-the-art models for aspect term extraction (ATE), "
    "aspect polarity classification (APC), and text classification (TC).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yangheng95/PyABSA",
    # Author details
    author="Yang, Heng",
    author_email="hy345@exeter.ac.uk",
    python_requires=">=3.7",
    packages=find_packages(),
    include_package_data=True,
    exclude_package_date={"": [".gitignore"]},
    # Choose your license
    license="MIT",
    install_requires=[
        "findfile>=1.7.10",
        "autocuda>=0.14",
        "metric-visualizer>=0.6.6",
        "boostaug>=2.3.0",
        "spacy",
        "networkx",
        "seqeval",
        "update-checker",
        "typing_extensions",
        "tqdm",
        "pytorch_warmup",
        "termcolor",
        "gitpython",  # need git installed in your OS
        "transformers>=4.18.0",
        "torch>=1.0.0",
        "sentencepiece",
        "protobuf<4.0.0",
        "pandas",
    ],
    extras_require=extras,
)
