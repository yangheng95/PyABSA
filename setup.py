# -*- coding: utf-8 -*-
# file: setup.py
# time: 2021/4/22 0022
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from setuptools import setup, find_packages

from pyabsa import __name__, __version__
from pathlib import Path

cwd = Path(__file__).parent
long_description = (cwd / "README.md").read_text(encoding="utf8")

extras = {}
# Packages required for installing docs.
extras["docs"] = [
    "recommonmark",
    "nbsphinx",
    "sphinx-autobuild",
    "sphinx-rtd-theme",
    "sphinx-markdown-tables",
    "sphinx-copybutton",
]
# Packages required for formatting code & running tests.
extras["test"] = [
    "docformatter",
    "isort",
    "flake8",
    "pytest",
    "pytest-xdist",
]

extras["deploy"] = [
    "twine",
    "wheel",
    "setuptools",
    "gradio",
]


extras["tensorflow"] = [
    "tensorflow",
    "tensorflow_hub",
    "tensorflow_text",
    "tensorboardX",
    "tensorflow-estimator",
]

extras["optional"] = [
    "sentence_transformers",
    "tensorflow",
    "tensorflow_hub",
]

# For developers, install development tools along with all optional dependencies.
extras["dev"] = (
    extras["docs"]
    + extras["test"]
    + extras["tensorflow"]
    + extras["optional"]
    + extras["deploy"]
)

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
    python_requires=">=3.8",
    packages=find_packages(),
    include_package_data=True,
    exclude_package_date={"": [".gitignore"]},
    # Choose your license
    license="MIT",
    install_requires=[
        "findfile>=2.0.0",
        "autocuda>=0.16",
        "metric-visualizer>=0.9.6",
        "boostaug>=2.3.5",
        "spacy",
        "networkx",
        "seqeval",
        "update-checker",
        "typing_extensions",
        "tqdm",
        "pytorch_warmup",
        "termcolor",
        "gitpython",  # need git installed in your OS
        "transformers>=4.29.0",
        "torch>=1.0.0",
        "sentencepiece",
        "protobuf<4.0.0",
        "pandas",
    ],
    extras_require=extras,
)
