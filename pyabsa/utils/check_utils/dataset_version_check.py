# -*- coding: utf-8 -*-
# file: dataset_version_check.py
# time: 02/11/2022 15:51
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

from packaging import version
import requests

from findfile import find_cwd_file
from termcolor import colored

from pyabsa.utils.exception_utils.exception_utils import time_out
from pyabsa.utils.pyabsa_utils import fprint


@time_out(10)
def query_local_datasets_version(**kwargs):
    try:
        fin = open(find_cwd_file(["__init__.py", "integrated_datasets"]))
        local_version = fin.read().split("'")[-2]
        fin.close()
    except:
        return None
    return local_version


@time_out(10)
def query_remote_datasets_version(**kwargs):
    logger = kwargs.get("logger", None)
    try:
        dataset_url = "https://raw.githubusercontent.com/yangheng95/ABSADatasets/v1.2/datasets/__init__.py"
        content = requests.get(dataset_url, timeout=5)
        remote_version = content.text.split("'")[-2]
    except Exception as e:
        try:
            dataset_url = "https://gitee.com/yangheng95/ABSADatasets/raw/v1.2/datasets/__init__.py"
            content = requests.get(dataset_url, timeout=5)
            remote_version = content.text.split("'")[-2]
        except Exception as e:
            if logger:
                logger.warning("Failed to query remote version")
            else:
                fprint(colored("Failed to query remote version", "red"))
            return None
    return remote_version


@time_out(10)
def check_datasets_version(**kwargs):
    """
    Check if the local dataset version is the same as the remote dataset version.
    """
    logger = kwargs.get("logger", None)
    try:
        local_version = query_local_datasets_version()
        remote_version = query_remote_datasets_version()

        if logger is not None:
            logger.info(f"Local dataset version: {local_version}")
            logger.info(f"Remote dataset version: {remote_version}")
        else:
            fprint(f"Local dataset version: {local_version}")
            fprint(f"Remote dataset version: {remote_version}")

        if not remote_version:
            if logger:
                logger.warning(
                    "Failed to check ABSADatasets version, please"
                    "check the latest version of ABSADatasets at https://github.com/yangheng95/ABSADatasets"
                )
            else:
                fprint(
                    colored(
                        "Failed to check ABSADatasets version, please"
                        "check the latest version of ABSADatasets at https://github.com/yangheng95/ABSADatasets",
                        "red",
                    )
                )
        if not local_version:
            if logger:
                logger.warning(
                    "Failed to check local ABSADatasets version, please make sure you have downloaded the latest version of ABSADatasets."
                )
            else:
                fprint(
                    colored(
                        "Failed to check local ABSADatasets version, please make sure you have downloaded the latest version of ABSADatasets.",
                        "red",
                    )
                )

        if version.parse(local_version) < version.parse(remote_version):
            if logger:
                logger.warning(
                    "Local ABSADatasets version is lower than remote ABSADatasets version, please upgrade your ABSADatasets."
                )
            else:
                fprint(
                    colored(
                        "Local ABSADatasets version is lower than remote ABSADatasets version, please upgrade your ABSADatasets.",
                        "red",
                    )
                )

    except Exception as e:
        if logger:
            logger.warning(
                "ABSADatasets version check failed: {}, please check the latest datasets at https://github.com/yangheng95/ABSADatasets manually.".format(
                    e
                )
            )
        else:
            fprint(
                colored(
                    "ABSADatasets version check failed: {}, please check the latest datasets at https://github.com/yangheng95/ABSADatasets manually.".format(
                        e
                    ),
                    "red",
                )
            )
