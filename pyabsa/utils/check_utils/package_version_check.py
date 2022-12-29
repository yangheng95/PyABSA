# -*- coding: utf-8 -*-
# file: package_version_check.py
# time: 02/11/2022 15:50
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
from packaging import version

import requests
from termcolor import colored
from update_checker import parse_version, UpdateChecker

from pyabsa import __version__ as current_version
from pyabsa.utils.exception_utils import time_out
from pyabsa.utils.pyabsa_utils import fprint


@time_out(10)
def check_package_version(min_version, max_version=None):
    if version.parse(current_version) < version.parse(min_version):
        raise ValueError(
            f"Current version {current_version} is lower than minimum version {min_version},"
            f" please upgrade your package."
        )
    if max_version is not None and version.parse(current_version) > version.parse(
        max_version
    ):
        raise ValueError(
            f"Current version {current_version} is higher than maximum version {max_version},"
            f" please downgrade your package."
        )


@time_out(10)
def validate_pyabsa_version():
    try:
        response = requests.get("https://pypi.org/pypi/pyabsa/json", timeout=1)
    except requests.exceptions.RequestException:
        return
    if response.status_code == 200:
        data = response.json()
        versions = list(data["releases"].keys())
        versions.sort(key=parse_version, reverse=True)
        if current_version not in versions:
            fprint(
                colored(
                    "You are using a DEPRECATED or TEST version of PyABSA. Consider update using pip install -U pyabsa!",
                    "red",
                )
            )


@time_out(10)
def query_release_notes(**kwargs):
    logger = kwargs.get("logger", None)
    try:
        release_url = (
            "https://github.com/yangheng95/PyABSA/blob/release/release-note.json"
        )
        content = requests.get(release_url, timeout=5)
        release_note = content.json()
        if logger:
            logger.info("Release note: ")
            logger.info(release_note)
        else:
            fprint("Release note: ")
            fprint(release_note[current_version])
    except Exception as e:
        try:
            release_url = (
                "https://gitee.com/yangheng95/PyABSA/raw/release/release-note.json"
            )
            content = requests.get(release_url, timeout=5)
            release_note = content.json()
            if logger:
                logger.info("Release note: ")
                logger.info(release_note[current_version])
        except Exception as e:
            if logger:
                logger.warning(
                    "Failed to query release notes, "
                    "please check the latest version of PyABSA at {}".format(
                        release_url
                    )
                )
            else:
                fprint(
                    colored(
                        "Failed to query release notes, "
                        "please check the latest version of PyABSA at {}".format(
                            release_url
                        ),
                        "red",
                    )
                )


@time_out(10)
def check_pyabsa_update():
    try:
        checker = UpdateChecker()
        check_result = checker.check(__name__, current_version)

        if check_result:
            fprint(check_result)
            query_release_notes()
    except Exception as e:
        fprint("Failed to check update for {}, error: {}".format(__name__, e))
