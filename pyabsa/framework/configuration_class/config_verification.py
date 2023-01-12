# -*- coding: utf-8 -*-
# file: config_verification.py
# time: 02/11/2022 17:05
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import torch

from pyabsa.utils.pyabsa_utils import fprint

one_shot_messages = set()


def config_check(args):
    try:
        if "SRD" in args:
            assert args["SRD"] >= 0
        if "lcf" in args:
            assert args["lcf"] in {"cdw", "cdm", "fusion"}
        if "window" in args:
            assert args["window"] in {"l", "r", "lr"}
        if "eta" in args:
            assert args["eta"] == -1 or 0 <= args["eta"] <= 1
        if "similarity_threshold" in args:
            assert 0 <= args["similarity_threshold"] <= 1
        if "evaluate_begin" in args:
            assert 0 <= args["evaluate_begin"] < args["num_epoch"]
        if "cross_validate_fold" in args:
            assert args["cross_validate_fold"] == -1 or args["cross_validate_fold"] > 1
            if (
                not 5 <= args["cross_validate_fold"] <= 10
                and not args["cross_validate_fold"] == -1
            ):
                message = "Warning! cross_validate_fold will be better in [5, 10], instead of {}".format(
                    args["cross_validate_fold"]
                )
                if message not in one_shot_messages:
                    fprint(message)
                    one_shot_messages.add(message)
        if "dlcf_a" in args:
            assert args["dlcf_a"] > 1
        if "dca_p" in args:
            assert args["dca_p"] >= 1
        if "dca_layer" in args:
            assert args["dca_layer"] >= 1
        if args["model"].__name__ == "LCA_BERT":
            assert args["lcf"] == "cdm"  # LCA-Net only support CDM mode
        if "ensemble_mode" in args:
            assert args["ensemble_mode"] in {"cat", "mean"}
        if "optimizer" in args:
            if (
                "radam" == args["optimizer"]
                or "nadam" == args["optimizer"]
                or "sparseadam" == args["optimizer"]
                and torch.version.__version__ < "1.10.0"
            ):
                message = "Optimizer {} is not available in PyTorch < 1.10, it will be redirected to Adam instead.".format(
                    args["optimizer"]
                )
                if message not in one_shot_messages:
                    fprint(message)
                    one_shot_messages.add(
                        "Optimizer {} is not available in PyTorch < 1.10, it will be redirected to Adam instead.".format(
                            args["optimizer"]
                        )
                    )
        if "use_amp" in args:
            assert args["use_amp"] in {True, False}

        if "patience" in args:
            assert args["patience"] > 0

    except AssertionError as e:
        raise RuntimeError(
            "Exception: {}. Some parameters are not valid, please see the main example.".format(
                e
            )
        )
