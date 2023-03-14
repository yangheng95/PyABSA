# -*- coding: utf-8 -*-
# file: inference.py
# time: 21:26 2023/3/13
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2023. All Rights Reserved.
import os

from pyabsa.framework.checkpoint_class.checkpoint_template import CheckpointManager

import pyabsa

from .utils import T5Generator


class ABSAGenerator:
    # Definition: The output will be the aspects, opinions, sentiment polarities
    # and aspect categories (both implicit and explicit). In cases where there are no aspects (or aspects, opinions,
    # and aspect categories) the output should be NULL.
    # prompt = """
    #         Example #1
    #         input: I charge it at night and skip taking the cord with me because of the good battery life.
    #         output: aspect:battery life|opinion:good|sentiment:positive|category:POWER_SUPPLY#GENERAL, aspect:cord|opinion:NULL|sentiment:positive|category:POWER_SUPPLY#GENERAL
    #         Example #2
    #         input: Great food, good size menu, great service.
    #         output: aspect:food|opinion:great|sentiment:positive|category:RESTAURANT#GENERAL, aspect:menu|opinion:good size|sentiment:positive|category:RESTAURANT#GENERAL, aspect:service|opinion:great|sentiment:positive|category:RESTAURANT#GENERAL
    #         Now complete the following example-
    #         input: """
    prompt = """Definition: The output will be the aspects, opinions, sentiment polarities
    and aspect categories (both implicit and explicit). In cases where there are no aspects (or aspects, opinions,
    and aspect categories) the output should be NULL.
    Positive example 1-
    input: I charge it at night and skip taking the cord with me because of the good battery life.
    output: aspect:battery life|opinion:good|sentiment:positive|category:POWER_SUPPLY#GENERAL, aspect:cord|opinion:NULL|sentiment:positive|category:POWER_SUPPLY#GENERAL
    Positive example 2-
    input: Great food, good size menu, great service and an unpretensious setting.
    output: aspect:food|opinion:great|sentiment:positive|category:RESTAURANT#GENERAL, aspect:menu|opinion:good size|sentiment:positive|category:RESTAURANT#GENERAL, aspect:service|opinion:great|sentiment:positive|category:RESTAURANT#GENERAL, aspect:setting|opinion:unpretensious|sentiment:positive|category:RESTAURANT#GENERAL
    Now complete the following example-
    input: """

    def __init__(self, checkpoint=None, device=None):
        try:
            checkpoint_path = CheckpointManager().parse_checkpoint(checkpoint, "ACOS")
        except Exception as e:
            print(e)
            checkpoint_path = checkpoint

        self.model = T5Generator(checkpoint_path, device=device)

    def predict(self, text_or_path):
        if isinstance(text_or_path, list):
            results = []
            for i, t in enumerate(text_or_path):
                text_or_path[i] = self.prompt + t + "\n        \noutput:"
                results.append(self.model.predict(text_or_path))
            return results
        elif isinstance(text_or_path, str):
            example = self.prompt + text_or_path + "\n        \noutput:"
            return self.model.predict(example)
        elif os.path.exists(text_or_path):
            lines = pyabsa.meta_load(text_or_path)
            results = []
            for line in lines:
                example = self.prompt + line.split("\t")[0] + "\n        \noutput:"
                results.append(self.model.predict(example))
            return results


if __name__ == "__main__":
    instruction_generator = ABSAGenerator()

    lines = pyabsa.meta_load(
        "integrated_datasets/acos_datasets/Restaurant-ACOS/rest16_quad_test.tsv.txt"
    )

    for line in lines:
        print(instruction_generator.predict(line.split("\t")[0]))
