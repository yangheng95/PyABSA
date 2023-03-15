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
from .instructions import InstructionsHandler


class ABSAGenerator:
    instruct_handler = InstructionsHandler()
    instruct_handler.load_instruction_set1()
    prompt = instruct_handler.multitask["bos_instruct3"]
    eos_prompt = instruct_handler.multitask["eos_instruct"]

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
                text_or_path[i] = self.prompt + t + self.eos_prompt
                outputs = self.model.predict(text_or_path[i])
                data = {}
                try:
                    data["aspect"] = outputs[0].split("|")
                except Exception as e:
                    data["aspect"] = [""]

                try:
                    data["opinion"] = outputs[1].split("|")
                except Exception as e:
                    data["opinion"] = [""]

                try:
                    data["polarity"] = outputs[2].split("|")
                except Exception as e:
                    data["polarity"] = [""]

                try:
                    data["category"] = outputs[3].split("|")
                except Exception as e:
                    data["category"] = [""]

                results.append(data)
            return results
        elif isinstance(text_or_path, str):
            example = self.prompt + text_or_path + self.eos_prompt
            outputs = self.model.predict(example)
            outputs, _, _ = outputs[0].partition("<EndOfAnswer>")
            outputs = outputs.strip().split(",")
            data = {}
            try:
                data["aspect"] = outputs[0].split("|")
            except Exception as e:
                data["aspect"] = ""

            try:
                data["opinion"] = outputs[1].split("|")
            except Exception as e:
                data["opinion"] = ""

            try:
                data["polarity"] = outputs[2].split("|")
            except Exception as e:
                data["polarity"] = ""

            try:
                data["category"] = outputs[3].split("|")
            except Exception as e:
                data["category"] = ""

            return data
        elif os.path.exists(text_or_path):
            lines = pyabsa.meta_load(text_or_path)
            results = []
            for line in lines:
                example = self.prompt + line.split("\t")[0] + self.eos_prompt
                outputs = self.model.predict(example)
                data = {}
                try:
                    data["aspect"] = outputs[0].split("|")
                except Exception as e:
                    data["aspect"] = ""

                try:
                    data["opinion"] = outputs[1].split("|")
                except Exception as e:
                    data["opinion"] = ""

                try:
                    data["polarity"] = outputs[2].split("|")
                except Exception as e:
                    data["polarity"] = ""

                try:
                    data["category"] = outputs[3].split("|")
                except Exception as e:
                    data["category"] = ""
                results.append(data)
            return results


if __name__ == "__main__":
    instruction_generator = ABSAGenerator()

    lines = pyabsa.meta_load(
        "integrated_datasets/acos_datasets/Restaurant-ACOS/rest16_quad_test.tsv.txt"
    )

    for line in lines:
        print(instruction_generator.predict(line.split("\t")[0]))
