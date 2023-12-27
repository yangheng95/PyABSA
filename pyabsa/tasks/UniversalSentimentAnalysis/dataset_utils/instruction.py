# -*- coding: utf-8 -*-
# file: instruction.py
# time: 15/03/2023
# author: HENG YANG <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
import random


# Universal Sentiment Analysis Instruction
class USAInstruction:
    def __init__(self, bos_instruction=None, eos_instruction=None):
        super().__init__()
        self.bos_instruction = bos_instruction
        self.eos_instruction = eos_instruction

        if self.bos_instruction is None:
            self.bos_instruction = """
For sentiment analysis, I will provide you with a series of texts and their potential corresponding inputs. 
Your task is to predict the sentiment analysis output for each text, following the same output format as the provided examples.
"""

        if self.eos_instruction is None:
            self.eos_instruction = "\nlet us analyse the sentiments like the examples, and your answer should be as short as you can: \n"

        if not self.bos_instruction:
            self.bos_instruction = bos_instruction
        if not self.eos_instruction:
            self.eos_instruction = eos_instruction

    def prepare_examples(self, examples, cols_idxs=None):
        str_examples = ""

        for example in examples:
            str_examples += f"Example:\n"
            str_examples += f"text: {str(example['text'])}\n"
            str_examples += f"->{str(example['text'])}\n" if "text" in example else ""

            # str_examples += f"{'|'.join(','.join(numpy.array([f'{k}:{v}' for k, v in d.items()])[cols_idxs]) for d in example['labels'])}\n"
            str_examples += str(example["labels"]) + "\n"

        return str_examples

    def encode_input(self, instance, examples=None, cols_idxs=None):
        _examples = """
Example:
text: I had the best ravioli ever .
->[{'aspect': 'ravioli', 'opinion': 'best', 'polarity': 'positive', 'category': 'NULL'}]
Example:
text: Grilled whole fish wonderful , great spicing .
->[{'aspect': 'fish', 'opinion': 'wonderful', 'polarity': 'positive', 'category': 'NULL'}, {'aspect': 'fish', 'opinion': 'great', 'polarity': 'positive', 'category': 'NULL'}]
"""
        if not examples:
            examples = []
        if not cols_idxs:
            cols_idxs = random.choices(range(4), k=2)
        str_text = "text: "
        str_input = "input: " if "input" in instance else ""
        str_label = ""

        assert isinstance(instance, dict)
        assert "text" in instance
        assert "labels" in instance

        str_text += f"{str(instance['text'])}\n"
        str_input += f"{str(instance['text'])}\n" if "text" in instance else ""

        if not isinstance(instance["labels"], list):
            instance["labels"] = [instance["labels"]]

        # str_label += f"{'|'.join(','.join([f'{k}:{v}' for k, v in d.items()]) for d in instance['labels'])}\n"
        str_label += str(instance["labels"]) + "\n"

        return (
            self.bos_instruction
            + (self.prepare_examples(examples, cols_idxs) if examples else _examples)
            + str_text
            + self.eos_instruction
            + str_input
            + "->"
        ), str_label
