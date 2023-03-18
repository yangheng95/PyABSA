# -*- coding: utf-8 -*-
# file: instruction.py
# time: 15/03/2023
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.


class Instruction:
    def __init__(self, bos_instruction=None, eos_instruction=None):
        self.bos_instruction = bos_instruction
        self.eos_instruction = eos_instruction

    def set_instruction(self, bos_instruction, eos_instruction):
        self.bos_instruction = bos_instruction
        self.eos_instruction = eos_instruction

    def get_instruction(self):
        return self.bos_instruction, self.eos_instruction


class ATEInstruction(Instruction):
    def __init__(self, bos_instruction=None, eos_instruction=None):
        super().__init__(bos_instruction, eos_instruction)
        if self.bos_instruction is None:
            self.bos_instruction = f"""
Definition: The input are sentences about a product or service. The task is to extract the aspects. Here are some examples:

example 1-
input: I charge it at night and skip taking the cord with me because of the good battery life.
{self.eos_instruction}
aspect:battery life|aspect:cord

example 2-
input: Great food, good size menu, great service and an unpretensious setting.
{self.eos_instruction}
aspect:food|aspect:menu|aspect:service|aspect:setting

Now extract aspects from the following example:
input: """

        if self.eos_instruction is None:
            self.eos_instruction = "\nlet us extract aspects one by one: \n"

        if not self.bos_instruction:
            self.bos_instruction = bos_instruction
        if not self.eos_instruction:
            self.eos_instruction = eos_instruction

    def prepare_input(self, input_text):
        return self.bos_instruction + input_text + self.eos_instruction


class APCInstruction(Instruction):
    def __init__(self, bos_instruction=None, eos_instruction=None):
        super().__init__(bos_instruction, eos_instruction)
        if self.bos_instruction is None:
            self.bos_instruction = f"""
Definition: The input are sentences about a product or service. The task is to extract the aspects and their corresponding polarity. Here are some examples:

example 1-
input: I charge it at night and skip taking the cord with me because of the good battery life.
The aspects are: battery life, cord
{self.eos_instruction}
battery life:positive|cord:positive

example 2-
input: Great food, good size menu, great service and an unpretensious setting.
The aspects are: food, menu, service, setting
{self.eos_instruction}
food:positive|menu:positive|service:positive|setting:positive
    
Now predict aspect sentiments from the following example:

input: """
        if self.eos_instruction is None:
            self.eos_instruction = "\nlet us predict sentiments one by one: \n"

        if not self.bos_instruction:
            self.bos_instruction = bos_instruction
        if not self.eos_instruction:
            self.eos_instruction = eos_instruction

    def prepare_input(self, input_text, aspects):
        return (
            self.bos_instruction
            + input_text
            + f"The aspects are: {aspects}"
            + self.eos_instruction
        )


class OpinionInstruction(Instruction):
    def __init__(self, bos_instruction=None, eos_instruction=None):
        super().__init__(bos_instruction, eos_instruction)
        if self.bos_instruction is None:
            self.bos_instruction = f"""
Definition: The input are sentences about a product or service. The task is to extract the aspects and their corresponding polarity. Here are some examples:

example 1-
input: I charge it at night and skip taking the cord with me because of the good battery life.
The aspects are: battery life, cord
{self.eos_instruction}
battery life:good|cord:NULL
    
example 2-
input: Great food, good size menu, great service and an unpretensious setting.
The aspects are: food, menu, service, setting
{self.eos_instruction}
food:great|menu:good|service:great|setting:unpretensious

Now extract opinions for the following example:
input:"""
        if self.eos_instruction is None:
            self.eos_instruction = "\nlet us extract opinions one by one: \n"

        if not self.bos_instruction:
            self.bos_instruction = bos_instruction
        if not self.eos_instruction:
            self.eos_instruction = eos_instruction

    def prepare_input(self, input_text, aspects):
        return (
            self.bos_instruction
            + input_text
            + f"The aspects are: {aspects}"
            + self.eos_instruction
        )


class CategoryInstruction(Instruction):
    def __init__(self, bos_instruction=None, eos_instruction=None):
        super().__init__(bos_instruction, eos_instruction)
        if self.bos_instruction is None:
            self.bos_instruction = f"""
Definition: The input are sentences about a product or service. The task is to extract the aspects and their corresponding categories. Here are some examples:
    
example 1-
input: I charge it at night and skip taking the cord with me because of the good battery life.
The aspects are: battery life, cord
{self.eos_instruction}
battery life:POWER_SUPPLY#GENERAL|cord:NULL

example 2-
input: Great food, good size menu, great service and an unpretensious setting.
The aspects are: food:FOOD#QUALITY| menu:RESTAURANT#GENERAL|service:SERVICE#GENERAL|setting:SERVICE#GENERAL
{self.eos_instruction}
food:FOOD#QUALITY, menu:RESTAURANT#GENERAL, service:SERVICE#GENERAL, setting:SERVICE#GENERAL

Now extract categories for the following example:
input: """
        if self.eos_instruction is None:
            self.eos_instruction = "\nlet us extract categories one by one: \n"

        if not self.bos_instruction:
            self.bos_instruction = bos_instruction
        if not self.eos_instruction:
            self.eos_instruction = eos_instruction

    def prepare_input(self, input_text, aspects):
        return (
            self.bos_instruction
            + input_text
            + f"The aspects are: {aspects}"
            + self.eos_instruction
        )
