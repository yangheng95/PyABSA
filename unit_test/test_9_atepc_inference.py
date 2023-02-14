# -*- coding: utf-8 -*-
# file: test_atepc_inference.py
# time: 11:12 2023/2/2
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
import os
import shutil
from pyabsa import (
    download_all_available_datasets,
    AspectTermExtraction as ATEPC,
    TaskCodeOption,
)


def test_multilingual_inference():
    if os.path.exists("checkpoints"):
        shutil.rmtree("checkpoints")

    if os.path.exists("integrated_datasets"):
        shutil.rmtree("integrated_datasets")

    if os.path.exists("source_datasets.backup"):
        shutil.rmtree("source_datasets.backup")

    download_all_available_datasets()
    aspect_extractor = ATEPC.AspectExtractor("Multilingual")
    examples = [
        "The food is good",
        "The food is bad",
        "The food is not good",
        "The food is not bad",
    ]
    expected_results = [
        {"aspect": ["food"], "sentiment": ["Positive"]},
        {"aspect": ["food"], "sentiment": ["Negative"]},
        {"aspect": ["food"], "sentiment": ["Negative"]},
        {"aspect": ["food"], "sentiment": ["Neutral"]},
    ]
    count = 0
    for i, example in enumerate(examples):
        result = aspect_extractor.predict(example, pred_sentiment=True)
        if result["aspect"] != expected_results[i]["aspect"]:
            count += 1
        if result["sentiment"] != expected_results[i]["sentiment"]:
            count += 1
    assert count <= 0


def test_english_inference():
    if os.path.exists("checkpoints"):
        shutil.rmtree("checkpoints")

    if os.path.exists("integrated_datasets"):
        shutil.rmtree("integrated_datasets")

    if os.path.exists("source_datasets.backup"):
        shutil.rmtree("source_datasets.backup")

    download_all_available_datasets()
    aspect_extractor = ATEPC.AspectExtractor("English")
    examples = [
        "The food is good",
        "The food is bad",
        "The food is not good",
        "The food is not bad",
    ]
    expected_results = [
        {"aspect": ["food"], "sentiment": ["Positive"]},
        {"aspect": ["food"], "sentiment": ["Negative"]},
        {"aspect": ["food"], "sentiment": ["Negative"]},
        {"aspect": ["food"], "sentiment": ["Neutral"]},
    ]
    count = 0
    for i, example in enumerate(examples):
        result = aspect_extractor.predict(example, pred_sentiment=True)
        if result["aspect"] != expected_results[i]["aspect"]:
            count += 1
        if result["sentiment"] != expected_results[i]["sentiment"]:
            count += 1

    assert count <= 0


# def test_chinese_inference():
#     if os.path.exists('checkpoints'):
#         shutil.rmtree('checkpoints')
#     if os.path.exists("integrated_datasets"):
#         shutil.rmtree("integrated_datasets")
#
#     if os.path.exists("source_datasets.backup"):
#         shutil.rmtree("source_datasets.backup")
#
#     download_all_available_datasets()
#     aspect_extractor = ATEPC.AspectExtractor('Chinese')
#     examples = [
#         '这家餐厅的食物很好吃',
#         '这家餐厅的食物很难吃',
#         '这家餐厅的食物不好吃',
#         '这家餐厅的食物并不难吃'
#     ]
#     expected_results = [
#         {'aspect': ['食 物'], 'sentiment': ['Positive']},
#         {'aspect': ['食 物'], 'sentiment': ['Negative']},
#         {'aspect': ['食 物'], 'sentiment': ['Negative']},
#         {'aspect': ['食 物'], 'sentiment': ['Neutral']}
#     ]
#     count = 0
#     for i, example in enumerate(examples):
#         result = aspect_extractor.predict(example, pred_sentiment=True)
#         if result['aspect'] != expected_results[i]['aspect']:
#             count += 1
#         if result['sentiment'] != expected_results[i]['sentiment']:
#             count += 1
#     assert count <= 2


if __name__ == "__main__":
    test_multilingual_inference()
    test_english_inference()
    # test_chinese_inference()
