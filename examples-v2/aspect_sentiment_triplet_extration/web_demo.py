# -*- coding: utf-8 -*-
# file: web_demo.py
# time: 2:37 2023/3/11
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.

import random
import gradio as gr
import pandas as pd
from pyabsa import (
    download_all_available_datasets,
    AspectTermExtraction as ATEPC,
    TaskCodeOption,
    available_checkpoints,
)
from pyabsa import AspectSentimentTripletExtraction as ASTE
from pyabsa.utils.data_utils.dataset_manager import detect_infer_dataset

download_all_available_datasets()

atepc_dataset_items = {dataset.name: dataset for dataset in ATEPC.ATEPCDatasetList()}
aste_dataset_items = {dataset.name: dataset for dataset in ASTE.ASTEDatasetList()}


def get_atepc_example(dataset):
    task = TaskCodeOption.Aspect_Polarity_Classification
    dataset_file = detect_infer_dataset(atepc_dataset_items[dataset], task)

    for fname in dataset_file:
        lines = []
        if isinstance(fname, str):
            fname = [fname]

        for f in fname:
            print("loading: {}".format(f))
            fin = open(f, "r", encoding="utf-8")
            lines.extend(fin.readlines())
            fin.close()
        for i in range(len(lines)):
            lines[i] = (
                lines[i][: lines[i].find("$LABEL$")]
                .replace("[B-ASP]", "")
                .replace("[E-ASP]", "")
                .strip()
            )
        return sorted(set(lines), key=lines.index)


def get_aste_example(dataset):
    task = TaskCodeOption.Aspect_Sentiment_Triplet_Extraction
    dataset_file = detect_infer_dataset(aste_dataset_items[dataset], task)

    for fname in dataset_file:
        lines = []
        if isinstance(fname, str):
            fname = [fname]

        for f in fname:
            print("loading: {}".format(f))
            fin = open(f, "r", encoding="utf-8")
            lines.extend(fin.readlines())
            fin.close()
        return sorted(set(lines), key=lines.index)


available_checkpoints("ASTE", True)

atepc_dataset_dict = {
    dataset.name: get_atepc_example(dataset.name)
    for dataset in ATEPC.ATEPCDatasetList()
}
aspect_extractor = ATEPC.AspectExtractor(checkpoint="multilingual")

aste_dataset_dict = {
    dataset.name: get_aste_example(dataset.name) for dataset in ASTE.ASTEDatasetList()
}
triplet_extractor = ASTE.AspectSentimentTripletExtractor(checkpoint="multilingual")


def perform_atepc_inference(text, dataset):
    if not text:
        text = atepc_dataset_dict[dataset][
            random.randint(0, len(atepc_dataset_dict[dataset]) - 1)
        ]

    result = aspect_extractor.predict(text, pred_sentiment=True)

    result = pd.DataFrame(
        {
            "aspect": result["aspect"],
            "sentiment": result["sentiment"],
            # 'probability': result[0]['probs'],
            "confidence": [round(x, 4) for x in result["confidence"]],
            "position": result["position"],
        }
    )
    return result, "{}".format(text)


def perform_aste_inference(text, dataset):
    if not text:
        text = aste_dataset_dict[dataset][
            random.randint(0, len(aste_dataset_dict[dataset]) - 1)
        ]

    result = triplet_extractor.predict(text)

    pred_triplets = pd.DataFrame(result["Triplets"])
    true_triplets = pd.DataFrame(result["True Triplets"])
    return pred_triplets, true_triplets, "{}".format(text)


demo = gr.Blocks()

with demo:
    with gr.Row():

        with gr.Column():
            gr.Markdown("# <p align='center'>Aspect Sentiment Triplet Extraction !</p>")

            with gr.Row():
                with gr.Column():
                    aste_input_sentence = gr.Textbox(
                        placeholder="Leave this box blank and choose a dataset will give you a random example...",
                        label="Example:",
                    )
                    gr.Markdown(
                        "You can find code and dataset at [ASTE examples](https://github.com/yangheng95/PyABSA/tree/v2/examples-v2/aspect_sentiment_triplet_extration)"
                    )
                    aste_dataset_ids = gr.Radio(
                        choices=[dataset.name for dataset in ASTE.ASTEDatasetList()[:-1]],
                        value="Restaurant14",
                        label="Datasets",
                    )
                    aste_inference_button = gr.Button("Let's go!")

                    aste_output_text = gr.TextArea(label="Example:")
                    aste_output_pred_df = gr.DataFrame(label="Predicted Triplets:")
                    aste_output_true_df = gr.DataFrame(label="Original Triplets:")

                    aste_inference_button.click(
                        fn=perform_aste_inference,
                        inputs=[aste_input_sentence, aste_dataset_ids],
                        outputs=[aste_output_pred_df, aste_output_true_df, aste_output_text],
                    )

        with gr.Column():
            gr.Markdown(
                "# <p align='center'>Multilingual Aspect-based Sentiment Analysis !</p>"
            )
            with gr.Row():
                with gr.Column():
                    atepc_input_sentence = gr.Textbox(
                        placeholder="Leave this box blank and choose a dataset will give you a random example...",
                        label="Example:",
                    )
                    gr.Markdown(
                        "You can find the datasets at [github.com/yangheng95/ABSADatasets](https://github.com/yangheng95/ABSADatasets/tree/v1.2/datasets/text_classification)"
                    )
                    atepc_dataset_ids = gr.Radio(
                        choices=[dataset.name for dataset in ATEPC.ATEPCDatasetList()[:-1]],
                        value="Laptop14",
                        label="Datasets",
                    )
                    atepc_inference_button = gr.Button("Let's go!")

                    atepc_output_text = gr.TextArea(label="Example:")
                    atepc_output_df = gr.DataFrame(label="Prediction Results:")

                    atepc_inference_button.click(
                        fn=perform_atepc_inference,
                        inputs=[atepc_input_sentence, atepc_dataset_ids],
                        outputs=[atepc_output_df, atepc_output_text],
                    )
    gr.Markdown(
        """### GitHub Repo: [PyABSA V2](https://github.com/yangheng95/PyABSA)
        ### Author: [Heng Yang](https://github.com/yangheng95) (杨恒)
        [![Downloads](https://pepy.tech/badge/pyabsa)](https://pepy.tech/project/pyabsa) 
        [![Downloads](https://pepy.tech/badge/pyabsa/month)](https://pepy.tech/project/pyabsa)
        """
    )

demo.launch()