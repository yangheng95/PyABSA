import os
import random
import gradio as gr
import pandas as pd
import requests

from pyabsa import ATEPCCheckpointManager
from pyabsa.functional.dataset.dataset_manager import download_datasets_from_github, ABSADatasetList, detect_infer_dataset

download_datasets_from_github(os.getcwd())

dataset_items = {dataset.name: dataset for dataset in ABSADatasetList()}

URL = 'https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fhuggingface.co%2Fspaces%2Fyangheng%2Fpyabsa_inference&label=Inference%20Count&labelColor=%2337d67a&countColor=%23f47373&style=flat&labelStyle=none'


def get_example(dataset):
    task = 'apc'
    dataset_file = detect_infer_dataset(dataset_items[dataset], task)

    for fname in dataset_file:
        lines = []
        if isinstance(fname, str):
            fname = [fname]

        for f in fname:
            print('loading: {}'.format(f))
            fin = open(f, 'r', encoding='utf-8')
            lines.extend(fin.readlines())
            fin.close()
        for i in range(len(lines)):
            lines[i] = lines[i][:lines[i].find('!sent!')].replace('[ASP]', '')
        return sorted(set(lines), key=lines.index)


dataset_dict = {dataset.name: get_example(dataset.name) for dataset in ABSADatasetList()}
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='fast_lcf_atepc_Multilingual_cdw_apcacc_86.68_apcf1_80.63_atef1_75.15')


def perform_inference(text, dataset):
    if not text:
        text = dataset_dict[dataset][random.randint(0, len(dataset_dict[dataset]) - 1)]

    result = aspect_extractor.extract_aspect(inference_source=[text],
                                             pred_sentiment=True)

    result = pd.DataFrame({
        'aspect': result[0]['aspect'],
        'sentiment': result[0]['sentiment'],
        # 'probability': result[0]['probs'],
        'confidence': [round(x, 4) for x in result[0]['confidence']],
        'position': result[0]['position']
    })
    requests.get(URL)
    return result, '{}'.format(text)


demo = gr.Blocks()

with demo:
    gr.Markdown("# <p align='center'>Multilingual Aspect-based Sentiment Analysis !</p>")
    gr.Markdown("""### Repo: [PyABSA](https://github.com/yangheng95/PyABSA)
                ### Author: [Heng Yang](https://github.com/yangheng95) (杨恒)
                [![Downloads](https://pepy.tech/badge/pyabsa)](https://pepy.tech/project/pyabsa) 
                [![Downloads](https://pepy.tech/badge/pyabsa/month)](https://pepy.tech/project/pyabsa)
                """
                )
    gr.Markdown("Your input text should be no more than 80 words, that's the longest text we used in training. However, you can try longer text in self-training ")
    gr.Markdown("**You don't need to split each Chinese (Korean, etc.) token as the provided, just input the natural language text.**")
    output_dfs = []
    with gr.Row():
        with gr.Column():
            input_sentence = gr.Textbox(placeholder='Leave this box blank and choose a dataset will give you a random example...', label="Example:")
            gr.Markdown("You can find the datasets at [github.com/yangheng95/ABSADatasets](https://github.com/yangheng95/ABSADatasets/tree/v1.2/datasets/text_classification)")
            dataset_ids = gr.Radio(choices=[dataset.name for dataset in ABSADatasetList()[:-1]], value='Laptop14', label="Datasets")
            inference_button = gr.Button("Let's go!")
            gr.Markdown("There is a [demo](https://huggingface.co/spaces/yangheng/PyABSA-ATEPC-Chinese) specialized for the Chinese langauge")
            gr.Markdown("This demo support many other language as well, you can try and explore the results of other languages by yourself.")

        with gr.Column():
            output_text = gr.TextArea(label="Example:")
            output_df = gr.DataFrame(label="Prediction Results:")
            output_dfs.append(output_df)

        inference_button.click(fn=perform_inference,
                               inputs=[input_sentence, dataset_ids],
                               outputs=[output_df, output_text])

    gr.Markdown("![visitor badge](https://visitor-badge.glitch.me/badge?page_id=https://huggingface.co/spaces/yangheng/Multilingual-Aspect-Based-Sentiment-Analysis)")
    gr.Markdown("![Visitors]({})".format(URL))

demo.launch(share=True)
