import os
import random
import gradio as gr
import pandas as pd
from findfile import find_files

from pyabsa import ATEPCCheckpointManager
from pyabsa.functional.dataset.dataset_manager import download_datasets_from_github, ABSADatasetList

download_datasets_from_github(os.getcwd())


def get_example(dataset):
    filter_key_words = ['.py', '.md', 'readme', 'log', 'result', 'zip', '.state_dict', '.model', '.png', 'acc_', 'f1_', '.origin', '.adv', '.csv']
    dataset_file = {'train': [], 'test': [], 'valid': []}
    search_path = './'
    task = 'apc_datasets'
    dataset_file['test'] += find_files(search_path, [dataset, 'test', task, '.inference'], exclude_key=['.adv', '.org', '.defense', 'train.'] + filter_key_words)

    for fname in dataset_file['test']:
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
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='english')


def perform_inference(text, dataset):
    if not text:
        text = dataset_dict[dataset][random.randint(0, len(dataset_dict[dataset]))]

    result = aspect_extractor.extract_aspect(inference_source=[text],
                                             pred_sentiment=True)

    result = pd.DataFrame({
        'aspect': result[0]['aspect'],
        'sentiment': result[0]['sentiment'],
        'position': result[0]['position']
    })

    return result, text


demo = gr.Blocks()

with demo:
    gr.Markdown("# Multilingual Aspect-based Sentiment Analysis!")
    gr.Markdown("### Repo: [PyABSA](https://github.com/yangheng95/PyABSA)")
    gr.Markdown("""### Author: [Heng Yang](https://github.com/yangheng95) (杨恒)
                [![Downloads](https://pepy.tech/badge/pyabsa)](https://pepy.tech/project/pyabsa) 
                [![Downloads](https://pepy.tech/badge/pyabsa/month)](https://pepy.tech/project/pyabsa)
                """
                )
    gr.Markdown("Your input text should be no more than 80 words, that's the longest text we used in training. However, you can try longer text in self-training ")
    output_dfs = []
    with gr.Row():
        with gr.Column():
            input_sentence = gr.Textbox(placeholder='Leave blank to give you a random result...', label="Example:")
            gr.Markdown("You can find the datasets at [github.com/yangheng95/ABSADatasets](https://github.com/yangheng95/ABSADatasets/tree/v1.2/datasets/text_classification)")
            dataset_ids = gr.Radio(choices=[dataset.name for dataset in ABSADatasetList()[:-1]], value='Laptop14', label="Datasets")
            inference_button = gr.Button("Let's go!")
            gr.Markdown("This demo support many other language as well, you can try and explore the results of other languages by yourself.")

        with gr.Column():
            output_df = gr.DataFrame(label="Prediction Results:")
            output_dfs.append(output_df)

        inference_button.click(fn=perform_inference,
                               inputs=[input_sentence, dataset_ids],
                               outputs=[output_df, input_sentence])

    gr.Markdown("![visitor badge](https://visitor-badge.glitch.me/badge?page_id=https://huggingface.co/spaces/yangheng/Multilingual-Aspect-Based-Sentiment-Analysis)")

demo.launch(share=True)
