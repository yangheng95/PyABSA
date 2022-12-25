Adversarial Defense for Text Classification
===========================================

## Inference with adversarial defense

```python3
from pyabsa import TextAdversarialDefense as TAD

classifier = TAD.TADTextClassifier('TAD-SST2')

# suppose that you have an adversarial example with real label 1
text = 'The movie is not good. $LABEL$ 1' # generate adversarial example by the following demo
classifier.predict(text, defense='pwws')

```

## Demo for adversarial defense

```python3
import os
import random
import zipfile
from difflib import Differ

import gradio as gr
import nltk
import pandas as pd
from findfile import find_files

from pyabsa import TADCheckpointManager
from textattack import Attacker
from textattack.attack_recipes import BAEGarg2019, PWWSRen2019, TextFoolerJin2019, PSOZang2020, IGAWang2019, \
    GeneticAlgorithmAlzantot2018, DeepWordBugGao2018
from textattack.attack_results import SuccessfulAttackResult
from textattack.datasets import Dataset
from textattack.models.wrappers import HuggingFaceModelWrapper

z = zipfile.ZipFile('checkpoints.zip', 'r')
z.extractall(os.getcwd())


class ModelWrapper(HuggingFaceModelWrapper):
    def __init__(self, model):
        self.model = model  # pipeline = pipeline

    def __call__(self, text_inputs, **kwargs):
        outputs = []
        for text_input in text_inputs:
            raw_outputs = self.model.infer(text_input, print_result=False, **kwargs)
            outputs.append(raw_outputs['probs'])
        return outputs


class SentAttacker:

    def __init__(self, model, recipe_class=BAEGarg2019):
        model = model
        model_wrapper = ModelWrapper(model)

        recipe = recipe_class.build(model_wrapper)
        # WordNet defaults to english. Set the default language to French ('fra')

        # recipe.transformation.language = "en"

        _dataset = [('', 0)]
        _dataset = Dataset(_dataset)

        self.attacker = Attacker(recipe, _dataset)


def diff_texts(text1, text2):
    d = Differ()
    return [
        (token[2:], token[0] if token[0] != " " else None)
        for token in d.compare(text1, text2)
    ]


def get_ensembled_tad_results(results):
    target_dict = {}
    for r in results:
        target_dict[r['label']] = target_dict.get(r['label']) + 1 if r['label'] in target_dict else 1

    return dict(zip(target_dict.values(), target_dict.keys()))[max(target_dict.values())]


nltk.download('omw-1.4')

sent_attackers = {}
tad_classifiers = {}

attack_recipes = {
    'bae': BAEGarg2019,
    'pwws': PWWSRen2019,
    'textfooler': TextFoolerJin2019,
    'pso': PSOZang2020,
    'iga': IGAWang2019,
    'GA': GeneticAlgorithmAlzantot2018,
    'wordbugger': DeepWordBugGao2018,
}

for attacker in [
    'pwws',
    'bae',
    'textfooler'
]:
    for dataset in [
        'agnews10k',
        'amazon',
        'sst2',
        'imdb'
    ]:
        if 'tad-{}'.format(dataset) not in tad_classifiers:
            tad_classifiers['tad-{}'.format(dataset)] = TADCheckpointManager.get_tad_text_classifier(
                'tad-{}'.format(dataset).upper())

        sent_attackers['tad-{}{}'.format(dataset, attacker)] = SentAttacker(tad_classifiers['tad-{}'.format(dataset)],
                                                                            attack_recipes[attacker])
        tad_classifiers['tad-{}'.format(dataset)].sent_attacker = sent_attackers['tad-{}pwws'.format(dataset)]


def get_sst2_example():
    filter_key_words = ['.py', '.md', 'readme', 'log', 'result', 'zip', '.state_dict', '.model', '.png', 'acc_', 'f1_',
                        '.origin', '.adv', '.csv']

    dataset_file = {'train': [], 'test': [], 'valid': []}
    dataset = 'sst2'
    search_path = './'
    task = 'text_defense'
    dataset_file['test'] += find_files(search_path, [dataset, 'test', task],
                                       exclude_key=['.adv', '.org', '.defense', '.inference',
                                                    'train.'] + filter_key_words)

    for dat_type in [
        'test'
    ]:
        data = []
        label_set = set()
        for data_file in dataset_file[dat_type]:

            with open(data_file, mode='r', encoding='utf8') as fin:
                lines = fin.readlines()
                for line in lines:
                    text, label = line.split('$LABEL$')
                    text = text.strip()
                    label = int(label.strip())
                    data.append((text, label))
                    label_set.add(label)
        return data[random.randint(0, len(data))]


def get_agnews_example():
    filter_key_words = ['.py', '.md', 'readme', 'log', 'result', 'zip', '.state_dict', '.model', '.png', 'acc_', 'f1_',
                        '.origin', '.adv', '.csv']

    dataset_file = {'train': [], 'test': [], 'valid': []}
    dataset = 'agnews'
    search_path = './'
    task = 'text_defense'
    dataset_file['test'] += find_files(search_path, [dataset, 'test', task],
                                       exclude_key=['.adv', '.org', '.defense', '.inference',
                                                    'train.'] + filter_key_words)
    for dat_type in [
        'test'
    ]:
        data = []
        label_set = set()
        for data_file in dataset_file[dat_type]:

            with open(data_file, mode='r', encoding='utf8') as fin:
                lines = fin.readlines()
                for line in lines:
                    text, label = line.split('$LABEL$')
                    text = text.strip()
                    label = int(label.strip())
                    data.append((text, label))
                    label_set.add(label)
        return data[random.randint(0, len(data))]


def get_amazon_example():
    filter_key_words = ['.py', '.md', 'readme', 'log', 'result', 'zip', '.state_dict', '.model', '.png', 'acc_', 'f1_',
                        '.origin', '.adv', '.csv']

    dataset_file = {'train': [], 'test': [], 'valid': []}
    dataset = 'amazon'
    search_path = './'
    task = 'text_defense'
    dataset_file['test'] += find_files(search_path, [dataset, 'test', task],
                                       exclude_key=['.adv', '.org', '.defense', '.inference',
                                                    'train.'] + filter_key_words)

    for dat_type in [
        'test'
    ]:
        data = []
        label_set = set()
        for data_file in dataset_file[dat_type]:

            with open(data_file, mode='r', encoding='utf8') as fin:
                lines = fin.readlines()
                for line in lines:
                    text, label = line.split('$LABEL$')
                    text = text.strip()
                    label = int(label.strip())
                    data.append((text, label))
                    label_set.add(label)
        return data[random.randint(0, len(data))]


def get_imdb_example():
    filter_key_words = ['.py', '.md', 'readme', 'log', 'result', 'zip', '.state_dict', '.model', '.png', 'acc_', 'f1_',
                        '.origin', '.adv', '.csv']

    dataset_file = {'train': [], 'test': [], 'valid': []}
    dataset = 'imdb'
    search_path = './'
    task = 'text_defense'
    dataset_file['test'] += find_files(search_path, [dataset, 'test', task],
                                       exclude_key=['.adv', '.org', '.defense', '.inference',
                                                    'train.'] + filter_key_words)

    for dat_type in [
        'test'
    ]:
        data = []
        label_set = set()
        for data_file in dataset_file[dat_type]:

            with open(data_file, mode='r', encoding='utf8') as fin:
                lines = fin.readlines()
                for line in lines:
                    text, label = line.split('$LABEL$')
                    text = text.strip()
                    label = int(label.strip())
                    data.append((text, label))
                    label_set.add(label)
        return data[random.randint(0, len(data))]


def generate_adversarial_example(dataset, attacker, text=None, label=None):
    if not text:
        if 'agnews' in dataset.lower():
            text, label = get_agnews_example()
        elif 'sst2' in dataset.lower():
            text, label = get_sst2_example()
        elif 'amazon' in dataset.lower():
            text, label = get_amazon_example()
        elif 'imdb' in dataset.lower():
            text, label = get_imdb_example()

    result = None
    attack_result = sent_attackers['tad-{}{}'.format(dataset.lower(), attacker.lower())].attacker.simple_attack(text,
                                                                                                                int(label))
    if isinstance(attack_result, SuccessfulAttackResult):

        if (attack_result.perturbed_result.output != attack_result.original_result.ground_truth_output) and (
                attack_result.original_result.output == attack_result.original_result.ground_truth_output):
            # with defense
            result = tad_classifiers['tad-{}'.format(dataset.lower())].infer(
                attack_result.perturbed_result.attacked_text.text + '$LABEL${},{},{}'.format(
                    attack_result.original_result.ground_truth_output, 1, attack_result.perturbed_result.output),
                print_result=True,
                defense='pwws',
            )

    if result:
        classification_df = {}
        classification_df['pred_label'] = result['label']
        classification_df['confidence'] = round(result['confidence'], 3)
        classification_df['is_correct'] = result['ref_label_check']
        classification_df['is_repaired'] = result['is_fixed']

        advdetection_df = {}
        if result['is_adv_label'] != '0':
            advdetection_df['is_adversary'] = result['is_adv_label']
            advdetection_df['perturbed_label'] = result['perturbed_label']
            advdetection_df['confidence'] = round(result['is_adv_confidence'], 3)
            # advdetection_df['ref_is_attack'] = result['ref_is_adv_label']
            # advdetection_df['is_correct'] = result['ref_is_adv_check']

    else:
        return generate_adversarial_example(dataset, attacker)

    return (text,
            label,
            attack_result.perturbed_result.attacked_text.text,
            diff_texts(text, attack_result.perturbed_result.attacked_text.text),
            diff_texts(text, result['restored_text']),
            attack_result.perturbed_result.output,
            pd.DataFrame(classification_df, index=[0]),
            pd.DataFrame(advdetection_df, index=[0])
            )


demo = gr.Blocks()

with demo:
    with gr.Row():
        with gr.Column():
            input_dataset = gr.Radio(choices=['SST2', 'AGNews10K', 'Amazon', 'IMDB'], value='SST2', label="Dataset")
            input_attacker = gr.Radio(choices=['BAE', 'PWWS', 'TextFooler'], value='TextFooler', label="Attacker")
            input_sentence = gr.Textbox(placeholder='Randomly choose a example from testing set if this box is blank',
                                        label="Sentence")
            input_label = gr.Textbox(placeholder='original label ... ', label="Original Label")

            gr.Markdown("Original Example")

            output_origin_example = gr.Textbox(label="Original Example")
            output_original_label = gr.Textbox(label="Original Label")

            gr.Markdown("Adversarial Example")
            output_adv_example = gr.Textbox(label="Adversarial Example")
            output_adv_label = gr.Textbox(label="Perturbed Label")

            gr.Markdown(
                'This demo is deployed on a CPU device so it may take a long time to execute. Please be patient.')
            button_gen = gr.Button("Click Here to Generate an Adversary and Run Adversary Detection & Repair")

        # Right column (outputs)
        with gr.Column():
            gr.Markdown("Example Difference")
            adv_text_diff = gr.HighlightedText(label="Adversarial Example Difference", combine_adjacent=True)
            restored_text_diff = gr.HighlightedText(label="Restored Example Difference", combine_adjacent=True)
            gr.Markdown(
                'The is_adversary indicates an adversarial example is detected. The perturbed_label is the predicted label of the adversarial example. '
                'The confidence is the confidence of the predicted adversarial example detection.')

            output_is_adv_df = gr.DataFrame(label="Adversary Prediction")
            output_df = gr.DataFrame(label="Standard Classification Prediction")
            gr.Markdown(
                'The pred_label indicates the classification result, and if the is_repaired=true it is repaired by RPD.'
                ' The confidence is the confidence of the predicted label. The is_correct indicates whether the predicted label is correct.')

    # Bind functions to buttons
    button_gen.click(fn=generate_adversarial_example,
                     inputs=[input_dataset, input_attacker, input_sentence, input_label],
                     outputs=[output_origin_example,
                              output_original_label,
                              output_adv_example,
                              adv_text_diff,
                              restored_text_diff,
                              output_adv_label,
                              output_df,
                              output_is_adv_df])

demo.launch()
```
