# -*- coding: utf-8 -*-
# file: file_utils.py
# time: 2021/7/13 0020
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


import copy
import json
import os
import pickle
import urllib.request

import torch
from findfile import find_files, find_dir, find_cwd_file
from pyabsa.core.atepc.dataset_utils.atepc_utils import split_text
from termcolor import colored

from pyabsa import __version__

# convert atepc_datasets in this repo for inferring_tutorials
from pyabsa.functional.dataset import DatasetItem
from pyabsa.utils.pyabsa_utils import save_args


def generate_inference_set_for_apc(dataset_path):
    print('To ensure your generation is successful, make sure the dataset name contain "apc" and "dataset" string ')
    if isinstance(dataset_path, DatasetItem):
        dataset_name = dataset_path.dataset_name
    else:
        dataset_name = dataset_path

    train_datasets = find_files(os.getcwd(), ['dataset', 'train', 'apc', dataset_name], exclude_key='.inference')
    valid_datasets = find_files(os.getcwd(), ['dataset', 'valid', 'apc', dataset_name], exclude_key='.inference')
    test_datasets = find_files(os.getcwd(), ['dataset', 'test', 'apc', dataset_name], exclude_key='.inference')
    for file in train_datasets + valid_datasets + test_datasets:
        try:
            fin = open(file, 'r', newline='\n', encoding='utf-8')
            lines = fin.readlines()
            fin.close()
            path_to_save = file + '.inference'
            fout = open(path_to_save, 'w', encoding='utf-8', newline='\n', errors='ignore')

            for i in range(0, len(lines), 3):
                sample = lines[i].strip().replace('$T$', '[ASP]{}[ASP]'.format(lines[i + 1].strip()))
                fout.write(sample + ' !sent! ' + lines[i + 2].strip() + '\n')
            fout.close()
        except:
            print('Unprocessed file:', file)
        print('save in: {}'.format(path_to_save))
    print('Inference set generation finished')


def is_similar(s1, s2):
    count = 0.0
    for token in s1.split(' '):
        if token in s2:
            count += 1
    if count / len(s1.split(' ')) >= 0.8 and count / len(s2.split(' ')) >= 0.8:
        return True
    else:
        return False


def assemble_aspects(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    for i in range(len(lines)):
        if i % 3 == 0 or i % 3 == 1:
            lines[i] = ' '.join(split_text(lines[i].strip())).replace('$ t $', '$T$')
        else:
            lines[i] = lines[i].strip()

    def unify_same_samples(same_samples):
        text = same_samples[0][0].replace('$T$', same_samples[0][1])
        polarities = [-999] * len(text.split())
        tags = ['O'] * len(text.split())
        samples = []
        for sample in same_samples:
            # print(sample)
            polarities_tmp = copy.deepcopy(polarities)

            try:
                asp_begin = (sample[0].split().index('$T$'))
                asp_end = sample[0].split().index('$T$') + len(sample[1].split())
                for i in range(asp_begin, asp_end):
                    polarities_tmp[i] = sample[2]
                    if i - sample[0].split().index('$T$') < 1:
                        tags[i] = 'B-ASP'
                    else:
                        tags[i] = 'I-ASP'
                samples.append([text, tags, polarities_tmp])
            except:
                print('Ignore Error:', sample[0])

        return samples

    samples = []
    aspects_in_one_sentence = []
    for i in range(0, len(lines), 3):

        lines[i] = lines[i].replace('$T$', ' $T$ ').replace('  ', ' ')

        if len(aspects_in_one_sentence) == 0:
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])
            continue
        if is_similar(aspects_in_one_sentence[-1][0], lines[i]):
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])
        else:
            samples.extend(unify_same_samples(aspects_in_one_sentence))
            aspects_in_one_sentence = []
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])
    samples.extend(unify_same_samples(aspects_in_one_sentence))

    return samples


def split_aspects(sentence):
    single_aspect_with_contex = []

    aspect_num = len(sentence[1].split("|"))
    aspects = sentence[1].split("|")
    polarity = sentence[2].split("|")
    pre_position = 0
    aspect_context = sentence[0]
    for i in range(aspect_num):
        aspect_context = aspect_context.replace("$A$", aspects[i], 1)
        single_aspect_with_contex.append(
            (aspect_context[pre_position:aspect_context.find("$A$")], aspects[i], polarity[i]))
        pre_position = aspect_context.find(aspects[i]) + len(aspects[i]) + 1

    return single_aspect_with_contex


def convert_atepc(fname):
    print('converting:', fname)
    dist_fname = fname.replace('apc_datasets', 'atepc_datasets')
    if not os.path.exists(os.path.dirname(dist_fname)):
        os.makedirs(os.path.dirname(dist_fname))
    dist_fname += '.atepc'
    lines = []
    samples = assemble_aspects(fname)

    for sample in samples:
        for token_index in range(len(sample[1])):
            token, label, polarity = sample[0].split()[token_index], sample[1][token_index], sample[2][token_index]
            lines.append(token + " " + label + " " + str(polarity))
        lines.append('\n')

    fout = open(dist_fname, 'w', encoding='utf8')
    for line in lines:
        fout.writelines((line + '\n').replace('\n\n', '\n'))
    fout.close()


def convert_apc_set_to_atepc_set(path):
    print('To ensure your conversion is successful, make sure the dataset name contain "apc" and "dataset" string ')

    if isinstance(path, DatasetItem):
        path = path.dataset_name
    if not os.path.exists(path):
        files = find_files(os.getcwd(), [path, 'dataset', 'apc'], exclude_key='.inference')
    else:
        files = find_files(path, ['dataset', 'apc'], exclude_key='infer')

    print('Find datasets files at {}:'.format(path))
    for f in files:
        print(f)
    for target_file in files:
        if not (target_file.endswith('.inference') or target_file.endswith('.atepc')):
            try:
                convert_atepc(target_file)
            except:
                print('failed to process"{}'.format(target_file))
        else:
            print('Ignore ', target_file)
    print('finished')


# 将数据集中的aspect切割出来
def refactor_chinese_dataset(fname, train_fname, test_fname):
    lines = []
    samples = assemble_aspects(fname)
    positive = 0
    negative = 0
    sum = 0
    # refactor testset
    for sample in samples[:int(len(samples) / 5)]:
        for token_index in range(len(sample[1])):
            token, label, polarty = sample[0].split()[token_index], sample[1][token_index], sample[2][token_index]
            lines.append(token + " " + label + " " + str(polarty))
        lines.append('\n')
        if 1 in sample[2]:
            positive += 1
        else:
            negative += 1
        sum += 1
    print(train_fname + f"sum={sum} positive={positive} negative={negative}")
    if os.path.exists(test_fname):
        os.remove(test_fname)
    fout = open(test_fname, 'w', encoding='utf8')
    for line in lines:
        fout.writelines((line + '\n').replace('\n\n', '\n'))
    fout.close()

    positive = 0
    negative = 0
    sum = 0
    # refactor trainset
    for sample in samples[int(len(samples) / 5):]:
        for token_index in range(len(sample[1])):
            tokens = sample[0].split()
            token, label, polarty = sample[0].split()[token_index], sample[1][token_index], sample[2][token_index]
            lines.append(token + " " + label + " " + str(polarty))
        lines.append('\n')
        if 1 in sample[2]:
            positive += 1
        else:
            negative += 1
        sum += 1
    print(train_fname + f"sum={sum} positive={positive} negative={negative}")
    if os.path.exists(train_fname):
        os.remove(train_fname)
    fout = open(train_fname, 'w', encoding='utf8')
    for line in lines:
        fout.writelines((line + '\n').replace('\n\n', '\n'))
    fout.close()


def detect_error_in_dataset(dataset):
    f = open(dataset, 'r', encoding='utf8')
    lines = f.readlines()
    for i in range(0, len(lines), 3):
        # print(lines[i].replace('$T$', lines[i + 1].replace('\n', '')))
        if i + 3 < len(lines):
            if is_similar(lines[i], lines[i + 3]) and len((lines[i] + " " + lines[i + 1]).split()) != len(
                (lines[i + 3] + " " + lines[i + 4]).split()):
                print(lines[i].replace('$T$', lines[i + 1].replace('\n', '')))
                print(lines[i + 3].replace('$T$', lines[i + 4].replace('\n', '')))


def save_model(opt, model, tokenizer, save_path):
    # Save a trained model, configuration and tokenizer
    if hasattr(model, 'module') or hasattr(model, 'core'):
        # print("save model from data-parallel!")
        model_to_save = model.module
    else:
        # print("save a single cuda model!")
        model_to_save = model
    if opt.save_mode == 1 or opt.save_mode == 2:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        f_config = open(save_path + opt.model_name + '.config', mode='wb')
        f_tokenizer = open(save_path + opt.model_name + '.tokenizer', mode='wb')
        pickle.dump(opt, f_config)
        pickle.dump(tokenizer, f_tokenizer)
        f_config.close()
        f_tokenizer.close()
        save_args(opt, save_path + opt.model_name + '.args.txt')
        if opt.save_mode == 1:
            torch.save(model_to_save.state_dict(), save_path + opt.model_name + '.state_dict')  # save the state dict
        elif opt.save_mode == 2:
            torch.save(model.cpu(), save_path + opt.model_name + '.model')  # save the state dict

    elif opt.save_mode == 3:
        # save the fine-tuned bert model
        model_output_dir = save_path + 'fine-tuned-pretrained-model'
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        output_model_file = os.path.join(model_output_dir, 'pytorch_model.bin')
        output_config_file = os.path.join(model_output_dir, 'config.json')

        if hasattr(model_to_save, 'bert4global'):
            model_to_save = model_to_save.bert4global
        elif hasattr(model_to_save, 'bert'):
            model_to_save = model_to_save.bert
        else:
            raise RuntimeError('No pretrained model found to save')

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        if hasattr(tokenizer, 'tokenizer'):
            tokenizer.tokenizer.save_pretrained(model_output_dir)
        else:
            tokenizer.save_pretrained(model_output_dir)

    else:
        raise ValueError('Invalid save_mode: {}'.format(opt.save_mode))
    model.to(opt.device)


def check_update_log():
    print(colored('check release notes at https://github.com/yangheng95/PyABSA/blob/release/release-note.json', 'red'))


def query_remote_version():
    try:
        dataset_url = 'https://raw.githubusercontent.com/yangheng95/ABSADatasets/v1.2/datasets/__init__.py'
        content = urllib.request.urlopen(dataset_url, timeout=5)
        content = content.read().decode("utf-8").split('\'')
        version = content[-2]
    except Exception as e:
        return 'N.A.'
    return version


def query_local_version():
    try:
        fin = open(find_cwd_file(['__init__.py', 'integrated_datasets']))
        local_version = fin.read().split('\'')[-2]
        fin.close()
    except:
        return 'N.A.'
    return local_version


def check_dataset():  # retry_count is for unstable conn to GitHub
    try:
        local_version = query_local_version()
        remote_version = query_remote_version()
        print('Remote ABSADataset version: {} Local ABSADatasets version: {}'.format(remote_version, local_version))
        if remote_version == 'N.A.':
            print('Unknown remote version for ABSADatasets, please check the latest version of ABSADatasets at https://github.com/yangheng95/ABSADatasets')
        elif local_version == 'N.A.':
            print('Unknown local version for ABSADatasets, please check the latest version of ABSADatasets at https://github.com/yangheng95/ABSADatasets')
        elif remote_version > local_version:
            print(colored('There is a new version of ABSADatasets({}), please remove the downloaded datasets to automatically download the new version.'.format(remote_version), 'green'))
    except Exception as e:
        print(colored('ABSADatasets version check failed: {}, please check the latest datasets at https://github.com/yangheng95/ABSADatasets manually.'.format(e), 'red'))
