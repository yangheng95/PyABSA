# -*- coding: utf-8 -*-
# file: pyabsa_utils.py
# time: 2021/5/20 0020
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import json
import os
import pickle
import threading
import time
import zipfile

import gdown
import numpy as np
import requests
import torch
import tqdm
from autocuda import auto_cuda, auto_cuda_name
from findfile import find_files, find_cwd_file, find_file
from termcolor import colored
from functools import wraps

from update_checker import parse_version

from pyabsa import __version__

SENTIMENT_PADDING = -999


def save_args(config, save_path):
    f = open(os.path.join(save_path), mode='w', encoding='utf8')
    for arg in config.args:
        if config.args_call_count[arg]:
            f.write('{}: {}\n'.format(arg, config.args[arg]))
    f.close()


def print_args(config, logger=None, mode=0):
    args = [key for key in sorted(config.args.keys())]
    for arg in args:
        if logger:
            logger.info('{0}:{1}\t-->\tCalling Count:{2}'.format(arg, config.args[arg], config.args_call_count[arg]))
        else:
            print('{0}:{1}\t-->\tCalling Count:{2}'.format(arg, config.args[arg], config.args_call_count[arg]))


def validate_example(text: str, aspect: str, polarity: str):
    if len(text) < len(aspect):
        raise ValueError(colored('AspectLengthExceedTextError -> <aspect: {}> is longer than <text: {}>, <polarity: {}>'.format(aspect, text, polarity), 'red'))

    if aspect.strip().lower() not in text.strip().lower():
        raise ValueError(colored('AspectNotInTextError -> <aspect: {}> is not in <text: {}>>'.format(aspect, text), 'yellow'))

    warning = False

    if len(aspect.split(' ')) > 10:
        print(colored('AspectTooLongWarning -> <aspect: {}> is too long, <text: {}>, <polarity: {}>'.format(aspect, text, polarity), 'yellow'))
        warning = True

    if not aspect.strip():
        raise ValueError(colored('AspectIsNullError -> <text: {}>, <aspect: {}>, <polarity: {}>'.format(aspect, text, polarity), 'yellow'))

    if len(polarity.split(' ')) > 3:
        print(colored('LabelTooLongWarning -> <polarity: {}> is too long, <text: {}>, <aspect: {}>'.format(polarity, text, aspect), 'yellow'))
        warning = True

    if not polarity.strip():
        raise ValueError(colored('PolarityIsNullError -> <text: {}>, <aspect: {}>, <polarity: {}>'.format(aspect, text, polarity), 'yellow'))

    if text.strip() == aspect.strip():
        print(colored('AspectEqualsTextWarning -> <aspect: {}> equals <text: {}>, <polarity: {}>'.format(aspect, text, polarity), 'yellow'))
        warning = True

    if not text.strip():
        raise ValueError(colored('TextIsNullError -> <text: {}>, <aspect: {}>, <polarity: {}>'.format(aspect, text, polarity), 'yellow'))

    return warning


def check_and_fix_labels(label_set: set, label_name, all_data, opt):
    # update polarities_dim, init model behind execution of this function!
    if '-100' in label_set:

        label_to_index = {origin_label: int(idx) - 1 if origin_label != '-100' else -100 for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
        index_to_label = {int(idx) - 1 if origin_label != '-100' else -100: origin_label for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
    else:
        label_to_index = {origin_label: int(idx) for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
        index_to_label = {int(idx): origin_label for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
    if 'index_to_label' not in opt.args:
        opt.index_to_label = index_to_label
        opt.label_to_index = label_to_index

    if opt.index_to_label != index_to_label:
        # raise KeyError('Fail to fix the labels, the number of labels are not equal among all datasets!')
        opt.index_to_label.update(index_to_label)
        opt.label_to_index.update(label_to_index)
    num_label = {l: 0 for l in label_set}
    num_label['Sum'] = len(all_data)
    for item in all_data:
        try:
            num_label[item[label_name]] += 1
            item[label_name] = label_to_index[item[label_name]]
        except Exception as e:
            # print(e)
            num_label[item.polarity] += 1
            item.polarity = label_to_index[item.polarity]
    print('Dataset Label Details: {}'.format(num_label))


def check_and_fix_IOB_labels(label_map, opt):
    # update polarities_dim, init model behind execution of this function!
    index_to_IOB_label = {int(label_map[origin_label]): origin_label for origin_label in label_map}
    opt.index_to_IOB_label = index_to_IOB_label


def get_device(auto_device):
    if isinstance(auto_device, str) and auto_device == 'allcuda':
        device = 'cuda'
    elif isinstance(auto_device, str):
        device = auto_device
    elif isinstance(auto_device, bool):
        device = auto_cuda() if auto_device else 'cpu'
    else:
        device = auto_cuda()
        try:
            torch.device(device)
        except RuntimeError as e:
            print(colored('Device assignment error: {}, redirect to CPU'.format(e), 'red'))
            device = 'cpu'
    device_name = auto_cuda_name()
    return device, device_name


def resume_from_checkpoint(trainer, from_checkpoint_path):
    if from_checkpoint_path:
        model_path = find_files(from_checkpoint_path, '.model')
        state_dict_path = find_files(from_checkpoint_path, '.state_dict')
        config_path = find_files(from_checkpoint_path, '.config')

        if from_checkpoint_path:
            if not config_path:
                raise FileNotFoundError('.config file is missing!')
            config = pickle.load(open(config_path[0], 'rb'))
            if model_path:
                if config.model != trainer.opt.model:
                    print(colored('Warning, the checkpoint was not trained using {} from param_dict'.format(trainer.opt.model.__name__)), 'yellow')
                trainer.model = torch.load(model_path[0])
            if state_dict_path:
                if torch.cuda.device_count() > 1 and trainer.opt.device == 'allcuda':
                    trainer.model.module.load_state_dict(torch.load(state_dict_path[0]))
                else:
                    trainer.model.load_state_dict(torch.load(state_dict_path[0]))
                trainer.model.opt = trainer.opt
                trainer.model.to(trainer.opt.device)
            else:
                print(colored('.model or .state_dict file is missing!', 'red'))
        else:
            print(colored('No checkpoint found in {}'.format(from_checkpoint_path), 'red'))
        print(colored('Resume training from Checkpoint: {}!'.format(from_checkpoint_path), 'green'))


def prepare_glove840_embedding(glove_path):
    glove840_id = '1G-vd6W1oF9ByyJ-pzp9dcqKnr_plh4Em'
    if os.path.exists(glove_path) and os.path.isfile(glove_path):
        return glove_path
    else:
        embedding_files = []
        dir_path = os.getenv('$HOME') if os.getenv('$HOME') else os.getcwd()

        if find_file(dir_path, 'glove.42B.300d.txt', exclude_key='.zip'):
            embedding_files += find_files(dir_path, 'glove.42B.300d.txt', exclude_key='.zip')
        elif find_file(dir_path, 'glove.840B.300d.txt', exclude_key='.zip'):
            embedding_files += find_files(dir_path, 'glove.840B.300d.txt', exclude_key='.zip')
        elif find_file(dir_path, 'glove.twitter.27B.txt', exclude_key='.zip'):
            embedding_files += find_files(dir_path, 'glove.twitter.27B.txt', exclude_key='.zip')

        if embedding_files:
            print(colored('Find embedding file: {}, use: {}'.format(embedding_files, embedding_files[0]), 'green'))
            return embedding_files[0]

        else:
            zip_glove_path = os.path.join(os.path.dirname(glove_path), 'glove.840B.300d.zip')
            print(colored('No GloVe embedding found at {},'
                          ' downloading glove.840B.300d.txt (2GB will be downloaded / 5.5GB after unzip)...'.format(glove_path), 'yellow'))
            try:
                response = requests.get('https://huggingface.co/spaces/yangheng/PyABSA-ATEPC/resolve/main/open-access/glove.840B.300d.zip', stream=True)
                with open(zip_glove_path, "wb") as f:
                    for chunk in tqdm.tqdm(response.iter_content(chunk_size=1024 * 1024),
                                           unit='MB',
                                           total=int(response.headers['content-length']) // 1024 // 1024,
                                           postfix=colored('Downloading GloVe-840B embedding...', 'yellow')):
                        f.write(chunk)
            except Exception as e:
                gdown.download(id=glove840_id, output=zip_glove_path)

        if find_cwd_file('glove.840B.300d.zip'):
            print(colored('unzip glove.840B.300d.zip...', 'yellow'))
            with zipfile.ZipFile(find_cwd_file('glove.840B.300d.zip'), 'r') as z:
                z.extractall()
            print(colored('Zip file extraction Done.', 'green'))

        return prepare_glove840_embedding(glove_path)


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in tqdm.tqdm(fin.readlines(), postfix='Loading embedding file...'):
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname, opt):
    if not os.path.exists('run'):
        os.makedirs('run')
    embed_matrix_path = 'run/{}'.format(os.path.join(opt.dataset_name, dat_fname))
    if os.path.exists(embed_matrix_path):
        print(colored('Loading cached embedding_matrix from {} (Please remove all cached files if there is any problem!)'.format(embed_matrix_path), 'green'))
        embedding_matrix = pickle.load(open(embed_matrix_path, 'rb'))
    else:
        glove_path = prepare_glove840_embedding(embed_matrix_path)
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros

        word_vec = _load_word_vec(glove_path, word2idx=word2idx, embed_dim=embed_dim)

        for word, i in tqdm.tqdm(word2idx.items(), postfix=colored('Building embedding_matrix {}'.format(dat_fname), 'yellow')):
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embed_matrix_path, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class TransformerConnectionError(ValueError):
    def __init__(self):
        pass


def retry(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        count = 5
        while count:

            try:
                return f(*args, **kwargs)
            except (
                TransformerConnectionError,
                requests.exceptions.RequestException,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ProxyError,
                requests.exceptions.SSLError,
                requests.exceptions.BaseHTTPError,
            ) as e:
                print(colored('Training Exception: {}, will retry later'.format(e)))
                time.sleep(60)
                count -= 1

    return decorated


def time_out(interval, callback=None):
    def decorator(func):
        def handler(signum, frame):
            raise TimeoutError("Timeout Func: {} ".format(func.__name__))

        def wrapper(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(interval)  # interval秒后向进程发送SIGALRM信号
                result = func(*args, **kwargs)
                signal.alarm(0)  # 函数在规定时间执行完后关闭alarm闹钟
                return result
            except TimeoutError as e:
                if callback:
                    callback(e)
                else:
                    raise e

        return wrapper

    return decorator


def save_json(dic, save_path):
    if isinstance(dic, str):
        dic = eval(dic)
    with open(save_path, 'w', encoding='utf-8') as f:
        # f.write(str(dict))
        str_ = json.dumps(dic, ensure_ascii=False)
        f.write(str_)


def load_json(save_path):
    with open(save_path, 'r', encoding='utf-8') as f:
        data = f.readline().strip()
        print(type(data), data)
        dic = json.loads(data)
    return dic


def validate_pyabsa_version():
    try:
        response = requests.get("https://pypi.org/pypi/pyabsa/json", timeout=1)
    except requests.exceptions.RequestException:
        return
    if response.status_code == 200:
        data = response.json()
        versions = list(data["releases"].keys())
        versions.sort(key=parse_version, reverse=True)
        if __version__ not in versions:
            print(colored('You are using a DEPRECATED or TEST version of PyABSA. Consider update using pip install -U pyabsa!', 'red'))


def init_optimizer(optimizer):
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
        'adamw': torch.optim.AdamW,
        torch.optim.Adadelta: torch.optim.Adadelta,  # default lr=1.0
        torch.optim.Adagrad: torch.optim.Adagrad,  # default lr=0.01
        torch.optim.Adam: torch.optim.Adam,  # default lr=0.001
        torch.optim.Adamax: torch.optim.Adamax,  # default lr=0.002
        torch.optim.ASGD: torch.optim.ASGD,  # default lr=0.01
        torch.optim.RMSprop: torch.optim.RMSprop,  # default lr=0.01
        torch.optim.SGD: torch.optim.SGD,
        torch.optim.AdamW: torch.optim.AdamW,
    }
    if optimizer in optimizers:
        return optimizers[optimizer]
    elif hasattr(torch.optim, optimizer.__name__):
        return optimizer
    else:
        raise KeyError('Unsupported optimizer: {}. Please use string or the optimizer objects in torch.optim as your optimizer'.format(optimizer))
