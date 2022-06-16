# -*- coding: utf-8 -*-
# file: sentiment_classifier.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.
import json
import os
import pickle
import random

import numpy
import torch
import tqdm
from findfile import find_file
from termcolor import colored
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, DebertaV2ForMaskedLM, RobertaForMaskedLM, BertForMaskedLM

from pyabsa.core.apc.classic.__glove__.dataset_utils.data_utils_for_training import build_embedding_matrix, build_tokenizer
from pyabsa.core.apc.models.ensembler import APCEnsembler
from pyabsa.utils.pyabsa_utils import print_args, TransformerConnectionError
from pyabsa.functional.dataset import detect_infer_dataset
from pyabsa.core.apc.models import (APCModelList,
                                    GloVeAPCModelList,
                                    BERTBaselineAPCModelList
                                    )
from pyabsa.core.apc.classic.__bert__.dataset_utils.data_utils_for_inference import BERTBaselineABSADataset
from pyabsa.core.apc.classic.__glove__.dataset_utils.data_utils_for_inference import GloVeABSADataset
from pyabsa.core.apc.dataset_utils.apc_utils import LABEL_PADDING
from pyabsa.core.apc.dataset_utils.data_utils_for_inference import ABSADataset


def get_mlm_and_tokenizer(text_classifier, config):
    if isinstance(text_classifier, SentimentClassifier):
        base_model = text_classifier.model.bert.base_model
    else:
        base_model = text_classifier.bert.base_model
    pretrained_config = AutoConfig.from_pretrained(config.pretrained_bert)
    if 'deberta-v3' in config.pretrained_bert:
        MLM = DebertaV2ForMaskedLM(pretrained_config)
        MLM.deberta = base_model
    elif 'roberta' in config.pretrained_bert:
        MLM = RobertaForMaskedLM(pretrained_config)
        MLM.roberta = base_model
    else:
        MLM = BertForMaskedLM(pretrained_config)
        MLM.bert = base_model
    return MLM, AutoTokenizer.from_pretrained(config.pretrained_bert)


class SentimentClassifier:
    def __init__(self, model_arg=None, cal_perplexity=False, eval_batch_size=128):
        '''
            from_train_model: load inferring_tutorials model from trained model
        '''
        self.cal_perplexity = cal_perplexity
        # load from a training
        if not isinstance(model_arg, str):
            print('Load sentiment classifier from training')
            self.model = model_arg[0]
            self.opt = model_arg[1]
            self.tokenizer = model_arg[2]
        else:
            # load from a model path
            try:
                if 'fine-tuned' in model_arg:
                    raise ValueError('Do not support to directly load a fine-tuned model, please load a .state_dict or .model instead!')
                print('Load sentiment classifier from', model_arg)
                state_dict_path = find_file(model_arg, '.state_dict', exclude_key=['__MACOSX'])
                model_path = find_file(model_arg, '.model', exclude_key=['__MACOSX'])
                tokenizer_path = find_file(model_arg, '.tokenizer', exclude_key=['__MACOSX'])
                config_path = find_file(model_arg, '.config', exclude_key=['__MACOSX'])

                print('config: {}'.format(config_path))
                print('state_dict: {}'.format(state_dict_path))
                print('model: {}'.format(model_path))
                print('tokenizer: {}'.format(tokenizer_path))

                with open(config_path, mode='rb') as f:
                    self.opt = pickle.load(f)

                if state_dict_path or model_path:
                    if state_dict_path:
                        self.model = APCEnsembler(self.opt, load_dataset=False)
                        self.model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
                    elif model_path:
                        self.model = torch.load(model_path, map_location='cpu')
                    with open(tokenizer_path, mode='rb') as f:
                        if hasattr(APCModelList, self.opt.model.__name__):
                            try:
                                self.tokenizer = AutoTokenizer.from_pretrained(self.opt.pretrained_bert, do_lower_case='uncased' in self.opt.pretrained_bert)
                            except ValueError:
                                if tokenizer_path:
                                    self.tokenizer = pickle.load(f)
                                else:
                                    raise TransformerConnectionError()
                        elif hasattr(BERTBaselineAPCModelList, self.opt.model.__name__):
                            if tokenizer_path:
                                self.tokenizer = pickle.load(f)
                            else:
                                raise ValueError('No .tokenizer file found in checkpoint path!')
                        else:
                            tokenizer = build_tokenizer(
                                dataset_list=self.opt.dataset_file,
                                max_seq_len=self.opt.max_seq_len,
                                dat_fname='{0}_tokenizer.dat'.format(os.path.basename(self.opt.dataset_name)),
                                opt=self.opt
                            )

                            self.tokenizer = tokenizer

                print('Config used in Training:')
                print_args(self.opt, mode=1)

            except Exception as e:
                raise RuntimeError('Fail to load the model from {}! '
                                   'Please make sure the version of checkpoint and PyABSA are compatible.'
                                   ' Try to remove he checkpoint and download again'
                                   ' \nException: {} '.format(e, model_arg))

        if isinstance(self.opt.model, list):
            if hasattr(APCModelList, self.opt.model[0].__name__):
                self.dataset = ABSADataset(tokenizer=self.tokenizer, opt=self.opt)

            elif hasattr(BERTBaselineAPCModelList, self.opt.model[0].__name__):
                self.dataset = BERTBaselineABSADataset(tokenizer=self.tokenizer, opt=self.opt)

            elif hasattr(GloVeAPCModelList, self.opt.model[0].__name__):
                self.dataset = GloVeABSADataset(tokenizer=self.tokenizer, opt=self.opt)
            else:
                raise KeyError('The checkpoint you are loading is not from APC model.')
        else:
            if hasattr(APCModelList, self.opt.model.__name__):
                self.dataset = ABSADataset(tokenizer=self.tokenizer, opt=self.opt)

            elif hasattr(BERTBaselineAPCModelList, self.opt.model.__name__):
                self.dataset = BERTBaselineABSADataset(tokenizer=self.tokenizer, opt=self.opt)

            elif hasattr(GloVeAPCModelList, self.opt.model.__name__):
                self.dataset = GloVeABSADataset(tokenizer=self.tokenizer, opt=self.opt)
            else:
                raise KeyError('The checkpoint you are loading is not from APC model.')

        self.infer_dataloader = None

        if self.opt.seed is not None:
            random.seed(self.opt.seed)
            numpy.random.seed(self.opt.seed)
            torch.manual_seed(self.opt.seed)
            torch.cuda.manual_seed(self.opt.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.opt.initializer = self.opt.initializer
        self.opt.eval_batch_size = eval_batch_size

        self.sentiment_map = None

        if self.cal_perplexity:
            try:
                self.MLM, self.MLM_tokenizer = get_mlm_and_tokenizer(self, self.opt)
            except Exception as e:
                self.MLM, self.MLM_tokenizer = None, None

    def set_sentiment_map(self, sentiment_map):
        if sentiment_map:
            print(colored('Warning: set_sentiment_map() is deprecated, please directly set labels within dataset.', 'red'))
            sentiment_map[LABEL_PADDING] = ''
        self.sentiment_map = sentiment_map

    def to(self, device=None):
        self.opt.device = device
        self.model.to(device)
        if hasattr(self, 'MLM'):
            self.MLM.to(self.opt.device)

    def cpu(self):
        self.opt.device = 'cpu'
        self.model.to('cpu')
        if hasattr(self, 'MLM'):
            self.MLM.to('cpu')

    def cuda(self, device='cuda:0'):
        self.opt.device = device
        self.model.to(device)
        if hasattr(self, 'MLM'):
            self.MLM.to(device)

    def batch_infer(self,
                    target_file=None,
                    print_result=True,
                    save_result=False,
                    ignore_error=True,
                    clear_input_samples=True):

        if clear_input_samples:
            self.clear_input_samples()

        save_path = os.path.join(os.getcwd(), 'apc_inference.result.json')

        target_file = detect_infer_dataset(target_file, task='apc')
        if not target_file:
            raise FileNotFoundError('Can not find inference datasets!')

        self.dataset.prepare_infer_dataset(target_file, ignore_error=ignore_error)
        self.infer_dataloader = DataLoader(dataset=self.dataset, batch_size=self.opt.eval_batch_size, pin_memory=True, shuffle=False)
        return self._infer(save_path=save_path if save_result else None, print_result=print_result)

    def infer(self, text: str = None,
              print_result=True,
              ignore_error=True,
              clear_input_samples=True):
        if text.count('[ASP]') < 2:
            text = '[ASP]ERROR, Please WRAP the target aspects![ASP]' + text
        if clear_input_samples:
            self.clear_input_samples()
        if text:
            self.dataset.prepare_infer_sample(text, ignore_error=ignore_error)
        else:
            raise RuntimeError('Please specify your datasets path!')
        self.infer_dataloader = DataLoader(dataset=self.dataset, batch_size=self.opt.eval_batch_size, shuffle=False)
        return self._infer(print_result=print_result)[0]

    def merge_results(self, results):
        """ merge APC results have the same input text
        """
        final_res = []
        for result in results:

            if final_res and "".join(final_res[-1]['text'].split()) == "".join(result['text'].split()):
                final_res[-1]['aspect'].append(result['aspect'])
                final_res[-1]['sentiment'].append(result['sentiment'])
                final_res[-1]['confidence'].append(result['confidence'])
                final_res[-1]['probs'].append(result['probs'])
                final_res[-1]['ref_sentiment'].append(result['ref_sentiment'])
                final_res[-1]['ref_check'].append(result['ref_check'])
                final_res[-1]['perplexity'] = result['perplexity']
            else:
                final_res.append(
                    {
                        'text': result['text'].replace('  ', ' '),
                        'aspect': [result['aspect']],
                        'sentiment': [result['sentiment']],
                        'confidence': [result['confidence']],
                        'probs': [result['probs']],
                        'ref_sentiment': [result['ref_sentiment']],
                        'ref_check': [result['ref_check']],
                        'perplexity': result['perplexity']
                    }
                )

        return final_res

    def _infer(self, save_path=None, print_result=True):

        _params = filter(lambda p: p.requires_grad, self.model.parameters())

        correct = {True: 'Correct', False: 'Wrong'}
        results = []

        with torch.no_grad():
            self.model.eval()
            n_correct = 0
            n_labeled = 0
            n_total = 0
            if len(self.infer_dataloader.dataset) >= 100:
                it = tqdm.tqdm(self.infer_dataloader, postfix='inferring...')
            else:
                it = self.infer_dataloader
            for _, sample in enumerate(it):
                inputs = {col: sample[col].to(self.opt.device) for col in self.opt.inputs_cols if col != 'polarity'}
                self.model.eval()
                outputs = self.model(inputs)
                sen_logits = outputs['logits']
                t_probs = torch.softmax(sen_logits, dim=-1)
                for i, i_probs in enumerate(t_probs):
                    if 'index_to_label' in self.opt.args and int(i_probs.argmax(axis=-1)) in self.opt.index_to_label:
                        sent = self.opt.index_to_label[int(i_probs.argmax(axis=-1))]
                        real_sent = sample['polarity'][i] if isinstance(sample['polarity'][i], str) else self.opt.index_to_label.get(int(sample['polarity'][i]), 'N.A.')
                        if real_sent != -999 and real_sent != '-999':
                            n_labeled += 1
                        if sent == real_sent:
                            n_correct += 1
                    else:  # for the former versions before 1.2.0
                        sent = int(i_probs.argmax(axis=-1))
                        real_sent = int(sample['polarity'][i])

                    confidence = float(max(i_probs))

                    aspect = sample['aspect'][i]
                    text_raw = sample['text_raw'][i]

                    if self.cal_perplexity:
                        ids = self.MLM_tokenizer(text_raw, return_tensors="pt")
                        ids['labels'] = ids['input_ids'].clone()
                        ids = ids.to(self.opt.device)
                        loss = self.MLM(**ids)['loss']
                        perplexity = float(torch.exp(loss / ids['input_ids'].size(1)))
                    else:
                        perplexity = 'N.A.'

                    results.append({
                        'text': text_raw,
                        'aspect': aspect,
                        'sentiment': sent,
                        'confidence': confidence,
                        'probs': i_probs.cpu().numpy(),
                        'ref_sentiment': real_sent,
                        'ref_check': correct[sent == real_sent] if real_sent != '-999' else '',
                        'perplexity': perplexity
                    })
                    n_total += 1

        results = self.merge_results(results)
        try:
            if print_result:
                for result in results:
                    # flag = False  # only print error cases
                    # for ref_check in result['ref_check']:
                    #     if ref_check == 'Wrong':
                    #         flag = True
                    # if not flag:
                    #     continue
                    text_printing = result['text']
                    for i in range(len(result['aspect'])):
                        if result['ref_sentiment'][i] != -999 and result['ref_sentiment'][i] != '-999':
                            if result['sentiment'][i] == result['ref_sentiment'][i]:
                                aspect_info = colored('<{}:{}(confidence:{}, ref:{})>'.format(
                                    result['aspect'][i],
                                    result['sentiment'][i],
                                    round(result['confidence'][i], 3),
                                    result['ref_sentiment'][i]),
                                    'green')
                            else:
                                aspect_info = colored('<{}:{}(confidence:{}, ref:{})>'.format(
                                    result['aspect'][i],
                                    result['sentiment'][i],
                                    round(result['confidence'][i], 3),
                                    result['ref_sentiment'][i]),
                                    'red')

                        else:
                            aspect_info = '<{}:{}(confidence:{})>'.format(result['aspect'][i],
                                                                          result['sentiment'][i],
                                                                          round(result['confidence'][i], 3)
                                                                          )
                        text_printing = text_printing.replace(result['aspect'][i], aspect_info)
                    text_printing += colored('<perplexity:{}>'.format(result['perplexity']), 'yellow')
                    print(text_printing)
            if save_path:
                with open(save_path, 'w', encoding='utf8') as fout:
                    json.dump(str(results), fout, ensure_ascii=False)
                    print('inference result saved in: {}'.format(save_path))
        except Exception as e:
            print('Can not save result: {}, Exception: {}'.format(text_raw, e))

        if len(results) > 1:
            print('Total samples:{}'.format(n_total))
            print('Labeled samples:{}'.format(n_labeled))
            print('Prediction Accuracy:{}%'.format(100 * n_correct / n_labeled if n_labeled else 'N.A.'))
        return results

    def clear_input_samples(self):
        self.dataset.all_data = []
