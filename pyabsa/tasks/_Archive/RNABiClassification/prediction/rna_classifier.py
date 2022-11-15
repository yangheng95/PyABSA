# -*- coding: utf-8 -*-
# file: rna_classifier.py
# author: yangheng <hy345@exeter.ac.uk>
# Copyright (C) 2020. All Rights Reserved.
import json
import os
import pickle

import numpy as np
import torch
import tqdm
from findfile import find_file, find_cwd_dir
from termcolor import colored
from torch.utils.data import DataLoader
from transformers import AutoModel
from sklearn import metrics

from pyabsa import TaskCodeOption, LabelPaddingOption
from pyabsa.framework.prediction_class.predictor_template import InferenceModel
from ..models import BERTRNACModelList, GloVeRNACModelList
from ..dataset_utils.data_utils_for_inference import GloVeRNACInferenceDataset
from ..dataset_utils.data_utils_for_inference import BERTRNACInferenceDataset
from pyabsa.utils.data_utils.dataset_manager import detect_infer_dataset
from pyabsa.utils.pyabsa_utils import get_device, print_args
from pyabsa.utils.text_utils.mlm import get_mlm_and_tokenizer
from pyabsa.framework.tokenizer_class.tokenizer_class import PretrainedTokenizer


class RNAClassifier(InferenceModel):
    task_code = TaskCodeOption.RNASequenceClassification

    def __init__(self, checkpoint=None, cal_perplexity=False, **kwargs):
        '''
            from_train_model: load inference model from trained model
        '''

        super().__init__(checkpoint, cal_perplexity, **kwargs)

        # load from a trainer
        if not isinstance(self.checkpoint, str):
            print('Load text classifier from trainer')
            self.model = self.checkpoint[0]
            self.config = self.checkpoint[1]
            self.tokenizer = self.checkpoint[2]
        else:
            try:
                if 'fine-tuned' in self.checkpoint:
                    raise ValueError(
                        'Do not support to directly load a fine-tuned model, please load a .state_dict or .model instead!')
                print('Load text classifier from', self.checkpoint)
                state_dict_path = find_file(self.checkpoint, key='.state_dict', exclude_key=['__MACOSX'])
                model_path = find_file(self.checkpoint, key='.model', exclude_key=['__MACOSX'])
                tokenizer_path = find_file(self.checkpoint, key='.tokenizer', exclude_key=['__MACOSX'])
                config_path = find_file(self.checkpoint, key='.config', exclude_key=['__MACOSX'])

                print('config: {}'.format(config_path))
                print('state_dict: {}'.format(state_dict_path))
                print('model: {}'.format(model_path))
                print('tokenizer: {}'.format(tokenizer_path))

                with open(config_path, mode='rb') as f:
                    self.config = pickle.load(f)
                    self.config.auto_device = kwargs.get('auto_device', True)
                    get_device(self.config)

                if state_dict_path or model_path:
                    if hasattr(BERTRNACModelList, self.config.model.__name__):
                        if state_dict_path:
                            if kwargs.get('offline', False):
                                self.bert = AutoModel.from_pretrained(
                                    find_cwd_dir(self.config.pretrained_bert.split('/')[-1]))
                            else:
                                self.bert = AutoModel.from_pretrained(self.config.pretrained_bert)
                            self.model = self.config.model(self.bert, self.config)
                            self.model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
                        elif model_path:
                            self.model = torch.load(model_path, map_location='cpu')

                        try:
                            self.tokenizer = PretrainedTokenizer(max_seq_len=self.config.max_seq_len, config=self.config,
                                                                 **kwargs)
                        except ValueError:
                            if tokenizer_path:
                                with open(tokenizer_path, mode='rb') as f:
                                    self.tokenizer = pickle.load(f)
                    else:
                        self.embedding_matrix = self.config.embedding_matrix
                        self.tokenizer = self.config.tokenizer
                        if model_path:
                            self.model = torch.load(model_path, map_location='cpu')
                        else:
                            self.model = self.config.model(self.embedding_matrix, self.config).to(self.config.device)
                            self.model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))

                if kwargs.get('verbose', False):
                    print('Config used in Training:')
                    print_args(self.config)

            except Exception as e:
                raise RuntimeError('Exception: {} Fail to load the model from {}! '.format(e, self.checkpoint))

            if not hasattr(GloVeRNACModelList, self.config.model.__name__) \
                and not hasattr(BERTRNACModelList, self.config.model.__name__):
                raise KeyError('The checkpoint_class you are loading is not from classifier model.')

        if hasattr(BERTRNACModelList, self.config.model.__name__):
            self.dataset = BERTRNACInferenceDataset(config=self.config, tokenizer=self.tokenizer)

        elif hasattr(GloVeRNACModelList, self.config.model.__name__):
            self.dataset = GloVeRNACInferenceDataset(config=self.config, tokenizer=self.tokenizer)

        self.infer_dataloader = None

        self.config.initializer = self.config.initializer

        if cal_perplexity:
            try:
                self.MLM, self.MLM_tokenizer = get_mlm_and_tokenizer(self.model, self.config)
            except Exception as e:
                self.MLM, self.MLM_tokenizer = None, None

        self.to(self.config.device)

    def to(self, device=None):
        self.config.device = device
        self.model.to(device)
        if hasattr(self, 'MLM'):
            self.MLM.to(self.config.device)

    def cpu(self):
        self.config.device = 'cpu'
        self.model.to('cpu')
        if hasattr(self, 'MLM'):
            self.MLM.to('cpu')

    def cuda(self, device='cuda:0'):
        self.config.device = device
        self.model.to(device)
        if hasattr(self, 'MLM'):
            self.MLM.to(device)

    def _log_write_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        for arg in vars(self.config):
            if getattr(self.config, arg) is not None:
                print('>>> {0}: {1}'.format(arg, getattr(self.config, arg)))

    def batch_predict(self,
                      target_file=None,
                      print_result=True,
                      save_result=False,
                      ignore_error=True,
                      **kwargs
                      ):

        self.config.eval_batch_size = kwargs.get('eval_batch_size', 32)

        save_path = os.path.join(os.getcwd(), 'rna_classification.result.json')

        target_file = detect_infer_dataset(target_file, task_code=TaskCodeOption.RNASequenceClassification)
        if not target_file:
            raise FileNotFoundError('Can not find inference datasets!')

        self.dataset.prepare_infer_dataset(target_file, ignore_error=ignore_error)
        self.infer_dataloader = DataLoader(dataset=self.dataset, batch_size=self.config.eval_batch_size, pin_memory=True,
                                           shuffle=False)
        return self._run_prediction(save_path=save_path if save_result else None, print_result=print_result)

    def predict(self, text: str = None,
                print_result=True,
                ignore_error=True,
                **kwargs
                ):

        self.config.eval_batch_size = kwargs.get('eval_batch_size', 32)

        if text:
            self.dataset.prepare_infer_sample(text, ignore_error=ignore_error)
        else:
            raise RuntimeError('Please specify your datasets path!')
        self.infer_dataloader = DataLoader(dataset=self.dataset, batch_size=self.config.eval_batch_size, shuffle=False)
        return self._run_prediction(print_result=print_result)[0]

    def _run_prediction(self, save_path=None, print_result=True):

        _params = filter(lambda p: p.requires_grad, self.model.parameters())

        correct = {True: 'Correct', False: 'Wrong'}
        results = []

        with torch.no_grad():
            self.model.eval()
            n_correct = 0
            n_labeled = 0
            n_total = 0
            decay_targets_all, decay_outputs_all = [], []
            seq_targets_all, seq_outputs_all = [], []

            if len(self.infer_dataloader.dataset) >= 100:
                it = tqdm.tqdm(self.infer_dataloader, postfix='run inference...')
            else:
                it = self.infer_dataloader
            for _, sample in enumerate(it):
                inputs = [sample[col].to(self.config.device) for col in self.config.inputs_cols if col != 'label']

                outputs = self.model(inputs)
                decay_logits, seq_logits = outputs
                t_probs = torch.softmax(decay_logits, dim=-1)
                s_probs = torch.softmax(seq_logits, dim=-1)

                decay_targets_all += [self.config.label1_to_index[x] for x in sample['label1']]
                decay_outputs_all += t_probs.argmax(dim=-1).tolist()
                seq_targets_all += [self.config.label2_to_index[x] for x in sample['label2']]
                seq_outputs_all += s_probs.argmax(dim=-1).tolist()

                for i, (i_probs1, i_probs2) in enumerate(zip(t_probs, s_probs)):

                    text_raw = sample['text_raw'][i]
                    ex_id = sample['ex_id'][i]
                    label1 = sample['label1'][i]
                    label2 = sample['label2'][i]

                    if self.cal_perplexity:
                        ids = self.MLM_tokenizer(text_raw, truncation=True, padding='max_length', max_length=self.config.max_seq_len, return_tensors='pt')
                        ids['labels'] = ids['input_ids'].clone()
                        ids = ids.to(self.config.device)
                        loss = self.MLM(**ids)['loss']
                        perplexity = float(torch.exp(loss / ids['input_ids'].size(1)))
                    else:
                        perplexity = 'N.A.'

                    results.append({
                        'ex_id': ex_id,
                        'text': text_raw,
                        'decay_label': self.config.index_to_label1[int(t_probs[i].argmax(axis=-1))],
                        'ref_decay_label': label1,
                        'seq_label': self.config.index_to_label2[int(s_probs[i].argmax(axis=-1))],
                        'ref_seq_label': label2,
                        'decay_confidence': round(float(max(i_probs1)), 4),
                        'seq_confidence': round(float(max(i_probs2)), 4),
                        'perplexity': perplexity
                    })
                    n_total += 1

        try:
            if print_result:
                for ex_id, result in enumerate(results):
                    text_printing = result['text'][:]
                    if result['ref_decay_label'] != LabelPaddingOption.LABEL_PADDING:
                        if result['decay_label'] == result['ref_decay_label']:
                            text_info = colored(
                                '#{}\t -> <Decay Label: {}(ref:{} decay_confidence:{})>\t'.format(result['ex_id'], result['decay_label'], result['ref_decay_label'],
                                                                                                  result['decay_confidence']), 'green')
                        else:
                            text_info = colored(
                                '#{}\t -> <Decay Label: {}(ref:{}) decay_confidence:{}>\t'.format(result['ex_id'], result['decay_label'], result['ref_decay_label'],
                                                                                                  result['decay_confidence']), 'red')
                    else:
                        text_info = '#{}\t -> Decay Label: {}\t'.format(result['ex_id'], result['decay_label'])

                    if result['ref_seq_label'] != LabelPaddingOption.LABEL_PADDING:
                        if result['seq_label'] == result['ref_seq_label']:
                            text_info += colored(
                                '\t -> <RNA Label: {}(ref:{} seq_confidence:{})>\t'.format(result['seq_label'], result['ref_seq_label'],
                                                                                           result['seq_confidence']), 'green')
                        else:
                            text_info += colored(
                                '\t -> <RNA Label: {}(ref:{}) seq_confidence:{}>\t'.format(result['seq_label'], result['ref_seq_label'],
                                                                                           result['seq_confidence']), 'red')
                    else:
                        text_info += '\t -> RNA Label: {}\t'.format(result['label'])
                    if self.cal_perplexity:
                        text_printing += colored(' --> <perplexity:{}>\t'.format(result['perplexity']), 'yellow')
                    text_printing = text_info + text_printing

                    print('Example :{}'.format(text_printing))
            if save_path:
                with open(save_path, 'w', encoding='utf8') as fout:
                    json.dump(str(results), fout, ensure_ascii=False)
                    print('inference result saved in: {}'.format(save_path))
        except Exception as e:
            print('Can not save result: {}, Exception: {}'.format(text_raw, e))

        if len(results) > 1:
            print('Total samples:{}'.format(n_total))
            print('Labeled samples:{}'.format(n_labeled))

            print('\n---------------------------- Classification Report ----------------------------\n')
            print(metrics.classification_report(decay_targets_all, np.argmax(decay_outputs_all, -1), digits=4,
                                                target_names=[str(self.config.index_to_label1[x]) for x in
                                                              self.config.index_to_label1]))
            print('\n-------------------------------------------------------------------------------\n')
            print(metrics.classification_report(seq_targets_all, np.argmax(seq_outputs_all, -1), digits=4,
                                                target_names=[str(self.config.index_to_label2[x]) for x in
                                                              self.config.index_to_label2]))
            print('\n---------------------------- Classification Report ----------------------------\n')

        return results

    def clear_input_samples(self):
        self.dataset.all_data = []
