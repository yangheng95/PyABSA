# -*- coding: utf-8 -*-
# file: sentiment_classifier.py
# author: yangheng <hy345@exeter.ac.uk>
# Copyright (C) 2020. All Rights Reserved.
import json
import os
import pickle
import random
from typing import Union

import numpy as np
import torch
import tqdm
from findfile import find_file, find_cwd_dir
from sklearn import metrics
from termcolor import colored
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from pyabsa import LabelPaddingOption, TaskCodeOption
from pyabsa.framework.prediction_class.predictor_template import InferenceModel
from ..models.__plm__ import BERTBaselineAPCModelList
from ..models.__classic__ import GloVeAPCModelList
from ..models.__lcf__ import APCModelList
from ..dataset_utils.__classic__.data_utils_for_inference import GloVeABSAInferenceDataset
from ..dataset_utils.__lcf__.data_utils_for_inference import ABSAInferenceDataset
from ..dataset_utils.__plm__.data_utils_for_inference import BERTABSAInferenceDataset
from ..instructor.ensembler import APCEnsembler
from pyabsa.utils.data_utils.dataset_manager import detect_infer_dataset
from pyabsa.utils.pyabsa_utils import set_device, print_args, fprint
from pyabsa.utils.text_utils.mlm import get_mlm_and_tokenizer


class SentimentClassifier(InferenceModel):
    def __init__(self, checkpoint=None, cal_perplexity=False, **kwargs):
        '''
            from_train_model: load inference model from trained model
        '''

        super().__init__(checkpoint, cal_perplexity, task_code=self.task_code, **kwargs)

        # load from a trainer
        if self.checkpoint and not isinstance(self.checkpoint, str):
            fprint('Load sentiment classifier from trainer')
            self.model = self.checkpoint[0]
            self.config = self.checkpoint[1]
            self.tokenizer = self.checkpoint[2]
        else:
            # load from a model path
            try:
                if 'fine-tuned' in self.checkpoint:
                    raise ValueError('Do not support to directly load a fine-tuned model, please load a .state_dict or .model instead!')
                fprint('Load sentiment classifier from', self.checkpoint)

                state_dict_path = find_file(self.checkpoint, '.state_dict', exclude_key=['__MACOSX'])
                model_path = find_file(self.checkpoint, '.model', exclude_key=['__MACOSX'])
                tokenizer_path = find_file(self.checkpoint, '.tokenizer', exclude_key=['__MACOSX'])
                config_path = find_file(self.checkpoint, '.config', exclude_key=['__MACOSX'])

                fprint('config: {}'.format(config_path))
                fprint('state_dict: {}'.format(state_dict_path))
                fprint('model: {}'.format(model_path))
                fprint('tokenizer: {}'.format(tokenizer_path))

                with open(config_path, mode='rb') as f:
                    self.config = pickle.load(f)
                    self.config.auto_device = kwargs.get('auto_device', True)
                    set_device(self.config, self.config.auto_device)

                if state_dict_path or model_path:
                    if state_dict_path:
                        self.model = APCEnsembler(self.config, load_dataset=False, **kwargs)
                        self.model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
                    elif model_path:
                        self.model = torch.load(model_path, map_location='cpu')
                    with open(tokenizer_path, mode='rb') as f:
                        if hasattr(APCModelList, self.config.model.__name__) or hasattr(BERTBaselineAPCModelList, self.config.model.__name__):
                            try:
                                if kwargs.get('offline', False):
                                    self.tokenizer = AutoTokenizer.from_pretrained(find_cwd_dir(self.config.pretrained_bert.split('/')[-1]),
                                                                                   do_lower_case='uncased' in self.config.pretrained_bert)
                                else:
                                    self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_bert, do_lower_case='uncased' in self.config.pretrained_bert)
                            except ValueError:
                                self.tokenizer = pickle.load(f)
                        elif hasattr(GloVeAPCModelList, self.config.model.__name__):
                            self.embedding_matrix = self.config.embedding_matrix
                            self.tokenizer = self.config.tokenizer

                if kwargs.get('verbose', False):
                    fprint('Config used in Training:')
                    print_args(self.config)

            except Exception as e:
                raise RuntimeError('Fail to load the model from {}! '
                                   'Please make sure the version of checkpoint and PyABSA are compatible.'
                                   ' Try to remove he checkpoint and download again'
                                   ' \nException: {} '.format(e, checkpoint))

        if isinstance(self.config.model, list):
            if hasattr(APCModelList, self.config.model[0].__name__):
                self.dataset = ABSAInferenceDataset(self.config, self.tokenizer)

            elif hasattr(BERTBaselineAPCModelList, self.config.model[0].__name__):
                self.dataset = BERTABSAInferenceDataset(self.config, self.tokenizer)

            elif hasattr(GloVeAPCModelList, self.config.model[0].__name__):
                self.dataset = GloVeABSAInferenceDataset(self.config, self.tokenizer)
            else:
                raise KeyError('The checkpoint you are loading is not from APC model.')
        else:
            if hasattr(APCModelList, self.config.model.__name__):
                self.dataset = ABSAInferenceDataset(self.config, self.tokenizer)

            elif hasattr(BERTBaselineAPCModelList, self.config.model.__name__):
                self.dataset = BERTABSAInferenceDataset(self.config, self.tokenizer)

            elif hasattr(GloVeAPCModelList, self.config.model.__name__):
                self.dataset = GloVeABSAInferenceDataset(self.config, self.tokenizer)
            else:
                raise KeyError('The checkpoint you are loading is not from APC model.')

        self.infer_dataloader = None

        self.config.initializer = self.config.initializer

        if self.cal_perplexity:
            try:
                self.MLM, self.MLM_tokenizer = get_mlm_and_tokenizer(self.model, self.config)
            except Exception as e:
                self.MLM, self.MLM_tokenizer = None, None

        self.to(self.config.device)

    def batch_infer(self,
                    target_file=None,
                    print_result=True,
                    save_result=False,
                    ignore_error=True,
                    **kwargs):
        """
        Old version of batch_predict, will be deprecated in the future
        """
        return self.batch_predict(target_file=target_file,
                                  print_result=print_result,
                                  save_result=save_result,
                                  ignore_error=ignore_error,
                                  **kwargs)

    def infer(self,
              text: str = None,
              print_result=True,
              ignore_error=True,
              **kwargs
              ):
        """
        Old version of predict, will be deprecated in the future
        """
        return self.predict(text=text,
                            print_result=print_result,
                            ignore_error=ignore_error,
                            **kwargs)

    def batch_predict(self,
                      target_file=None,
                      print_result=True,
                      save_result=False,
                      ignore_error=True,
                      **kwargs):
        """
        Predict the sentiment from a file of sentences.
        param: target_file: the file path of the sentences to be predicted.
        param: print_result: whether to print the result.
        param: save_result: whether to save the result.
        param: ignore_error: whether to ignore the error when predicting.
        param: kwargs: other parameters.
        """
        self.config.eval_batch_size = kwargs.get('eval_batch_size', 32)

        save_path = os.path.join(os.getcwd(), 'apc_inference.result.json')

        target_file = detect_infer_dataset(target_file, task_code=TaskCodeOption.Aspect_Polarity_Classification)
        if not target_file:
            raise FileNotFoundError('Can not find inference datasets!')

        self.dataset.prepare_infer_dataset(target_file, ignore_error=ignore_error)
        self.infer_dataloader = DataLoader(dataset=self.dataset, batch_size=self.config.eval_batch_size, pin_memory=True, shuffle=False)
        return self._run_prediction(save_path=save_path if save_result else None, print_result=print_result)

    def predict(self,
                text: Union[str, list] = None,
                print_result=True,
                ignore_error=True,
                **kwargs
                ):
        """
        Predict the sentiment from a sentence or a list of sentences.
        param: text: the sentence to be predicted.
        param: print_result: whether to print the result.
        param: ignore_error: whether to ignore the error when predicting.
        param: kwargs: other parameters.
        """
        self.config.eval_batch_size = kwargs.get('eval_batch_size', 32)
        self.infer_dataloader = DataLoader(dataset=self.dataset, batch_size=self.config.eval_batch_size, shuffle=False)
        if text:
            self.dataset.prepare_infer_sample(text, ignore_error=ignore_error)
        else:
            raise RuntimeError('Please specify your datasets path!')
        if isinstance(text, str):
            return self._run_prediction(print_result=print_result)[0]
        else:
            return self._run_prediction(print_result=print_result)

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

    def _run_prediction(self, save_path=None, print_result=True):

        _params = filter(lambda p: p.requires_grad, self.model.parameters())

        correct = {True: 'Correct', False: 'Wrong'}
        results = []

        with torch.no_grad():
            self.model.eval()
            n_correct = 0
            n_labeled = 0
            n_total = 0
            t_targets_all, t_outputs_all = None, None

            if len(self.infer_dataloader.dataset) >= 100:
                it = tqdm.tqdm(self.infer_dataloader, postfix='run inference...')
            else:
                it = self.infer_dataloader
            for _, sample in enumerate(it):
                inputs = {col: sample[col].to(self.config.device) for col in self.config.inputs_cols if col != 'polarity'}
                self.model.eval()
                outputs = self.model(inputs)
                sen_logits = outputs['logits']

                if t_targets_all is None:
                    t_targets_all = np.array([self.config.label_to_index[x] if x in self.config.label_to_index else
                                              LabelPaddingOption.SENTIMENT_PADDING for x in sample['polarity']])
                    t_outputs_all = np.array(sen_logits.cpu()).astype(np.float32)
                else:
                    t_targets_all = np.concatenate((t_targets_all, [self.config.label_to_index[x] if x in self.config.label_to_index else
                                                                    LabelPaddingOption.SENTIMENT_PADDING for x in sample['polarity']]), axis=0)
                    t_outputs_all = np.concatenate((t_outputs_all, np.array(sen_logits.cpu()).astype(np.float32)), axis=0)

                t_probs = torch.softmax(sen_logits, dim=-1)
                for i, i_probs in enumerate(t_probs):
                    sent = self.config.index_to_label[int(i_probs.argmax(axis=-1))]
                    real_sent = sample['polarity'][i]
                    if real_sent != LabelPaddingOption.SENTIMENT_PADDING:
                        n_labeled += 1
                    if sent == real_sent:
                        n_correct += 1

                    confidence = float(max(i_probs))

                    aspect = sample['aspect'][i]
                    text_raw = sample['text_raw'][i]

                    if self.cal_perplexity:
                        ids = self.MLM_tokenizer(text_raw, truncation=True, padding='max_length', max_length=self.config.max_seq_len, return_tensors='pt')
                        ids['labels'] = ids['input_ids'].clone()
                        ids = ids.to(self.config.device)
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
                        'ref_check': correct[sent == real_sent] if real_sent != str(LabelPaddingOption.LABEL_PADDING) else '',
                        'perplexity': perplexity
                    })
                    n_total += 1

        results = self.merge_results(results)
        try:
            if print_result:
                for ex_id, result in enumerate(results):
                    # flag = False  # only print error cases
                    # for ref_check in result['ref_check']:
                    #     if ref_check == 'Wrong':
                    #         flag = True
                    # if not flag:
                    #     continue
                    text_printing = result['text']
                    for i in range(len(result['aspect'])):
                        if result['ref_sentiment'][i] != LabelPaddingOption.SENTIMENT_PADDING:
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
                    if self.cal_perplexity:
                        text_printing += colored(' --> <perplexity:{}>'.format(result['perplexity']), 'yellow')
                    fprint('Example {}: {}'.format(ex_id, text_printing))
            if save_path:
                with open(save_path, 'w', encoding='utf8') as fout:
                    json.dump(str(results), fout, ensure_ascii=False)
                    fprint('inference result saved in: {}'.format(save_path))
        except Exception as e:
            fprint('Can not save result: {}, Exception: {}'.format(text_raw, e))

        if len(results) > 1:
            fprint('Total samples:{}'.format(n_total))
            fprint('Labeled samples:{}'.format(n_labeled))
            fprint('Prediction Accuracy:{}%'.format(100 * n_correct / n_labeled if n_labeled else 'N.A.'))

            try:
                fprint('\n---------------------------- Classification Report ----------------------------\n')
                fprint(metrics.classification_report(t_targets_all, np.argmax(t_outputs_all, -1), digits=4,
                                                     target_names=[self.config.index_to_label[x] for x in
                                                                   self.config.index_to_label]))
                fprint('\n---------------------------- Classification Report ----------------------------\n')
                fprint('\n---------------------------- Confusion Matrix ----------------------------\n')
                fprint(metrics.confusion_matrix(t_targets_all, np.argmax(t_outputs_all, -1)))
                fprint('\n---------------------------- Confusion Matrix ----------------------------\n')
            except Exception as e:
                fprint('Can not print classification report: {}'.format(e))

        return results

    def clear_input_samples(self):
        self.dataset.all_data = []
