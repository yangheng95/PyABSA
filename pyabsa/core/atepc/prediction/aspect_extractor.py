# -*- coding: utf-8 -*-
# file: aspect_extractor.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os
import pickle
import random

import json
import numpy as np
import torch
import torch.nn.functional as F
from findfile import find_file
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel

from pyabsa.functional.dataset import detect_infer_dataset, DatasetItem
from pyabsa.core.atepc.models import ATEPCModelList
from pyabsa.core.atepc.dataset_utils.atepc_utils import load_atepc_inference_datasets
from pyabsa.utils.pyabsa_utils import print_args, save_json
from ..dataset_utils.data_utils_for_inferring import (ATEPCProcessor,
                                                      convert_ate_examples_to_features,
                                                      convert_apc_examples_to_features,
                                                      SENTIMENT_PADDING)


class AspectExtractor:

    def __init__(self, model_arg=None, sentiment_map=None):
        print('This is the aspect extractor aims to extract aspect and predict sentiment,'
              ' note that use_bert_spc is disabled while extracting aspects and classifying sentiment!')
        optimizers = {
            'adadelta': torch.optim.Adadelta,  # default lr=1.0
            'adagrad': torch.optim.Adagrad,  # default lr=0.01
            'adam': torch.optim.Adam,  # default lr=0.001
            'adamax': torch.optim.Adamax,  # default lr=0.002
            'asgd': torch.optim.ASGD,  # default lr=0.01
            'rmsprop': torch.optim.RMSprop,  # default lr=0.01
            'sgd': torch.optim.SGD,
            'adamw': torch.optim.AdamW
        }
        # load from a training
        if not isinstance(model_arg, str):
            print('Load aspect extractor from training')
            self.model = model_arg[0]
            self.opt = model_arg[1]
            self.tokenizer = model_arg[2]
        else:
            # load from a model path
            print('Load aspect extractor from', model_arg)
            try:
                state_dict_path = find_file(model_arg, '.state_dict', exclude_key=['__MACOSX'])
                model_path = find_file(model_arg, '.model', exclude_key=['__MACOSX'])
                tokenizer_path = find_file(model_arg, '.tokenizer', exclude_key=['__MACOSX'])
                config_path = find_file(model_arg, '.config', exclude_key=['__MACOSX'])

                print('config: {}'.format(config_path))
                print('state_dict: {}'.format(state_dict_path))
                print('model: {}'.format(model_path))
                print('tokenizer: {}'.format(tokenizer_path))

                self.opt = pickle.load(open(config_path, mode='rb'))
                if 'pretrained_bert_name' in self.opt.args:
                    self.opt.pretrained_bert = self.opt.pretrained_bert_name
                if state_dict_path:
                    bert_base_model = BertModel.from_pretrained(self.opt.pretrained_bert)
                    bert_base_model.config.num_labels = self.opt.num_labels
                    self.model = self.opt.model(bert_base_model, self.opt)
                    self.model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
                if model_path:
                    self.model = torch.load(model_path, map_location='cpu')
                    self.model.opt = self.opt
                if tokenizer_path:
                    self.tokenizer = pickle.load(open(tokenizer_path, mode='rb'))
                else:
                    self.tokenizer = BertTokenizer.from_pretrained(self.opt.pretrained_bert, do_lower_case=True)

                self.tokenizer.bos_token = self.tokenizer.bos_token if self.tokenizer.bos_token else '[CLS]'
                self.tokenizer.eos_token = self.tokenizer.eos_token if self.tokenizer.eos_token else '[SEP]'

            except Exception as e:
                raise RuntimeError('Exception: {} Fail to load the model from {}! '.format(e, model_arg))

            if not hasattr(ATEPCModelList, self.model.__class__.__name__):
                raise KeyError('The checkpoint you are loading is not from ATEPC model.')

        self.processor = ATEPCProcessor(self.tokenizer)
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list) + 1
        random.seed(self.opt.seed)
        np.random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)

        print('Config used in Training:')
        print_args(self.opt, mode=1)

        if self.opt.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                self.opt.gradient_accumulation_steps))

        self.opt.batch_size = 1
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.opt.l2reg},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': self.opt.l2reg}
        ]

        self.optimizer = optimizers[self.opt.optimizer](optimizer_grouped_parameters,
                                                        lr=self.opt.learning_rate,
                                                        weight_decay=self.opt.l2reg)

        self.eval_dataloader = None
        self.sentiment_map = None
        self.set_sentiment_map(sentiment_map)

    def set_sentiment_map(self, sentiment_map):
        if sentiment_map and SENTIMENT_PADDING not in sentiment_map:
            sentiment_map[SENTIMENT_PADDING] = ''
        self.sentiment_map = sentiment_map

    def to(self, device=None):
        self.opt.device = device
        self.model.to(device)

    def cpu(self):
        self.opt.device = 'cpu'
        self.model.to('cpu')

    def cuda(self, device='cuda:0'):
        self.opt.device = device
        self.model.to(device)

    def merge_result(self, results):
        merged_results = []

        if results['polarity_res'] is not None:
            for item1, item2 in zip(results['extraction_res'], results['polarity_res']):
                if not merged_results or ' '.join(item1[0]) != merged_results[-1]['sentence']:
                    merged_results.append(
                        {'sentence': item2['sentence'],
                         'IOB': item1[1],
                         'tokens': item2['tokens'],
                         'aspect': [item2['aspect']],
                         'position': [item2['positions']],
                         'sentiment': [item2['sentiment']],
                         }
                    )
                else:
                    merged_results[-1]['aspect'].append(item2['aspect'])
                    merged_results[-1]['position'].append(item2['positions'])
                    merged_results[-1]['sentiment'].append(item2['sentiment'])

        else:
            for item in results['extraction_res']:
                if not merged_results or ' '.join(item[0]) != merged_results[-1]['sentence']:
                    merged_results.append(
                        {'sentence': ' '.join(item[0]),
                         'IOB': item[1],
                         'tokens': item[0],
                         }
                    )
        return merged_results

    def extract_aspect(self, inference_source, save_result=True, print_result=True, pred_sentiment=True):
        results = {'extraction_res': None, 'polarity_res': None}

        if isinstance(inference_source, DatasetItem):
            # using integrated inference dataset
            for d in inference_source:
                inference_set = detect_infer_dataset(d, task='apc')
                inference_source = load_atepc_inference_datasets(inference_set)

        elif isinstance(inference_source, str):  # for dataset path
            inference_source = DatasetItem(inference_source)
            # using custom inference dataset
            inference_set = detect_infer_dataset(inference_source, task='apc')
            inference_source = load_atepc_inference_datasets(inference_set)

        else:
            print('Please run inference using examples list or inference dataset path (list)!')

        if inference_source:
            results['extraction_res'] = self._extract(inference_source)
            if pred_sentiment:
                results['polarity_res'] = self._infer(results['extraction_res'])
            results = self.merge_result(results)
            if save_result:
                save_path = os.path.join(os.getcwd(), 'atepc_inference.result.json')
                print('The results of aspect term extraction have been saved in {}'.format(save_path))
                json.dump(json.JSONEncoder().encode({'results': results}), open(save_path, 'w'), ensure_ascii=False)
            if print_result:
                for r in results:
                    print(r)

            return results

    # Temporal code, pending optimization
    def _extract(self, examples):

        res = []  # extraction result

        self.eval_dataloader = None
        examples = self.processor.get_examples_for_aspect_extraction(examples)
        eval_features = convert_ate_examples_to_features(examples,
                                                         self.label_list,
                                                         self.opt.max_seq_len,
                                                         self.tokenizer,
                                                         self.opt)
        all_spc_input_ids = torch.tensor([f.input_ids_spc for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_polarities = torch.tensor([f.polarity for f in eval_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)

        all_tokens = [f.tokens for f in eval_features]
        eval_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                  all_polarities, all_valid_ids, all_lmask_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        self.eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=128)

        # extract_aspects
        self.model.eval()
        label_map = {i: label for i, label in enumerate(self.label_list, 1)}
        for input_ids_spc, input_mask, segment_ids, label_ids, polarity, valid_ids, l_mask in self.eval_dataloader:
            input_ids_spc = input_ids_spc.to(self.opt.device)
            input_mask = input_mask.to(self.opt.device)
            segment_ids = segment_ids.to(self.opt.device)
            valid_ids = valid_ids.to(self.opt.device)
            label_ids = label_ids.to(self.opt.device)
            polarity = polarity.to(self.opt.device)
            l_mask = l_mask.to(self.opt.device)
            with torch.no_grad():
                ate_logits, apc_logits = self.model(input_ids_spc,
                                                    segment_ids,
                                                    input_mask,
                                                    valid_ids=valid_ids,
                                                    polarity=polarity,
                                                    attention_mask_label=l_mask,
                                                    )

            ate_logits = torch.argmax(F.log_softmax(ate_logits, dim=2), dim=2)
            ate_logits = ate_logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
            for i, i_ate_logits in enumerate(ate_logits):
                pred_iobs = []

                for j, m in enumerate(label_ids[i]):
                    if j == 0:
                        continue
                    elif len(pred_iobs) == len(all_tokens[i]):
                        break
                    else:
                        pred_iobs.append(label_map.get(i_ate_logits[j], 'O'))

                ate_result = []
                polarity = []
                for t, l in zip(all_tokens[i], pred_iobs):
                    ate_result.append('{}({})'.format(t, l))
                    if 'ASP' in l:
                        polarity.append(-SENTIMENT_PADDING)
                    else:
                        polarity.append(SENTIMENT_PADDING)

                POLARITY_PADDING = [SENTIMENT_PADDING] * len(polarity)
                for iob_idx in range(len(polarity) - 1):
                    if pred_iobs[iob_idx].endswith('ASP') and not pred_iobs[iob_idx + 1].endswith('I-ASP'):
                        _polarity = polarity[:iob_idx + 1] + POLARITY_PADDING[iob_idx + 1:]
                        polarity = POLARITY_PADDING[:iob_idx + 1] + polarity[iob_idx + 1:]
                        res.append((all_tokens[i], pred_iobs, _polarity,i))
        
        return res

    def _infer(self, examples):

        res = []  # sentiment classification result

        self.eval_dataloader = None
        examples = self.processor.get_examples_for_sentiment_classification(examples)
        eval_features = convert_apc_examples_to_features(examples,
                                                         self.label_list,
                                                         self.opt.max_seq_len,
                                                         self.tokenizer,
                                                         self.opt)
        all_spc_input_ids = torch.tensor([f.input_ids_spc for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        lcf_cdm_vec = torch.tensor([f.lcf_cdm_vec for f in eval_features], dtype=torch.float32)
        lcf_cdw_vec = torch.tensor([f.lcf_cdw_vec for f in eval_features], dtype=torch.float32)
        all_tokens = [f.tokens for f in eval_features]
        all_aspects = [f.aspect for f in eval_features]
        all_positions = [f.positions for f in eval_features]
        eval_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                  all_valid_ids, all_lmask_ids, lcf_cdm_vec, lcf_cdw_vec)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        self.eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=128)

        # extract_aspects
        self.model.eval()
        if self.sentiment_map:
            sentiments = self.sentiment_map
        elif self.opt.polarities_dim == 3:
            sentiments = {0: 'Negative', 1: "Neutral", 2: 'Positive', -999: ''}
        else:
            sentiments = {p: str(p) for p in range(self.opt.polarities_dim + 1)}
            sentiments[-999] = ''
        # ate example id map to apc example id
        example_id_map = dict([(apc_id, ex[3]) for apc_id, ex in enumerate(examples)])
        # Correct = {True: 'Correct', False: 'Wrong'}
        for i_batch, batch in enumerate(self.eval_dataloader):
            input_ids_spc, segment_ids, input_mask, label_ids, \
            valid_ids, l_mask, lcf_cdm_vec, lcf_cdw_vec = batch
            input_ids_spc = input_ids_spc.to(self.opt.device)
            input_mask = input_mask.to(self.opt.device)
            segment_ids = segment_ids.to(self.opt.device)
            valid_ids = valid_ids.to(self.opt.device)
            label_ids = label_ids.to(self.opt.device)
            l_mask = l_mask.to(self.opt.device)
            lcf_cdm_vec = lcf_cdm_vec.to(self.opt.device)
            lcf_cdw_vec = lcf_cdw_vec.to(self.opt.device)
            with torch.no_grad():
                ate_logits, apc_logits = self.model(input_ids_spc,
                                                    token_type_ids=segment_ids,
                                                    attention_mask=input_mask,
                                                    labels=None,
                                                    valid_ids=valid_ids,
                                                    attention_mask_label=l_mask,
                                                    lcf_cdm_vec=lcf_cdm_vec,
                                                    lcf_cdw_vec=lcf_cdw_vec)
                for i, i_apc_logits in enumerate(apc_logits):
                    if 'origin_label_map' in self.opt.args:
                        sent = self.opt.origin_label_map[int(i_apc_logits.argmax(axis=-1))]
                    else:
                        sent = int(torch.argmax(i_apc_logits, -1))
                    result = {}

                    result['sentence'] = ' '.join(all_tokens[i])
                    result['tokens'] = all_tokens[i]
                    result['aspect'] = all_aspects[i]
                    result['positions'] = all_positions[i]
                    result['sentiment'] = sentiments[sent] if sent in sentiments else sent
                    result['example_id'] = example_id_map[i_batch]
                    res.append(result)

        return res
