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
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from findfile import find_file
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from transformers import BertTokenizer, AutoTokenizer, AutoModel
from transformers.models.bert.modeling_bert import BertModel

from pyabsa.functional.dataset import detect_infer_dataset, DatasetItem
from pyabsa.core.atepc.models import ATEPCModelList
from pyabsa.core.atepc.dataset_utils.atepc_utils import load_atepc_inference_datasets
from pyabsa.utils.pyabsa_utils import print_args, save_json, TransformerConnectionError
from ..dataset_utils.data_utils_for_inferring import (ATEPCProcessor,
                                                      convert_ate_examples_to_features,
                                                      convert_apc_examples_to_features,
                                                      SENTIMENT_PADDING)


class AspectExtractor:

    def __init__(self, model_arg=None, sentiment_map=None, eval_batch_size=128):
        # print('This is the aspect extractor aims to extract aspect and predict sentiment,'
        #       ' note that use_bert_spc is disabled while extracting aspects and classifying sentiment!')
        # optimizers = {
        #     'adadelta': torch.optim.Adadelta,  # default lr=1.0
        #     'adagrad': torch.optim.Adagrad,  # default lr=0.01
        #     'adam': torch.optim.Adam,  # default lr=0.001
        #     'adamax': torch.optim.Adamax,  # default lr=0.002
        #     'asgd': torch.optim.ASGD,  # default lr=0.01
        #     'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        #     'sgd': torch.optim.SGD,
        #     'adamw': torch.optim.AdamW
        # }
        # load from a training
        if not isinstance(model_arg, str):
            print('Load aspect extractor from training')
            self.model = model_arg[0]
            self.opt = model_arg[1]
            self.tokenizer = model_arg[2]
        else:
            if 'fine-tuned' in model_arg:
                raise ValueError('Do not support to directly load a fine-tuned model, please load a .state_dict or .model instead!')
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
                    try:
                        bert_base_model = AutoModel.from_pretrained(self.opt.pretrained_bert)
                    except ValueError:
                        raise TransformerConnectionError()

                    bert_base_model.config.num_labels = self.opt.num_labels
                    self.model = self.opt.model(bert_base_model, self.opt)
                    self.model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
                if model_path:
                    self.model = torch.load(model_path, map_location='cpu')
                    self.model.opt = self.opt
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.opt.pretrained_bert, do_lower_case='uncased' in self.opt.pretrained_bert)
                except ValueError:
                    if tokenizer_path:
                        self.tokenizer = pickle.load(open(tokenizer_path, mode='rb'))
                    else:
                        raise TransformerConnectionError()

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

        self.eval_dataloader = None
        self.opt.infer_batch_size = eval_batch_size
        self.sentiment_map = None
        self.set_sentiment_map(sentiment_map)

    def set_sentiment_map(self, sentiment_map):
        if sentiment_map and SENTIMENT_PADDING not in sentiment_map:
            print('Warning: set_sentiment_map is deprecated, please directly set labels within dataset.')
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

    def merge_result(self, sentence_res, results):
        """ merge ate sentence result and apc results, and restore to original sentence order
        Args:
            sentence_res ([tuple]): list of ate sentence results, which has (tokens, iobs)
            results ([dict]): list of apc results
        Returns:
            [dict]: merged extraction/polarity results for each input example
        """
        final_res = []
        if results['polarity_res'] is not None:
            merged_results = OrderedDict()
            pre_example_id = None
            # merge ate and apc results, assume they are same ordered
            for item1, item2 in zip(results['extraction_res'], results['polarity_res']):
                cur_example_id = item1[3]
                assert cur_example_id == item2['example_id'], "ate and apc results should be same ordered"
                if pre_example_id is None or cur_example_id != pre_example_id:
                    merged_results[cur_example_id] = {
                        'sentence': item2['sentence'],
                        'aspect': [item2['aspect']],
                        'position': [item2['pos_ids']],
                        'sentiment': [item2['sentiment']]
                    }
                else:
                    merged_results[cur_example_id]['aspect'].append(item2['aspect'])
                    merged_results[cur_example_id]['position'].append(item2['pos_ids'])
                    merged_results[cur_example_id]['sentiment'].append(item2['sentiment'])
                # remember example id
                pre_example_id = item1[3]
            for i, item in enumerate(sentence_res):
                asp_res = merged_results.get(i)
                final_res.append(
                    {
                        'sentence': ' '.join(item[0]),
                        'IOB': item[1],
                        'tokens': item[0],
                        'aspect': asp_res['aspect'] if asp_res else [],
                        'position': asp_res['position'] if asp_res else [],
                        'sentiment': asp_res['sentiment'] if asp_res else [],
                    }
                )
        else:
            for item in sentence_res:
                final_res.append(
                    {
                        'sentence': ' '.join(item[0]),
                        'IOB': item[1],
                        'tokens': item[0]
                    }
                )

        return final_res

    def extract_aspect(self, inference_source, save_result=True, print_result=True, pred_sentiment=True):
        results = {'extraction_res': OrderedDict(), 'polarity_res': OrderedDict()}

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

        elif isinstance(inference_source, list):
            pass

        else:
            raise ValueError('Please run inference using examples list or inference dataset path (list)!')

        if inference_source:
            extraction_res, sentence_res = self._extract(inference_source)
            results['extraction_res'] = extraction_res
            if pred_sentiment:
                results['polarity_res'] = self._infer(results['extraction_res'])
            results = self.merge_result(sentence_res, results)
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
        sentence_res = []  # extraction result by sentence
        extraction_res = []  # extraction result flatten by aspect

        self.infer_dataloader = None
        examples = self.processor.get_examples_for_aspect_extraction(examples)
        infer_features = convert_ate_examples_to_features(examples,
                                                          self.label_list,
                                                          self.opt.max_seq_len,
                                                          self.tokenizer,
                                                          self.opt)
        all_spc_input_ids = torch.tensor([f.input_ids_spc for f in infer_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in infer_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in infer_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in infer_features], dtype=torch.long)
        all_polarities = torch.tensor([f.polarity for f in infer_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in infer_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in infer_features], dtype=torch.long)

        all_tokens = [f.tokens for f in infer_features]
        infer_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                   all_polarities, all_valid_ids, all_lmask_ids)
        # Run prediction for full data
        infer_sampler = SequentialSampler(infer_data)
        if 'infer_batch_size' not in self.opt.args:
            self.opt.infer_batch_size = 128
        self.infer_dataloader = DataLoader(infer_data, sampler=infer_sampler, pin_memory=True, batch_size=self.opt.infer_batch_size)

        # extract_aspects
        self.model.eval()
        if 'index_to_IOB_label' not in self.opt.args:
            label_map = {i: label for i, label in enumerate(self.label_list, 1)}
        else:
            label_map = self.opt.index_to_IOB_label
        for i_batch, (input_ids_spc, input_mask, segment_ids, label_ids, polarity, valid_ids, l_mask) in enumerate(self.infer_dataloader):
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
                sentence_res.append((all_tokens[i + (self.opt.infer_batch_size * i_batch)], pred_iobs))
                for j, m in enumerate(label_ids[i]):
                    if j == 0:
                        continue
                    elif len(pred_iobs) == len(all_tokens[i + (self.opt.infer_batch_size * i_batch)]):
                        break
                    else:
                        pred_iobs.append(label_map.get(i_ate_logits[j], 'O'))

                ate_result = []
                polarity = []
                for t, l in zip(all_tokens[i + (self.opt.infer_batch_size * i_batch)], pred_iobs):
                    ate_result.append('{}({})'.format(t, l))
                    if 'ASP' in l:
                        polarity.append(-SENTIMENT_PADDING)
                    else:
                        polarity.append(SENTIMENT_PADDING)

                # POLARITY_PADDING = [SENTIMENT_PADDING] * len(polarity)
                # for iob_idx in range(len(polarity) - 1):
                #     example_id = i_batch * self.opt.infer_batch_size + i
                #     if pred_iobs[iob_idx].endswith('ASP') and not pred_iobs[iob_idx + 1].endswith('I-ASP'):
                #         _polarity = polarity[:iob_idx + 1] + POLARITY_PADDING[iob_idx + 1:]
                #         polarity = POLARITY_PADDING[:iob_idx + 1] + polarity[iob_idx + 1:]
                #         extraction_res.append((all_tokens[i + (self.opt.infer_batch_size * i_batch)], pred_iobs, _polarity, example_id))

                # Contributed by jkkl@github
                # Ref. https://github.com/yangheng95/PyABSA/issues/156#issuecomment-1077443060
                POLARITY_PADDING = [SENTIMENT_PADDING] * len(polarity)
                start_sentiment_index = -1
                end_sentiment_index = -1
                example_id = i_batch * self.opt.infer_batch_size + i
                for idx in range(len(polarity)):
                    if polarity[idx] != SENTIMENT_PADDING and (idx == 0 or polarity[idx] != polarity[idx - 1]):
                        start_sentiment_index = idx
                    elif polarity[idx] == SENTIMENT_PADDING and idx != 0 and polarity[idx - 1] != SENTIMENT_PADDING:
                        end_sentiment_index = idx
                        one_sentiment_labels = POLARITY_PADDING[:start_sentiment_index] + polarity[start_sentiment_index: end_sentiment_index] + POLARITY_PADDING[end_sentiment_index:]
                        extraction_res.append((all_tokens[i + (self.opt.infer_batch_size * i_batch)], pred_iobs, one_sentiment_labels, example_id))
                if start_sentiment_index > end_sentiment_index:
                    # 处理尾部情况
                    one_sentiment_labels = POLARITY_PADDING[:start_sentiment_index] + polarity[start_sentiment_index:]
                    extraction_res.append((all_tokens[i + (self.opt.infer_batch_size * i_batch)], pred_iobs, one_sentiment_labels, example_id))

        return extraction_res, sentence_res

    def _infer(self, examples):

        res = []  # sentiment classification result
        # ate example id map to apc example id
        example_id_map = dict([(apc_id, ex[3]) for apc_id, ex in enumerate(examples)])

        self.infer_dataloader = None
        examples = self.processor.get_examples_for_sentiment_classification(examples)
        infer_features = convert_apc_examples_to_features(examples,
                                                          self.label_list,
                                                          self.opt.max_seq_len,
                                                          self.tokenizer,
                                                          self.opt)
        all_spc_input_ids = torch.tensor([f.input_ids_spc for f in infer_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in infer_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in infer_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in infer_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in infer_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in infer_features], dtype=torch.long)
        lcf_cdm_vec = torch.tensor([f.lcf_cdm_vec for f in infer_features], dtype=torch.float32)
        lcf_cdw_vec = torch.tensor([f.lcf_cdw_vec for f in infer_features], dtype=torch.float32)
        all_tokens = [f.tokens for f in infer_features]
        all_aspects = [f.aspect for f in infer_features]
        all_positions = [f.positions for f in infer_features]
        infer_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                   all_valid_ids, all_lmask_ids, lcf_cdm_vec, lcf_cdw_vec)
        # Run prediction for full data
        self.opt.infer_batch_size = 128
        self.model.opt.use_bert_spc = True

        infer_sampler = SequentialSampler(infer_data)
        self.infer_dataloader = DataLoader(infer_data, sampler=infer_sampler, pin_memory=True, batch_size=self.opt.infer_batch_size)

        # extract_aspects
        self.model.eval()

        # Correct = {True: 'Correct', False: 'Wrong'}
        for i_batch, batch in enumerate(self.infer_dataloader):
            input_ids_spc, input_mask, segment_ids, label_ids, \
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
                    if 'index_to_label' in self.opt.args and int(i_apc_logits.argmax(axis=-1)) in self.opt.index_to_label:
                        sent = self.opt.index_to_label.get(int(i_apc_logits.argmax(axis=-1)))
                    else:
                        sent = int(torch.argmax(i_apc_logits, -1))
                    result = {}
                    apc_id = i_batch * self.opt.infer_batch_size + i
                    result['sentence'] = ' '.join(all_tokens[apc_id])
                    result['tokens'] = all_tokens[apc_id]
                    result['aspect'] = all_aspects[apc_id]
                    result['pos_ids'] = all_positions[apc_id]
                    result['sentiment'] = sent
                    result['example_id'] = example_id_map[apc_id]
                    res.append(result)

        return res
