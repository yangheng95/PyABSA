# -*- coding: utf-8 -*-
# file: aspect_extractor.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from findfile import find_file
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel

from pyabsa.utils.dataset_utils import detect_infer_dataset
from pyabsa.core.atepc.models import ATEPCModelList
from pyabsa.core.atepc.dataset_utils.atepc_utils import load_atepc_datasets
from pyabsa.utils.pyabsa_utils import print_args
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
                state_dict_path = find_file(model_arg, '.state_dict')
                model_path = find_file(model_arg, '.model')
                tokenizer_path = find_file(model_arg, '.tokenizer')
                config_path = find_file(model_arg, '.config')
                self.opt = pickle.load(open(config_path, 'rb'))
                if 'pretrained_bert_name' in self.opt.args:
                    self.opt.pretrained_bert = self.opt.pretrained_bert_name
                if state_dict_path:
                    bert_base_model = BertModel.from_pretrained(self.opt.pretrained_bert)
                    bert_base_model.config.num_labels = self.opt.num_labels
                    self.model = self.opt.model(bert_base_model, self.opt)
                    self.model.load_state_dict(torch.load(state_dict_path))
                if model_path:
                    self.model = torch.load(model_path)
                    self.model.opt = self.opt
                if tokenizer_path:
                    self.tokenizer = pickle.load(open(tokenizer_path, 'rb'))
                else:
                    self.tokenizer = BertTokenizer.from_pretrained(self.opt.pretrained_bert, do_lower_case=True)

                self.tokenizer.bos_token = self.tokenizer.bos_token if self.tokenizer.bos_token else '[CLS]'
                self.tokenizer.eos_token = self.tokenizer.eos_token if self.tokenizer.eos_token else '[SEP]'

            except Exception as e:
                raise KeyError('Fail to load the model from {}! the checkpoint is broken, '
                               'or maybe the checkpoint is not compatible with this version.'.format(model_arg),
                               )

            if not hasattr(ATEPCModelList, self.model.__class__.__name__):
                raise KeyError('The checkpoint you are loading is not from ATEPC model.')

        self.processor = ATEPCProcessor(self.tokenizer)
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list) + 1
        random.seed(self.opt.seed)
        np.random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)

        print('Config used in Training:')
        print_args(self.opt)

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

    def extract_aspect(self, inference_source, save_result=True, print_result=True, pred_sentiment=True):
        extraction_res = None
        polarity_res = None
        if isinstance(inference_source, str):
            inference_set = detect_infer_dataset(inference_source, task='apc')
            inference_source = load_atepc_datasets(inference_set)
            # print(examples)
            if not inference_source:
                extraction_res = self._extract(inference_source, print_result)
                if pred_sentiment:
                    polarity_res = self._infer(extraction_res, print_result)
                return {'extraction_res': extraction_res, 'polarity_res': polarity_res}
        if isinstance(inference_source, list):
            results = []
            save_path = os.path.join(os.getcwd(), 'atepc_inference.result.txt')
            fout = open(save_path, 'w', encoding='utf8') if save_result else None
            for example in inference_source:
                extraction_res = self._extract(example, print_result)
                if pred_sentiment:
                    polarity_res = self._infer(extraction_res, print_result)
                results.append({'extraction_res': extraction_res, 'polarity_res': polarity_res})
                if fout:
                    fout.write('extraction_res:{} polarity_res:{}\n'.format(extraction_res, polarity_res))
            print('The results of aspect term extraction have been saved in {}'.format(save_path))
            return results

    # Temporal code, pending optimization
    def _extract(self, example, print_result):

        res = []  # extraction result

        self.eval_dataloader = None
        example = self.processor.get_examples_for_aspect_extraction(example)
        eval_features = convert_ate_examples_to_features(example,
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
        self.eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

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
            pred_iobs = []
            for j, m in enumerate(label_ids[0]):
                if j == 0:
                    continue
                elif len(pred_iobs) == len(all_tokens[0]):
                    break
                else:
                    pred_iobs.append(label_map.get(ate_logits[0][j], 'O'))

            ate_result = []
            polarity = []
            for t, l in zip(all_tokens[0], pred_iobs):
                ate_result.append('{}({})'.format(t, l))
                if 'ASP' in l:
                    polarity.append(-SENTIMENT_PADDING)
                else:
                    polarity.append(SENTIMENT_PADDING)
            if print_result:
                print('Sentence with predicted labels:')
                print(' '.join(ate_result))
            asp_idx = 0
            asp_num = pred_iobs.count('B-ASP')
            IOB_PADDING = ['O'] * len(pred_iobs)
            POLARITY_PADDING = [SENTIMENT_PADDING] * len(pred_iobs)
            while asp_idx < asp_num:
                _pred_iobs = pred_iobs[:]
                _polarity = polarity[:]
                for iob_idx in range(len(_pred_iobs) - 1):
                    if pred_iobs[iob_idx].endswith('ASP') and not pred_iobs[iob_idx + 1].endswith('I-ASP'):
                        _pred_iobs = _pred_iobs[:iob_idx + 1] + IOB_PADDING[iob_idx + 1:]
                        pred_iobs = IOB_PADDING[:iob_idx + 1] + pred_iobs[iob_idx + 1:]
                        _polarity = _polarity[:iob_idx + 1] + POLARITY_PADDING[iob_idx + 1:]
                        polarity = POLARITY_PADDING[:iob_idx + 1] + polarity[iob_idx + 1:]
                        break

                res.append((all_tokens[0], _pred_iobs, _polarity))
                asp_idx += 1
        return res

    def _infer(self, example, print_result):

        res = []  # sentiment classification result

        self.eval_dataloader = None
        example = self.processor.get_examples_for_sentiment_classification(example)
        eval_features = convert_apc_examples_to_features(example,
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
        lcf_cdm_vec = torch.tensor([f.lcf_cdm_vec for f in eval_features], dtype=torch.float32)
        lcf_cdw_vec = torch.tensor([f.lcf_cdw_vec for f in eval_features], dtype=torch.float32)
        all_tokens = [f.tokens for f in eval_features]
        eval_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                  all_polarities, all_valid_ids, all_lmask_ids, lcf_cdm_vec, lcf_cdw_vec)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        self.eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

        # extract_aspects
        self.model.eval()
        if self.sentiment_map:
            sentiments = self.sentiment_map
        elif self.opt.polarities_dim == 3:
            sentiments = {0: 'Negative', 1: "Neutral", 2: 'Positive', -999: ''}
        else:
            sentiments = {p: str(p) for p in range(self.opt.polarities_dim + 1)}
            sentiments[-999] = ''

        # Correct = {True: 'Correct', False: 'Wrong'}
        for i, batch in enumerate(self.eval_dataloader):
            input_ids_spc, segment_ids, input_mask, label_ids, polarity, \
            valid_ids, l_mask, lcf_cdm_vec, lcf_cdw_vec = batch
            input_ids_spc = input_ids_spc.to(self.opt.device)
            input_mask = input_mask.to(self.opt.device)
            segment_ids = segment_ids.to(self.opt.device)
            valid_ids = valid_ids.to(self.opt.device)
            label_ids = label_ids.to(self.opt.device)
            polarity = polarity.to(self.opt.device)
            l_mask = l_mask.to(self.opt.device)
            lcf_cdm_vec = lcf_cdm_vec.to(self.opt.device)
            lcf_cdw_vec = lcf_cdw_vec.to(self.opt.device)
            result = {}
            with torch.no_grad():
                ate_logits, apc_logits = self.model(input_ids_spc,
                                                    token_type_ids=segment_ids,
                                                    attention_mask=input_mask,
                                                    labels=None,
                                                    polarity=polarity,
                                                    valid_ids=valid_ids,
                                                    attention_mask_label=l_mask,
                                                    lcf_cdm_vec=lcf_cdm_vec,
                                                    lcf_cdw_vec=lcf_cdw_vec)

                sent = int(torch.argmax(apc_logits, -1))
                aspect_idx = torch.where(polarity[0] > 0)
                aspect = []
                positions = []
                for idx in list(aspect_idx[0].cpu().numpy()):
                    positions.append(str(idx))
                    aspect.append(all_tokens[0][int(idx)])
                result['aspect'] = ' '.join(aspect)
                result['position'] = ','.join(positions)
                result['sentiment'] = sentiments[sent]
                if print_result:
                    print(result)
                res.append(result)
        return res
