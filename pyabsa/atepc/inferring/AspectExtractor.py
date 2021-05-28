# -*- coding: utf-8 -*-
# file: AspectExtractor.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os
import random
import pickle
import string

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel

from ..dataset_utils.data_utils_for_inferring import (ATEPCProcessor,
                                                      convert_examples_to_features,
                                                      SENTIMENT_PADDING)
from ..models.lcf_atepc import LCF_ATEPC
from pyabsa.pyabsa_utils import find_target_file


class AspectExtractor:

    def __init__(self, model_arg=None):
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
        model_classes = {
            'lcf_atepc': LCF_ATEPC,
        }
        self.processor = ATEPCProcessor()
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list) + 1
        # load from a model path
        if not isinstance(model_arg, str):
            # self.model = model_arg[0]
            # self.args = model_arg[1]
            raise NotImplementedError('No implemented yet')

        else:
            print('Try to load trained model and config from', model_arg)
            state_dict_path = find_target_file(model_arg, 'state_dict')
            config_path = find_target_file(model_arg, 'config')
            self.args = pickle.load(open(config_path, 'rb'))
            self.args.bert_model = self.args.pretrained_bert_name
            bert_base_model = BertModel.from_pretrained(self.args.bert_model)
            bert_base_model.config.num_labels = self.num_labels
            self.model = model_classes[self.args.model_name](bert_base_model, args=self.args)
            self.model.load_state_dict(torch.load(state_dict_path))

        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_model, do_lower_case=True)

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        for arg in vars(self.args):
            print('>>> {0}: {1}'.format(arg, getattr(self.args, arg)))

        if self.args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                self.args.gradient_accumulation_steps))

        self.args.batch_size = self.args.batch_size // self.args.gradient_accumulation_steps

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.l2reg},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': self.args.l2reg}
        ]

        self.optimizer = optimizers[self.args.optimizer](optimizer_grouped_parameters, lr=self.args.learning_rate,
                                                         weight_decay=self.args.l2reg)

        self.eval_dataloader = None

    def to(self, device=None):
        self.args.device = device
        self.model.to(device)

    def cpu(self):
        self.args.device = 'cpu'
        self.model.to('cpu')

    def cuda(self, device='cuda:0'):
        self.args.device = device
        self.model.to(device)

    def extract_aspect(self, examples: list):
        for example in examples:
            extraction_res = self._extract(example)
            polarity_res = self._infer(extraction_res)
            return {'extraction_res': extraction_res, 'polarity_res': polarity_res}

    # Temporal code, pending optimization
    def _extract(self, example):

        self.eval_dataloader = None
        example = self.processor.get_examples_for_aspect_extraction(example)
        eval_features = convert_examples_to_features(example,
                                                     self.label_list,
                                                     self.args.max_seq_len,
                                                     self.tokenizer)
        all_spc_input_ids = torch.tensor([f.input_ids_spc for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_polarities = torch.tensor([f.polarities for f in eval_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        all_tokens = [f.tokens for f in eval_features]
        eval_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                  all_polarities, all_valid_ids, all_lmask_ids)
        # Run prediction for full data
        eval_sampler = RandomSampler(eval_data)
        self.eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

        # extract_aspects
        self.model.eval()
        label_map = {i: label for i, label in enumerate(self.label_list, 1)}
        for input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask in self.eval_dataloader:
            input_ids_spc = input_ids_spc.to(self.args.device)
            input_mask = input_mask.to(self.args.device)
            segment_ids = segment_ids.to(self.args.device)
            valid_ids = valid_ids.to(self.args.device)
            label_ids = label_ids.to(self.args.device)
            polarities = polarities.to(self.args.device)
            l_mask = l_mask.to(self.args.device)

            with torch.no_grad():
                ate_logits, apc_logits = self.model(input_ids_spc, segment_ids, input_mask,
                                                    valid_ids=valid_ids, polarities=polarities,
                                                    attention_mask_label=l_mask)

            ate_logits = torch.argmax(F.log_softmax(ate_logits, dim=2), dim=2)
            ate_logits = ate_logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label_ids[0]):
                if j == 0:
                    continue
                elif label_ids[0][j] == len(self.label_list):
                    break
                else:
                    temp_1.append(label_map.get(label_ids[0][j], 'O'))
                    temp_2.append(label_map.get(ate_logits[0][j], 'O'))

            print('Sentence with predicted labels:')
            ate_result = []
            polarity = []
            for t, l in zip(all_tokens[0], temp_2):
                ate_result.append('{}({})'.format(t, l))
                if 'ASP' in l:
                    polarity.append(-SENTIMENT_PADDING)
                else:
                    polarity.append(SENTIMENT_PADDING)

            print(' '.join(ate_result))

        res = [(all_tokens[0], temp_2, polarity)]
        return res

    def _infer(self, example):
        self.eval_dataloader = None
        example = self.processor.get_examples_for_sentiment_classification(example)
        eval_features = convert_examples_to_features(example,
                                                     self.label_list,
                                                     self.args.max_seq_len,
                                                     self.tokenizer)
        all_spc_input_ids = torch.tensor([f.input_ids_spc for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_polarities = torch.tensor([f.polarities for f in eval_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        all_tokens = [f.tokens for f in eval_features]
        eval_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                  all_polarities, all_valid_ids, all_lmask_ids)
        # Run prediction for full data
        eval_sampler = RandomSampler(eval_data)
        self.eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

        # extract_aspects
        self.model.eval()
        label_map = {i: label for i, label in enumerate(self.label_list, 1)}

        sentiments = {0: 'Negative', 1: "Neutral", 2: 'Positive', -999: ''}
        # Correct = {True: 'Correct', False: 'Wrong'}

        for input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask in self.eval_dataloader:
            input_ids_spc = input_ids_spc.to(self.args.device)
            input_mask = input_mask.to(self.args.device)
            segment_ids = segment_ids.to(self.args.device)
            valid_ids = valid_ids.to(self.args.device)
            label_ids = label_ids.to(self.args.device)
            polarities = polarities.to(self.args.device)
            l_mask = l_mask.to(self.args.device)
            result = {}
            with torch.no_grad():
                ate_logits, apc_logits = self.model(input_ids_spc, segment_ids, input_mask,
                                                    valid_ids=valid_ids, polarities=polarities,
                                                    attention_mask_label=l_mask)

            sent = int(torch.argmax(apc_logits, -1))

            result['text'] = ' '.join(all_tokens[0])
            result['sentiment'] = sentiments[sent]

            print(result)
            return result


# Assume the original polarity are labeled as 0: negative  2: positive
def convert_to_binary_polarity(examples):
    for i in range(len(examples)):
        polarities = []
        for polarity in examples[i].polarity:
            if polarity == 2:
                polarities.append(1)
            else:
                polarities.append(polarity)
        examples[i].polarity = polarities
