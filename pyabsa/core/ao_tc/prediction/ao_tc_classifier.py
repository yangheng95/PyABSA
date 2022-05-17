# -*- coding: utf-8 -*-
# file: text_classifier.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.
import json
import os
import pickle
import random

import autocuda
import numpy
import torch
from findfile import find_file
from termcolor import colored
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, DebertaV2ForMaskedLM, RobertaForMaskedLM, BertForMaskedLM

from pyabsa.functional.dataset import detect_infer_dataset

from ..models import AOGloVeTCModelList, AOBERTTCModelList
from ..classic.__glove__.dataset_utils.data_utils_for_inference import AOGloVeTCDataset
from ..classic.__bert__.dataset_utils.data_utils_for_inference import AOBERTTCDataset, Tokenizer4Pretraining

from ..classic.__glove__.dataset_utils.data_utils_for_training import LABEL_PADDING, build_embedding_matrix, build_tokenizer

from pyabsa.utils.pyabsa_utils import print_args, TransformerConnectionError
from ...atepc.dataset_utils.atepc_utils import split_text


def get_mlm_and_tokenizer(text_classifier, config):
    if isinstance(text_classifier, AOTCTextClassifier):
        base_model = text_classifier.model.bert.base_model
    else:
        base_model = text_classifier.bert.base_model
    pretrained_config = AutoConfig.from_pretrained(config.pretrained_bert)
    if 'deberta-v3' in config.pretrained_bert:
        MLM = DebertaV2ForMaskedLM(pretrained_config).to(text_classifier.opt.device)
        MLM.deberta = base_model
    elif 'roberta' in config.pretrained_bert:
        MLM = RobertaForMaskedLM(pretrained_config).to(text_classifier.opt.device)
        MLM.roberta = base_model
    else:
        MLM = BertForMaskedLM(pretrained_config).to(text_classifier.opt.device)
        MLM.bert = base_model
    return MLM, AutoTokenizer.from_pretrained(config.pretrained_bert)


class AOTCTextClassifier:
    def __init__(self, model_arg=None, eval_batch_size=128):
        '''
            from_train_model: load inferring_tutorials model from trained model
        '''

        # load from a training
        if not isinstance(model_arg, str):
            print('Load text classifier from training')
            self.model = model_arg[0]
            self.opt = model_arg[1]
            self.tokenizer = model_arg[2]
        else:
            try:
                if 'fine-tuned' in model_arg:
                    raise ValueError('Do not support to directly load a fine-tuned model, please load a .state_dict or .model instead!')
                print('Load text classifier from', model_arg)
                state_dict_path = find_file(model_arg, key='.state_dict', exclude_key=['__MACOSX'])
                model_path = find_file(model_arg, key='.model', exclude_key=['__MACOSX'])
                tokenizer_path = find_file(model_arg, key='.tokenizer', exclude_key=['__MACOSX'])
                config_path = find_file(model_arg, key='.config', exclude_key=['__MACOSX'])

                print('config: {}'.format(config_path))
                print('state_dict: {}'.format(state_dict_path))
                print('model: {}'.format(model_path))
                print('tokenizer: {}'.format(tokenizer_path))

                with open(config_path, mode='rb') as f:
                    self.opt = pickle.load(f)

                if state_dict_path or model_path:
                    if hasattr(AOBERTTCModelList, self.opt.model.__name__):
                        if state_dict_path:
                            self.bert = AutoModel.from_pretrained(self.opt.pretrained_bert)
                            self.model = self.opt.model(self.bert, self.opt)
                            self.model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
                        elif model_path:
                            self.model = torch.load(model_path, map_location='cpu')

                        try:
                            self.tokenizer = Tokenizer4Pretraining(max_seq_len=self.opt.max_seq_len, opt=self.opt)
                        except ValueError:
                            if tokenizer_path:
                                with open(tokenizer_path, mode='rb') as f:
                                    self.tokenizer = pickle.load(f)
                            else:
                                raise TransformerConnectionError()
                    else:
                        tokenizer = build_tokenizer(
                            dataset_list=self.opt.dataset_file,
                            max_seq_len=self.opt.max_seq_len,
                            dat_fname='{0}_tokenizer.dat'.format(os.path.basename(self.opt.dataset_name)),
                            opt=self.opt
                        )
                        if model_path:
                            self.model = torch.load(model_path, map_location='cpu')
                        else:
                            embedding_matrix = build_embedding_matrix(
                                word2idx=tokenizer.word2idx,
                                embed_dim=self.opt.embed_dim,
                                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(self.opt.embed_dim), os.path.basename(self.opt.dataset_name)),
                                opt=self.opt
                            )
                            self.model = self.opt.model(embedding_matrix, self.opt).to(self.opt.device)
                            self.model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))

                        self.tokenizer = tokenizer

                print('Config used in Training:')
                print_args(self.opt, mode=1)

            except Exception as e:
                raise RuntimeError('Exception: {} Fail to load the model from {}! '.format(e, model_arg))

            if not hasattr(AOGloVeTCModelList, self.opt.model.__name__) \
                and not hasattr(AOBERTTCModelList, self.opt.model.__name__):
                raise KeyError('The checkpoint you are loading is not from classifier model.')

        if hasattr(AOBERTTCModelList, self.opt.model.__name__):
            self.dataset = AOBERTTCDataset(tokenizer=self.tokenizer, opt=self.opt)

        elif hasattr(AOGloVeTCModelList, self.opt.model.__name__):
            self.dataset = AOGloVeTCDataset(tokenizer=self.tokenizer, opt=self.opt)

        self.infer_dataloader = None
        self.opt.eval_batch_size = eval_batch_size

        if self.opt.seed is not None:
            random.seed(self.opt.seed)
            numpy.random.seed(self.opt.seed)
            torch.manual_seed(self.opt.seed)
            torch.cuda.manual_seed(self.opt.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.opt.initializer = self.opt.initializer

        try:
            self.MLM, self.MLM_tokenizer = get_mlm_and_tokenizer(self, self.opt)
        except Exception as e:
            self.MLM, self.MLM_tokenizer = None, None

    def to(self, device=None):
        self.opt.device = device
        self.model.to(device)

    def cpu(self):
        self.opt.device = 'cpu'
        self.model.to('cpu')

    def cuda(self, device='cuda:0'):
        self.opt.device = device
        self.model.to(device)

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
        for arg in vars(self.opt):
            if getattr(self.opt, arg) is not None:
                print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def batch_infer(self,
                    target_file=None,
                    print_result=True,
                    save_result=False,
                    clear_input_samples=True,
                    ignore_error=True):

        if clear_input_samples:
            self.clear_input_samples()

        save_path = os.path.join(os.getcwd(), 'ao_text_classification.result.json')

        target_file = detect_infer_dataset(target_file, task='ao_tc')
        if not target_file:
            raise FileNotFoundError('Can not find inference datasets!')

        self.dataset.prepare_infer_dataset(target_file, ignore_error=ignore_error)
        self.infer_dataloader = DataLoader(dataset=self.dataset, batch_size=self.opt.eval_batch_size, pin_memory=True, shuffle=False)
        return self._infer(save_path=save_path if save_result else None, print_result=print_result)

    def infer(self, text: str = None,
              print_result=True,
              clear_input_samples=True):

        if clear_input_samples:
            self.clear_input_samples()
        if text:
            self.dataset.prepare_infer_sample(text)
        else:
            raise RuntimeError('Please specify your datasets path!')
        self.infer_dataloader = DataLoader(dataset=self.dataset, batch_size=self.opt.eval_batch_size, shuffle=False)
        return self._infer(print_result=print_result)[0]

    def _infer(self, save_path=None, print_result=True):

        _params = filter(lambda p: p.requires_grad, self.model.parameters())

        correct = {True: 'Correct', False: 'Wrong'}
        results = []
        perplexity = 'N.A.'

        self.model.eval()
        with torch.no_grad():
            self.model.eval()
            n_sent_correct = 0
            n_sent_labeled = 0
            n_sent_total = 0

            n_advdet_correct = 0
            n_advdet_labeled = 0
            n_advdet_total = 0

            n_ooddet_correct = 0
            n_ooddet_labeled = 0
            n_ooddet_total = 0

        with torch.no_grad():
            self.model.eval()
            for _, sample in enumerate(self.infer_dataloader):
                inputs = [sample[col].to(self.opt.device) for col in self.opt.inputs_cols]

                sen_logits, advdet_logits, ooddet_logits = self.model(inputs)
                sent_probs, advdet_probs, ooddet_probs = torch.softmax(sen_logits, dim=-1), torch.softmax(advdet_logits, dim=-1), torch.softmax(ooddet_logits, dim=-1)

                for i, (sent_prob, advdet_prob, ooddet_prob) in enumerate(zip(sent_probs, advdet_probs, ooddet_probs)):
                    sent = int(sent_prob.argmax(axis=-1))
                    advdet = int(advdet_prob.argmax(axis=-1))
                    ooddet = int(ooddet_prob.argmax(axis=-1))
                    real_sent = int(sample['label'][i]) if int(sample['label'][i]) in self.opt.index_to_label else ''
                    real_adv = int(sample['advdet_label'][i]) if int(sample['advdet_label'][i]) in self.opt.index_to_adv_label else ''
                    real_ood = int(sample['ood_label'][i]) if int(sample['ood_label'][i]) in self.opt.index_to_ood_label else ''

                    if real_sent != -100:
                        n_sent_labeled += 1
                        if sent == real_sent:
                            n_sent_correct += 1

                    if real_adv and real_adv != -100:
                        n_advdet_labeled += 1
                        if real_adv == advdet:
                            n_advdet_correct += 1

                    if real_ood and real_ood != -100:
                        n_ooddet_labeled += 1
                        if real_ood == ooddet:
                            n_ooddet_correct += 1

                    text_raw = sample['text_raw'][i]

                    # if self.MLM:
                    #     ids = self.MLM_tokenizer(text_raw, return_tensors="pt")
                    #     ids['labels'] = ids['input_ids'].clone()
                    #     ids = ids.to(self.opt.device)
                    #     loss = self.MLM(**ids)['loss']
                    #     perplexity = float(torch.exp(loss / ids['input_ids'].size(1)))
                    # else:
                    #     perplexity = 'N.A.'

                    results.append({
                        'text': text_raw,
                        # 'sent_label': self.opt.index_to_label[sent],
                        'sent_label': '',
                        'sent_confidence': float(max(sent_prob)),
                        'ref_sent_label': self.opt.index_to_label[real_sent] if isinstance(real_sent, int) else real_sent,
                        'ref_sent_check': correct[sent == real_sent] if real_sent != -100 and isinstance(real_sent, int) else '',

                        'adv_label': self.opt.index_to_adv_label[advdet],
                        'adv_confidence': float(max(advdet_prob)),
                        'ref_adv_label': self.opt.index_to_adv_label[real_adv] if isinstance(real_adv, int) else real_adv,
                        'ref_adv_check': correct[advdet == real_adv] if real_adv != -100 and isinstance(real_adv, int) else '',

                        'ood_label': self.opt.index_to_ood_label[ooddet],
                        'ood_confidence': float(max(ooddet_prob)),
                        'ref_ood_label': self.opt.index_to_ood_label[real_ood] if isinstance(real_ood, int) else real_ood,
                        'ref_ood_check': correct[ooddet == real_ood] if real_ood != -100 and isinstance(real_ood, int) else '',

                        'perplexity': perplexity,
                    })

        try:
            if print_result:
                for result in results:
                    text_printing = result['text']
                    text_info = ''
                    # CLS
                    if result['ref_sent_label'] != '-100':
                        if result['sent_label'] == result['ref_sent_label']:
                            text_info += colored(' -> <CLS:{}(ref:{} confidence:{})>'.format(result['sent_label'], result['ref_sent_label'], result['sent_confidence']), 'green')
                        else:
                            text_info += colored(' -> <CLS:{}(ref:{} confidence:{})>'.format(result['sent_label'], result['ref_sent_label'], result['adv_confidence']), 'red')

                    # AdvDet
                    if result['adv_label'] != '-100':
                        if result['adv_label'] == result['ref_adv_label']:
                            text_info += colored(' -> <AdvDet:{}(ref:{})>'.format(result['adv_label'], result['ref_adv_check']), 'green')
                        else:
                            text_info += colored(' -> <AdvDet:{}(ref:{})>'.format(result['adv_label'], result['ref_adv_check']), 'red')

                    # OODDet
                    if result['ood_label'] != '-100':
                        if result['ood_label'] == result['ref_ood_label']:
                            text_info += colored(' -> <OODDet:{}(ref:{})>'.format(result['ood_label'], result['ref_ood_check']), 'green')
                        else:
                            text_info += colored(' -> <OODDet:{}(ref:{})>'.format(result['ood_label'], result['ref_ood_check']), 'red')

                    text_printing += text_info + colored('<perplexity:{}>'.format(result['perplexity']), 'yellow')
                    print(text_printing)
            if save_path:
                with open(save_path, 'w', encoding='utf8') as fout:
                    json.dump(results, fout, ensure_ascii=False)
                    # fout.write('Total samples:{}\n'.format(n_total))
                    # fout.write('Labeled samples:{}\n'.format(n_labeled))
                    # fout.write('Prediction Accuracy:{}%\n'.format(100 * n_correct / n_labeled)) if n_labeled else ''
                    print('inference result saved in: {}'.format(save_path))
        except Exception as e:
            print('Can not save result: {}, Exception: {}'.format(text_raw, e))

        if len(self.infer_dataloader) > 1:
            # print('Total samples:{}'.format(n_sent_total))
            # print('Labeled samples:{}'.format(n_sent_labeled))
            print('CLS Acc:{}%'.format(100 * n_sent_correct / n_sent_labeled if n_sent_labeled else ''))
            print('AdvDet Acc:{}%'.format(100 * n_advdet_correct / n_advdet_labeled if n_advdet_labeled else ''))
            print('OODDet Acc:{}%'.format(100 * n_ooddet_correct / n_ooddet_labeled if n_ooddet_labeled else ''))

        return results

    def clear_input_samples(self):
        self.dataset.all_data = []
