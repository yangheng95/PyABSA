# -*- coding: utf-8 -*-
# file: text_classifier.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.
import json
import os
import pickle
import random
import time

import numpy
import numpy as np
import torch
import tqdm
from autocuda import auto_cuda
from findfile import find_file
from termcolor import colored

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, DebertaV2ForMaskedLM, RobertaForMaskedLM, BertForMaskedLM

from ....functional.dataset import detect_infer_dataset

from ..models import GloVeTADModelList, BERTTADModelList
from ..classic.__glove__.dataset_utils.data_utils_for_inference import GloVeTADDataset
from ..classic.__bert__.dataset_utils.data_utils_for_inference import BERTTADDataset, Tokenizer4Pretraining

from ..classic.__glove__.dataset_utils.data_utils_for_training import build_embedding_matrix, build_tokenizer

from ....utils.pyabsa_utils import print_args, TransformerConnectionError, get_device


def init_attacker(tad_classifier, defense):
    try:
        from textattack import Attacker
        from textattack.attack_recipes import BAEGarg2019, PWWSRen2019, TextFoolerJin2019, PSOZang2020, IGAWang2019, GeneticAlgorithmAlzantot2018, DeepWordBugGao2018
        from textattack.datasets import Dataset
        from textattack.models.wrappers import HuggingFaceModelWrapper

        class PyABSAModelWrapper(HuggingFaceModelWrapper):
            def __init__(self, model):
                self.model = model  # pipeline = pipeline

            def __call__(self, text_inputs, **kwargs):
                outputs = []
                for text_input in text_inputs:
                    raw_outputs = self.model.infer(text_input, print_result=False, **kwargs)
                    outputs.append(raw_outputs['probs'])
                return outputs

        class SentAttacker:

            def __init__(self, model, recipe_class=BAEGarg2019):
                model = model
                model_wrapper = PyABSAModelWrapper(model)

                recipe = recipe_class.build(model_wrapper)

                _dataset = [('', 0)]
                _dataset = Dataset(_dataset)

                self.attacker = Attacker(recipe, _dataset)

        attackers = {
            'bae': BAEGarg2019,
            'pwws': PWWSRen2019,
            'textfooler': TextFoolerJin2019,
            'pso': PSOZang2020,
            'iga': IGAWang2019,
            'ga': GeneticAlgorithmAlzantot2018,
            'wordbugger': DeepWordBugGao2018,
        }
        return SentAttacker(tad_classifier, attackers[defense])
    except Exception as e:
        print('If you need to evaluate text adversarial attack, please make sure you have installed:\n',
              colored('[1] pip install git+https://github.com/yangheng95/TextAttack\n', 'red'), 'and \n',
              colored('[2] pip install tensorflow_text \n', 'red'))
        print('Original error:', e)


def get_mlm_and_tokenizer(text_classifier, config):
    if isinstance(text_classifier, TADTextClassifier):
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


class TADTextClassifier:
    def __init__(self, model_arg=None, cal_perplexity=False, **kwargs):
        '''
            from_train_model: load inferring_tutorials model from trained model
        '''
        self.cal_perplexity = cal_perplexity
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
                    self.opt.device = get_device(kwargs.pop('auto_device', True))[0]

                if state_dict_path or model_path:
                    if hasattr(BERTTADModelList, self.opt.model.__name__):
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

            if not hasattr(GloVeTADModelList, self.opt.model.__name__) \
                and not hasattr(BERTTADModelList, self.opt.model.__name__):
                raise KeyError('The checkpoint you are loading is not from classifier model.')

        self.infer_dataloader = None
        self.opt.eval_batch_size = kwargs.pop('eval_batch_size', 128)

        if self.opt.seed is not None:
            random.seed(self.opt.seed)
            numpy.random.seed(self.opt.seed)
            torch.manual_seed(self.opt.seed)
            torch.cuda.manual_seed(self.opt.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.opt.initializer = self.opt.initializer

        if self.cal_perplexity:
            try:
                self.MLM, self.MLM_tokenizer = get_mlm_and_tokenizer(self, self.opt)
            except Exception as e:
                self.MLM, self.MLM_tokenizer = None, None

        self.to(self.opt.device)

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
                    ignore_error=True,
                    defense: str = None
                    ):

        # if clear_input_samples:
        #     self.clear_input_samples()

        save_path = os.path.join(os.getcwd(), 'tad_text_classification.result.json')

        target_file = detect_infer_dataset(target_file, task='text_defense')
        if not target_file:
            raise FileNotFoundError('Can not find inference datasets!')

        if hasattr(BERTTADModelList, self.opt.model.__name__):
            dataset = BERTTADDataset(tokenizer=self.tokenizer, opt=self.opt)

        else:
            dataset = GloVeTADDataset(tokenizer=self.tokenizer, opt=self.opt)

        dataset.prepare_infer_dataset(target_file, ignore_error=ignore_error)
        self.infer_dataloader = DataLoader(dataset=dataset, batch_size=self.opt.eval_batch_size, pin_memory=True, shuffle=False)
        return self._infer(save_path=save_path if save_result else None, print_result=print_result, defense=defense)

    def infer(self,
              text: str = None,
              print_result=True,
              ignore_error=True,
              defense: str = None
              ):

        if hasattr(BERTTADModelList, self.opt.model.__name__):
            dataset = BERTTADDataset(tokenizer=self.tokenizer, opt=self.opt)

        else:
            dataset = GloVeTADDataset(tokenizer=self.tokenizer, opt=self.opt)

        if text:
            dataset.prepare_infer_sample(text, ignore_error=ignore_error)
        else:
            raise RuntimeError('Please specify your datasets path!')
        self.infer_dataloader = DataLoader(dataset=dataset, batch_size=self.opt.eval_batch_size, shuffle=False)
        return self._infer(print_result=print_result, defense=defense)[0]

    def _infer(self, save_path=None, print_result=True, defense=None):

        _params = filter(lambda p: p.requires_grad, self.model.parameters())

        correct = {True: 'Correct', False: 'Wrong'}
        results = []

        with torch.no_grad():
            self.model.eval()
            n_correct = 0
            n_labeled = 0

            n_advdet_correct = 0
            n_advdet_labeled = 0
            if len(self.infer_dataloader.dataset) >= 100:
                it = tqdm.tqdm(self.infer_dataloader, postfix='inferring...')
            else:
                it = self.infer_dataloader
            for _, sample in enumerate(it):
                inputs = [sample[col].to(self.opt.device) for col in self.opt.inputs_cols]

                logits, advdet_logits, adv_tr_logits = self.model(inputs)
                probs, advdet_probs, adv_tr_probs = torch.softmax(logits, dim=-1), torch.softmax(advdet_logits, dim=-1), torch.softmax(adv_tr_logits, dim=-1)

                for i, (prob, advdet_prob, adv_tr_prob) in enumerate(zip(probs, advdet_probs, adv_tr_probs)):
                    text_raw = sample['text_raw'][i]

                    pred_label = int(prob.argmax(axis=-1))
                    pred_is_adv_label = int(advdet_prob.argmax(axis=-1))
                    pred_adv_tr_label = int(adv_tr_prob.argmax(axis=-1))
                    ref_label = int(sample['label'][i]) if int(sample['label'][i]) in self.opt.index_to_label else ''
                    ref_is_adv_label = int(sample['is_adv'][i]) if int(sample['is_adv'][i]) in self.opt.index_to_is_adv else ''
                    ref_adv_tr_label = int(sample['adv_train_label'][i]) if int(sample['adv_train_label'][i]) in self.opt.index_to_adv_train_label else ''

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

                        'label': self.opt.index_to_label[pred_label],
                        'probs': prob.cpu().numpy(),
                        'confidence': float(max(prob)),
                        'ref_label': self.opt.index_to_label[ref_label] if isinstance(ref_label, int) else ref_label,
                        'ref_label_check': correct[pred_label == ref_label] if ref_label != -100 else '',
                        'is_fixed': False,

                        'is_adv_label': self.opt.index_to_is_adv[pred_is_adv_label],
                        'is_adv_probs': advdet_prob.cpu().numpy(),
                        'is_adv_confidence': float(max(advdet_prob)),
                        'pred_adv_tr_label': pred_adv_tr_label,
                        'ref_adv_tr_label': ref_adv_tr_label,
                        'ref_is_adv_label': self.opt.index_to_is_adv[ref_is_adv_label] if isinstance(ref_is_adv_label, int) else ref_is_adv_label,
                        'ref_is_adv_check': correct[pred_is_adv_label == ref_is_adv_label] if ref_is_adv_label != -100 and isinstance(ref_is_adv_label, int) else '',

                        'perplexity': perplexity,
                    })

                    if defense:
                        try:
                            if not hasattr(self, 'sent_attacker'):
                                self.sent_attacker = init_attacker(self, defense)
                            if results[-1]['is_adv_label'] == '1':
                                res = self.sent_attacker.attacker.simple_attack(text_raw, int(results[-1]['label']))
                                new_infer_res = self.infer(res.perturbed_result.attacked_text.text, print_result=False)
                                results[-1]['perturbed_label'] = results[-1]['label']
                                results[-1]['label'] = new_infer_res['label']
                                results[-1]['probs'] = new_infer_res['probs']
                                results[-1]['ref_label_check'] = correct[int(results[-1]['label']) == ref_label] if ref_label != -100 else ''
                                results[-1]['restored_text'] = res.perturbed_result.attacked_text.text
                                results[-1]['is_fixed'] = True
                        except Exception as e:
                            print('Error:{}, try install TextAttack and tensorflow_text after 10 seconds...'.format(e))
                            time.sleep(10)
                            os.system('pip install git+https://github.com/yangheng95/TextAttack')
                            os.system('pip install tensorflow_text')
                            raise RuntimeError('Installation done, please run again...')

                    if ref_label != -100:
                        n_labeled += 1

                        if results[-1]['label'] == results[-1]['ref_label']:
                            n_correct += 1

                    if ref_is_adv_label != -100:
                        n_advdet_labeled += 1
                        if ref_is_adv_label == pred_is_adv_label:
                            n_advdet_correct += 1

        try:
            if print_result:
                for result in results:
                    text_printing = result['text']
                    text_info = ''
                    if result['label'] != '-100':
                        if not result['ref_label']:
                            text_info += ' -> <CLS:{}(ref:{} confidence:{})>'.format(result['label'], result['ref_label'], result['confidence'])
                        elif result['label'] == result['ref_label']:
                            text_info += colored(' -> <CLS:{}(ref:{} confidence:{})>'.format(result['label'], result['ref_label'], result['confidence']), 'green')
                        else:
                            text_info += colored(' -> <CLS:{}(ref:{} confidence:{})>'.format(result['label'], result['ref_label'], result['confidence']), 'red')

                    # AdvDet
                    if result['is_adv_label'] != '-100':
                        if not result['ref_is_adv_label']:
                            text_info += ' -> <AdvDet:{}(ref:{} confidence:{})>'.format(result['is_adv_label'], result['ref_is_adv_check'], result['is_adv_confidence'])
                        elif result['is_adv_label'] == result['ref_is_adv_label']:
                            text_info += colored(' -> <AdvDet:{}(ref:{} confidence:{})>'.format(result['is_adv_label'], result['ref_is_adv_label'], result['is_adv_confidence']), 'green')
                        else:
                            text_info += colored(' -> <AdvDet:{}(ref:{} confidence:{})>'.format(result['is_adv_label'], result['ref_is_adv_label'], result['is_adv_confidence']), 'red')

                    text_printing += text_info + colored('<perplexity:{}>'.format(result['perplexity']), 'yellow')
                    print(text_printing)
            if save_path:
                with open(save_path, 'w', encoding='utf8') as fout:
                    json.dump(str(results), fout, ensure_ascii=False)
                    print('inference result saved in: {}'.format(save_path))
        except Exception as e:
            print('Can not save result: {}, Exception: {}'.format(text_raw, e))

        if len(results) > 1:
            print('CLS Acc:{}%'.format(100 * n_correct / n_labeled if n_labeled else ''))
            print('AdvDet Acc:{}%'.format(100 * n_advdet_correct / n_advdet_labeled if n_advdet_labeled else ''))

        return results

    def clear_input_samples(self):
        self.dataset.all_data = []
