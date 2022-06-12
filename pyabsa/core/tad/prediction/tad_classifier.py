# -*- coding: utf-8 -*-
# file: text_classifier.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.
import json
import os
import pickle
import random

import numpy
import torch
from findfile import find_file
from termcolor import colored
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, DebertaV2ForMaskedLM, RobertaForMaskedLM, BertForMaskedLM

from ....functional.config import TADConfigManager
from ....functional.dataset import TCDatasetList

from ....functional.dataset import detect_infer_dataset

from ..models import TADGloVeTCModelList, TADBERTTCModelList
from ..classic.__glove__.dataset_utils.data_utils_for_inference import TADGloVeTCDataset
from ..classic.__bert__.dataset_utils.data_utils_for_inference import TADBERTTCDataset, Tokenizer4Pretraining

from ..classic.__glove__.dataset_utils.data_utils_for_training import build_embedding_matrix, build_tokenizer

from ....utils.pyabsa_utils import print_args, TransformerConnectionError


def get_mlm_and_tokenizer(text_classifier, config):
    if isinstance(text_classifier, TADTextClassifier):
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


class TADTextClassifier:
    def __init__(self, model_arg=None, cal_perplexity=False, eval_batch_size=128):
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

                if state_dict_path or model_path:
                    if hasattr(TADBERTTCModelList, self.opt.model.__name__):
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

            if not hasattr(TADGloVeTCModelList, self.opt.model.__name__) \
                and not hasattr(TADBERTTCModelList, self.opt.model.__name__):
                raise KeyError('The checkpoint you are loading is not from classifier model.')

        # if hasattr(TADBERTTCModelList, self.opt.model.__name__):
        #     self.dataset = TADBERTTCDataset(tokenizer=self.tokenizer, opt=self.opt)
        #
        # elif hasattr(TADGloVeTCModelList, self.opt.model.__name__):
        #     self.dataset = TADGloVeTCDataset(tokenizer=self.tokenizer, opt=self.opt)

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

        if self.cal_perplexity:
            try:
                self.MLM, self.MLM_tokenizer = get_mlm_and_tokenizer(self, self.opt)
            except Exception as e:
                self.MLM, self.MLM_tokenizer = None, None

        from boost_aug import AugmentBackend, TADBoostAug

        tc_config = TADConfigManager.get_tad_config_english()
        tc_config.model = TADBERTTCModelList.TADBERT  # 'BERT' model can be used for DeBERTa or BERT
        backend = AugmentBackend.EDA
        dataset_map = {
            'sst2': TCDatasetList.SST2,
            'agnews10k': TCDatasetList.AGNews10K,
            'yelp10k': TCDatasetList.Yelp10K,
            'imdb10k': TCDatasetList.IMDB10K
        }
        self.augmentor = TADBoostAug(ROOT=os.getcwd(),
                                     AUGMENT_BACKEND=backend,
                                     CLASSIFIER_TRAINING_NUM=2,
                                     WINNER_NUM_PER_CASE=8,
                                     AUGMENT_NUM_PER_CASE=16,
                                     CONFIDENCE_THRESHOLD=0.8,
                                     PERPLEXITY_THRESHOLD=5,
                                     USE_LABEL=True,
                                     device=self.opt.device)
        # self.augmentor.tc_mono_augment(tc_config,
        #                                dataset_map[self.opt.dataset_name.lower()],
        #                                rewrite_cache=False,
        #                                train_after_aug=False
        #                                )
        # self.augmentor.tc_boost_augment(tc_config,
        #                                 dataset_map[self.opt.dataset_name.lower()],
        #                                 rewrite_cache=True,
        #                                 train_after_aug=True
        #                                 )
        self.augmentor.USE_LABEL = False
        # self.augmentor.load_augmentor('tad-{}'.format(self.opt.dataset_name))
        self.augmentor.load_augmentor(self)

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
                    attack_defense=True,
                    print_result=True,
                    save_result=False,
                    ignore_error=True):

        # if clear_input_samples:
        #     self.clear_input_samples()

        save_path = os.path.join(os.getcwd(), 'tad_text_classification.result.json')

        target_file = detect_infer_dataset(target_file, task='text_defense')
        if not target_file:
            raise FileNotFoundError('Can not find inference datasets!')

        if hasattr(TADBERTTCModelList, self.opt.model.__name__):
            dataset = TADBERTTCDataset(tokenizer=self.tokenizer, opt=self.opt)

        else:
            dataset = TADGloVeTCDataset(tokenizer=self.tokenizer, opt=self.opt)

        dataset.prepare_infer_dataset(target_file, ignore_error=ignore_error)
        self.infer_dataloader = DataLoader(dataset=dataset, batch_size=self.opt.eval_batch_size, pin_memory=True, shuffle=False)
        return self._infer(save_path=save_path if save_result else None, print_result=print_result, attack_defense=attack_defense)

    def infer(self,
              text: str = None,
              attack_defense=True,
              print_result=True,
              ignore_error=True,
              ):

        if hasattr(TADBERTTCModelList, self.opt.model.__name__):
            dataset = TADBERTTCDataset(tokenizer=self.tokenizer, opt=self.opt)

        else:
            dataset = TADGloVeTCDataset(tokenizer=self.tokenizer, opt=self.opt)

        if text:
            dataset.prepare_infer_sample(text, ignore_error=ignore_error)
        else:
            raise RuntimeError('Please specify your datasets path!')
        self.infer_dataloader = DataLoader(dataset=dataset, batch_size=self.opt.eval_batch_size, shuffle=False)
        return self._infer(print_result=print_result, attack_defense=attack_defense)[0]

    def _infer(self, save_path=None, print_result=True, attack_defense=True):

        _params = filter(lambda p: p.requires_grad, self.model.parameters())

        correct = {True: 'Correct', False: 'Wrong'}
        results = []
        perplexity = 'N.A.'

        with torch.no_grad():
            self.model.eval()
            n_correct = 0
            n_labeled = 0

            n_advdet_correct = 0
            n_advdet_labeled = 0

            for _, sample in enumerate(self.infer_dataloader):
                inputs = [sample[col].to(self.opt.device) for col in self.opt.inputs_cols]

                logits, advdet_logits = self.model(inputs)
                probs, advdet_probs = torch.softmax(logits, dim=-1), torch.softmax(advdet_logits, dim=-1)

                for i, (prob, advdet_prob) in enumerate(zip(probs, advdet_probs)):
                    text_raw = sample['text_raw'][i]

                    pred_label = int(prob.argmax(axis=-1))
                    advdet = int(advdet_prob.argmax(axis=-1))
                    real_label = int(sample['label'][i]) if int(sample['label'][i]) in self.opt.index_to_label else ''
                    perturb_label = int(sample['perturb_label'][i]) if int(sample['perturb_label'][i]) in self.opt.index_to_perturb_label else ''
                    real_adv = int(sample['is_adv'][i]) if int(sample['is_adv'][i]) in self.opt.index_to_is_adv else ''

                    if real_label != -100:
                        n_labeled += 1
                        if pred_label == real_label:
                            n_correct += 1

                    if real_adv != -100:
                        n_advdet_labeled += 1
                        if real_adv == advdet:
                            n_advdet_correct += 1

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
                        'ref_label': self.opt.index_to_label[real_label] if isinstance(real_label, int) else real_label,
                        'ref_label_check': correct[pred_label == real_label] if real_label != -100 and isinstance(real_label, int) else '',

                        'is_adv_label': self.opt.index_to_is_adv[advdet],
                        'is_adv_probs': advdet_prob.cpu().numpy(),
                        'is_adv_confidence': float(max(advdet_prob)),
                        'ref_is_adv_label': self.opt.index_to_is_adv[real_adv] if isinstance(real_adv, int) else real_adv,
                        'ref_is_adv_check': correct[advdet == real_adv] if real_adv != -100 and isinstance(real_adv, int) else '',

                        'perplexity': perplexity,
                    })

                    if attack_defense:
                        if advdet != 0 and real_label != perturb_label and real_label != -100:
                            augs = self.augmentor.single_augment(text_raw, real_label, 1, attack_defense=False)
                            if augs:
                                for aug in augs:
                                    infer_res = self.augmentor.tad_classifier.infer(aug + '!ref!{},-100,-100'.format(real_label), attack_defense=False)
                            print('\n')
        try:
            if print_result:
                for result in results:
                    text_printing = result['text']
                    text_info = ''
                    # CLS
                    if result['label'] != '-100':
                        if not result['ref_label']:
                            text_info += ' -> <CLS:{}(ref:{} confidence:{})>'.format(result['label'], result['ref_label'], result['confidence'])
                        elif result['label'] == result['ref_label']:
                            text_info += colored(' -> <CLS:{}(ref:{} confidence:{})>'.format(result['label'], result['ref_label'], result['confidence']), 'green')
                        else:
                            text_info += colored(' -> <CLS:{}(ref:{} confidence:{})>'.format(result['label'], result['ref_label'], result['confidence']), 'red')

                    # AdvDet
                    if result['is_adv_label'] != '-100':
                        if not result['ref_label']:
                            text_info += ' -> <AdvDet:{}(ref:{})>'.format(result['is_adv_label'], result['ref_is_adv_check'])
                        elif result['is_adv_label'] == result['ref_is_adv_label']:
                            text_info += colored(' -> <AdvDet:{}(ref:{})>'.format(result['is_adv_label'], result['ref_is_adv_check']), 'green')
                        else:
                            text_info += colored(' -> <AdvDet:{}(ref:{})>'.format(result['is_adv_label'], result['ref_is_adv_check']), 'red')

                    text_printing += text_info + colored('<perplexity:{}>'.format(result['perplexity']), 'yellow')
                    print(text_printing)
            if save_path:
                with open(save_path, 'w', encoding='utf8') as fout:
                    json.dump(str(results), fout, ensure_ascii=False)
                    print('inference result saved in: {}'.format(save_path))
        except Exception as e:
            print('Can not save result: {}, Exception: {}'.format(text_raw, e))

        if len(self.infer_dataloader) > 1:
            print('CLS Acc:{}%'.format(100 * n_correct / n_labeled if n_labeled else ''))
            print('AdvDet Acc:{}%'.format(100 * n_advdet_correct / n_advdet_labeled if n_advdet_labeled else ''))

        return results

    def clear_input_samples(self):
        self.dataset.all_data = []
