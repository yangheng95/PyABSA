# -*- coding: utf-8 -*-
# file: test_train_atepc.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import math
import os
import pickle
import random
import re
import time
from hashlib import sha256

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from seqeval.metrics import classification_report
from sklearn.metrics import f1_score
from torch import cuda
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from transformers import AutoTokenizer, AutoModel

from pyabsa.utils.file_utils import save_model
from pyabsa.utils.pyabsa_utils import print_args, resume_from_checkpoint, retry, TransformerConnectionError, optimizers
from ..dataset_utils.data_utils_for_training import ATEPCProcessor, convert_examples_to_features

import pytorch_warmup as warmup

try:
    import apex.amp as amp

    # assert torch.version.__version__ < '1.10.0'
    print('Use FP16 via Apex!')
except Exception:
    amp = None


class Instructor:

    def __init__(self, opt, logger):
        self.warmup_scheduler = None
        self.lr_scheduler = None
        self.opt = opt
        self.logger = logger

        self.train_dataloader = None
        self.test_dataloader = None
        # if opt.use_bert_spc:
        #     self.logger.info('Warning: The use_bert_spc is disabled for extracting aspect,'
        #                      ' reset use_bert_spc=False and go on... ')
        #     opt.use_bert_spc = False
        import warnings
        warnings.filterwarnings('ignore')
        if self.opt.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                self.opt.gradient_accumulation_steps))

        self.opt.batch_size = self.opt.batch_size // self.opt.gradient_accumulation_steps

        random.seed(self.opt.seed)
        np.random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)
        torch.cuda.manual_seed(self.opt.seed)

        if self.opt.model_path_to_save and not os.path.exists(self.opt.model_path_to_save):
            os.makedirs(self.opt.model_path_to_save)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.opt.pretrained_bert, do_lower_case='uncased' in self.opt.pretrained_bert)
            bert_base_model = AutoModel.from_pretrained(self.opt.pretrained_bert)
            self.opt.sep_indices = self.tokenizer.sep_token_id

        except ValueError as e:
            print('Init pretrained model failed, exception: {}'.format(e))
            raise TransformerConnectionError()

        processor = ATEPCProcessor(self.tokenizer)

        config_str = re.sub(r'<.*?>', '', str(sorted([str(self.opt.args[k]) for k in self.opt.args if k != 'seed'])))
        hash_tag = sha256(config_str.encode()).hexdigest()
        cache_path = '{}.{}.dataset.{}.cache'.format(self.opt.model_name, self.opt.dataset_name, hash_tag)

        if os.path.exists(cache_path):
            print('Loading dataset cache:', cache_path)
            with open(cache_path, mode='rb') as f:
                if self.opt.dataset_file['test']:
                    self.train_data, self.test_data, opt = pickle.load(f)

                else:
                    self.train_data, opt = pickle.load(f)
                # reset output dim according to dataset labels
                self.opt.polarities_dim = opt.polarities_dim

        else:
            self.train_examples = processor.get_train_examples(self.opt.dataset_file['train'], 'train')
            train_features = convert_examples_to_features(self.train_examples, self.opt.max_seq_len, self.tokenizer, self.opt)
            self.opt.label_list = sorted(list(self.opt.IOB_label_to_index.keys()))
            self.opt.num_labels = len(self.opt.label_list) + 1
            all_spc_input_ids = torch.tensor([f.input_ids_spc for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
            all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
            all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
            all_polarities = torch.tensor([f.polarity for f in train_features], dtype=torch.long)
            lcf_cdm_vec = torch.tensor([f.lcf_cdm_vec for f in train_features], dtype=torch.float32)
            lcf_cdw_vec = torch.tensor([f.lcf_cdw_vec for f in train_features], dtype=torch.float32)

            self.train_data = TensorDataset(all_spc_input_ids, all_segment_ids, all_input_mask, all_label_ids,
                                            all_polarities, all_valid_ids, all_lmask_ids, lcf_cdm_vec, lcf_cdw_vec)

            if self.opt.dataset_file['test']:
                self.test_examples = processor.get_test_examples(self.opt.dataset_file['test'], 'test')
                test_features = convert_examples_to_features(self.test_examples, self.opt.max_seq_len,
                                                             self.tokenizer, self.opt)
                all_spc_input_ids = torch.tensor([f.input_ids_spc for f in test_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
                all_polarities = torch.tensor([f.polarity for f in test_features], dtype=torch.long)
                all_valid_ids = torch.tensor([f.valid_ids for f in test_features], dtype=torch.long)
                all_lmask_ids = torch.tensor([f.label_mask for f in test_features], dtype=torch.long)
                lcf_cdm_vec = torch.tensor([f.lcf_cdm_vec for f in test_features], dtype=torch.float32)
                lcf_cdw_vec = torch.tensor([f.lcf_cdw_vec for f in test_features], dtype=torch.float32)
                self.test_data = TensorDataset(all_spc_input_ids, all_segment_ids, all_input_mask, all_label_ids, all_polarities,
                                               all_valid_ids, all_lmask_ids, lcf_cdm_vec, lcf_cdw_vec)

            if self.opt.cache_dataset and not os.path.exists(cache_path):
                print('Caching dataset... please remove cached dataset if change model or dataset')
                with open(cache_path, mode='wb') as f:
                    if self.opt.dataset_file['test']:
                        pickle.dump((self.train_data, self.test_data, self.opt), f)
                    else:
                        pickle.dump((self.train_data, self.opt), f)

        # only identify the labels in training set, make sure the labels are the same type in the test set
        for key in opt.args:
            if key not in self.opt.args:
                self.opt.args[key] = opt.args[key]
                self.opt.args_call_count[key] = opt.args_call_count[key]

        bert_base_model.config.num_labels = self.opt.num_labels
        self.opt.label_list = opt.label_list

        self.num_train_optimization_steps = int(
            len(self.train_data) / self.opt.batch_size / self.opt.gradient_accumulation_steps) * self.opt.num_epoch

        train_sampler = RandomSampler(self.train_data)
        self.train_dataloader = DataLoader(self.train_data, sampler=train_sampler, pin_memory=True, batch_size=self.opt.batch_size)
        if self.opt.dataset_file['test']:
            test_sampler = SequentialSampler(self.test_data)
            self.test_dataloader = DataLoader(self.test_data, sampler=test_sampler, pin_memory=True, batch_size=self.opt.batch_size)

        self.model = self.opt.model(bert_base_model, opt=self.opt)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.opt.l2reg},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0}
        ]
        if isinstance(self.opt.optimizer, str):
            self.optimizer = optimizers[self.opt.optimizer](self.optimizer_grouped_parameters,
                                                            lr=self.opt.learning_rate,
                                                            weight_decay=self.opt.l2reg)
        if amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

        if self.opt.auto_device == 'allcuda':
            self.model.to(self.opt.device)
            self.model = torch.nn.parallel.DataParallel(self.model)
        else:
            self.model.to(self.opt.device)

        self.opt.device = torch.device(self.opt.device)
        if self.opt.device.type == 'cuda':
            self.logger.info(
                "cuda memory allocated:{}".format(torch.cuda.memory_allocated(device=self.opt.device)))

        print_args(self.opt, self.logger)

    def run(self):
        patience = self.opt.patience + self.opt.evaluate_begin
        if self.opt.log_step < 0:
            self.opt.log_step = len(self.train_dataloader) if self.opt.log_step < 0 else self.opt.log_step

        if self.opt.warmup_step >= 0:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_dataloader) * self.opt.num_epoch)
            self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)

        self.logger.info("***** Running training for Aspect Term Extraction *****")
        self.logger.info("  Num examples = %d", len(self.train_data))
        self.logger.info("  Batch size = %d", self.opt.batch_size)
        self.logger.info("  Num steps = %d", self.num_train_optimization_steps)
        sum_loss = 0
        sum_apc_test_acc = 0
        sum_apc_test_f1 = 0
        sum_ate_test_f1 = 0
        self.opt.max_test_metrics = {'max_apc_test_acc': 0, 'max_apc_test_f1': 0, 'max_ate_test_f1': 0}
        self.opt.metrics_of_this_checkpoint = {'apc_acc': 0, 'apc_f1': 0, 'ate_f1': 0}
        global_step = 0
        save_path = '{0}/{1}_{2}'.format(self.opt.model_path_to_save,
                                         self.opt.model_name,
                                         self.opt.dataset_name
                                         )
        for epoch in range(int(self.opt.num_epoch)):
            nb_tr_examples, nb_tr_steps = 0, 0
            iterator = tqdm.tqdm(self.train_dataloader, postfix='Epoch:{}'.format(epoch))
            postfix = ''
            patience -= 1
            for step, batch in enumerate(iterator):
                self.model.train()
                input_ids_spc, segment_ids, input_mask, label_ids, polarity, \
                valid_ids, l_mask, lcf_cdm_vec, lcf_cdw_vec = batch
                input_ids_spc = input_ids_spc.to(self.opt.device)
                segment_ids = segment_ids.to(self.opt.device)
                input_mask = input_mask.to(self.opt.device)
                label_ids = label_ids.to(self.opt.device)
                polarity = polarity.to(self.opt.device)
                valid_ids = valid_ids.to(self.opt.device)
                l_mask = l_mask.to(self.opt.device)
                lcf_cdm_vec = lcf_cdm_vec.to(self.opt.device)
                lcf_cdw_vec = lcf_cdw_vec.to(self.opt.device)
                loss_ate, loss_apc = self.model(input_ids_spc,
                                                token_type_ids=segment_ids,
                                                attention_mask=input_mask,
                                                labels=label_ids,
                                                polarity=polarity,
                                                valid_ids=valid_ids,
                                                attention_mask_label=l_mask,
                                                lcf_cdm_vec=lcf_cdm_vec,
                                                lcf_cdw_vec=lcf_cdw_vec
                                                )
                # for multi-gpu, average loss by gpu instance number
                if self.opt.auto_device == 'allcuda':
                    loss_ate, loss_apc = loss_ate.mean(), loss_apc.mean()
                # loss_ate = loss_ate.item() / (loss_ate.item() + loss_apc.item()) * loss_ate
                # loss_apc = loss_apc.item() / (loss_ate.item() + loss_apc.item()) * loss_apc
                # loss = loss_ate + loss_apc
                loss = loss_ate + loss_apc  # the optimal weight of loss may be different according to dataset

                sum_loss += loss.item()

                if amp:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                    self.optimizer.step()

                if self.opt.warmup_step >= 0:
                    with self.warmup_scheduler.dampening():
                        self.lr_scheduler.step()

                nb_tr_examples += input_ids_spc.size(0)
                nb_tr_steps += 1
                self.optimizer.zero_grad()
                global_step += 1
                global_step += 1
                if global_step % self.opt.log_step == 0:
                    if self.opt.dataset_file['test'] and epoch >= self.opt.evaluate_begin:
                        apc_result, ate_result = self.evaluate()
                        sum_apc_test_acc += apc_result['apc_test_acc']
                        sum_apc_test_f1 += apc_result['apc_test_f1']
                        sum_ate_test_f1 += ate_result
                        self.opt.metrics_of_this_checkpoint['apc_acc'] = apc_result['apc_test_acc']
                        self.opt.metrics_of_this_checkpoint['apc_f1'] = apc_result['apc_test_f1']
                        self.opt.metrics_of_this_checkpoint['ate_f1'] = ate_result

                        if apc_result['apc_test_acc'] > self.opt.max_test_metrics['max_apc_test_acc'] or \
                            apc_result['apc_test_f1'] > self.opt.max_test_metrics['max_apc_test_f1'] or \
                            ate_result > self.opt.max_test_metrics['max_ate_test_f1']:
                            patience = self.opt.patience
                            if apc_result['apc_test_acc'] > self.opt.max_test_metrics['max_apc_test_acc']:
                                self.opt.max_test_metrics['max_apc_test_acc'] = apc_result['apc_test_acc']
                            if apc_result['apc_test_f1'] > self.opt.max_test_metrics['max_apc_test_f1']:
                                self.opt.max_test_metrics['max_apc_test_f1'] = apc_result['apc_test_f1']
                            if ate_result > self.opt.max_test_metrics['max_ate_test_f1']:
                                self.opt.max_test_metrics['max_ate_test_f1'] = ate_result

                            if self.opt.model_path_to_save:
                                # if save_path:
                                #     try:
                                #         shutil.rmtree(save_path)
                                #         # self.logger.info('Remove sub-self.optimal trained model:', save_path)
                                #     except:
                                #         self.logger.info('Can not remove sub-self.optimal trained model:', save_path)

                                save_path = '{0}/{1}_{2}_{3}_apcacc_{4}_apcf1_{5}_atef1_{6}/'.format(
                                    self.opt.model_path_to_save,
                                    self.opt.model_name,
                                    self.opt.dataset_name,
                                    self.opt.lcf,
                                    round(apc_result['apc_test_acc'], 2),
                                    round(apc_result['apc_test_f1'], 2),
                                    round(ate_result, 2)
                                )

                                save_model(self.opt, self.model, self.tokenizer, save_path)

                        current_apc_test_acc = apc_result['apc_test_acc']
                        current_apc_test_f1 = apc_result['apc_test_f1']
                        current_ate_test_f1 = round(ate_result, 2)

                        postfix = 'Epoch:{} | '.format(epoch)

                        postfix += 'loss_apc:{:.4f} | loss_ate:{:.4f} |'.format(loss_apc.item(), loss_ate.item())

                        postfix += ' APC_ACC: {}(max:{}) | APC_F1: {}(max:{}) | '.format(current_apc_test_acc,
                                                                                         self.opt.max_test_metrics[
                                                                                             'max_apc_test_acc'],
                                                                                         current_apc_test_f1,
                                                                                         self.opt.max_test_metrics[
                                                                                             'max_apc_test_f1']
                                                                                         )
                        postfix += 'ATE_F1: {}(max:{})'.format(current_ate_test_f1, self.opt.max_test_metrics[
                            'max_ate_test_f1'])
                    else:
                        if self.opt.save_mode:
                            save_model(self.opt, self.model, self.tokenizer, save_path + '_{}/'.format(loss.item()))
                        postfix = 'Epoch:{} | Loss: {} | No evaluation until epoch:{}'.format(epoch, round(loss.item(), 8), self.opt.evaluate_begin)

                iterator.postfix = postfix
                iterator.refresh()

            if patience < 0:
                break

        self.opt.MV.add_metric('Max-APC-Test-Acc', self.opt.max_test_metrics['max_apc_test_acc'])
        self.opt.MV.add_metric('Max-APC-Test-F1', self.opt.max_test_metrics['max_apc_test_f1'])
        self.opt.MV.add_metric('Max-ATE-Test-F1', self.opt.max_test_metrics['max_ate_test_f1'])

        self.opt.MV.summary(no_print=True)
        self.logger.info(self.opt.MV.summary(no_print=True))

        print('Training finished, we hope you can share your checkpoint with community, please see:',
              'https://github.com/yangheng95/PyABSA/blob/release/demos/documents/share-checkpoint.md')

        print_args(self.opt, self.logger)

        # return the model paths of multiple training
        # in case of loading the best model after training
        if self.opt.save_mode:
            del self.train_dataloader
            del self.test_dataloader
            del self.model
            cuda.empty_cache()
            time.sleep(3)
            return save_path
        else:
            # direct return model if do not evaluate
            del self.train_dataloader
            del self.test_dataloader
            cuda.empty_cache()
            time.sleep(3)
            return self.model, self.opt, self.tokenizer, sum_apc_test_acc, sum_apc_test_f1, sum_ate_test_f1

    def evaluate(self, eval_ATE=True, eval_APC=True):
        apc_result = {'apc_test_acc': 0, 'apc_test_f1': 0}
        ate_result = 0
        y_true = []
        y_pred = []
        n_test_correct, n_test_total = 0, 0
        test_apc_logits_all, test_polarities_all = None, None
        self.model.eval()
        label_map = {i: label for i, label in enumerate(self.opt.label_list, 1)}

        for i, batch in enumerate(self.test_dataloader):
            input_ids_spc, segment_ids, input_mask, label_ids, polarity, \
            valid_ids, l_mask, lcf_cdm_vec, lcf_cdw_vec = batch
            input_ids_spc = input_ids_spc.to(self.opt.device)
            segment_ids = segment_ids.to(self.opt.device)
            input_mask = input_mask.to(self.opt.device)
            label_ids = label_ids.to(self.opt.device)
            polarity = polarity.to(self.opt.device)
            valid_ids = valid_ids.to(self.opt.device)
            l_mask = l_mask.to(self.opt.device)
            lcf_cdm_vec = lcf_cdm_vec.to(self.opt.device)
            lcf_cdw_vec = lcf_cdw_vec.to(self.opt.device)
            with torch.no_grad():
                if self.opt.auto_device == 'allcuda':
                    ate_logits, apc_logits = self.model.module(input_ids_spc,
                                                               token_type_ids=segment_ids,
                                                               attention_mask=input_mask,
                                                               labels=None,
                                                               polarity=polarity,
                                                               valid_ids=valid_ids,
                                                               attention_mask_label=l_mask,
                                                               lcf_cdm_vec=lcf_cdm_vec,
                                                               lcf_cdw_vec=lcf_cdw_vec
                                                               )
                else:
                    ate_logits, apc_logits = self.model(input_ids_spc,
                                                        token_type_ids=segment_ids,
                                                        attention_mask=input_mask,
                                                        labels=None,
                                                        polarity=polarity,
                                                        valid_ids=valid_ids,
                                                        attention_mask_label=l_mask,
                                                        lcf_cdm_vec=lcf_cdm_vec,
                                                        lcf_cdw_vec=lcf_cdw_vec
                                                        )
            if eval_APC:
                n_test_correct += (torch.argmax(apc_logits, -1) == polarity).sum().item()
                n_test_total += len(polarity)

                if test_polarities_all is None:
                    test_polarities_all = polarity
                    test_apc_logits_all = apc_logits
                else:
                    test_polarities_all = torch.cat((test_polarities_all, polarity), dim=0)
                    test_apc_logits_all = torch.cat((test_apc_logits_all, apc_logits), dim=0)

            if eval_ATE:
                if not self.opt.use_bert_spc:
                    if self.opt.auto_device == 'allcuda':
                        label_ids = self.model.module.get_batch_token_labels_bert_base_indices(label_ids)
                    else:
                        label_ids = self.model.get_batch_token_labels_bert_base_indices(label_ids)
                ate_logits = torch.argmax(F.log_softmax(ate_logits, dim=2), dim=2)
                ate_logits = ate_logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()
                for i, label in enumerate(label_ids):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(label):
                        if j == 0:
                            continue
                        elif label_ids[i][j] == len(self.opt.label_list):
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                            break
                        else:
                            temp_1.append(label_map.get(label_ids[i][j], 'O'))
                            temp_2.append(label_map.get(ate_logits[i][j], 'O'))
        if eval_APC:
            test_acc = n_test_correct / n_test_total

            test_f1 = f1_score(torch.argmax(test_apc_logits_all, -1).cpu(), test_polarities_all.cpu(),
                               labels=list(range(self.opt.polarities_dim)), average='macro')

            test_acc = round(test_acc * 100, 2)
            test_f1 = round(test_f1 * 100, 2)
            apc_result = {'apc_test_acc': test_acc, 'apc_test_f1': test_f1}

        if eval_ATE:
            report = classification_report(y_true, y_pred, digits=4)
            tmps = report.split()
            ate_result = round(float(tmps[7]) * 100, 2)
        return apc_result, ate_result


@retry
def train4atepc(opt, from_checkpoint_path, logger):
    # in case of handling ConnectionError exception
    trainer = Instructor(opt, logger)
    resume_from_checkpoint(trainer, from_checkpoint_path)

    return trainer.run()
