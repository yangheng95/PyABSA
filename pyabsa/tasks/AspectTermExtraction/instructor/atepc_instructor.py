# -*- coding: utf-8 -*-
# file: test_train_atepc.py
# time: 2021/5/26 0026
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os
import pickle
import re
import time
from hashlib import sha256

import numpy
import pandas
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
import tqdm
from seqeval.metrics import classification_report
from sklearn.metrics import f1_score
from torch import cuda
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from transformers import AutoTokenizer, AutoModel

from pyabsa import DeviceTypeOption
from pyabsa.framework.instructor_class.instructor_template import BaseTrainingInstructor
from ..dataset_utils.__lcf__.data_utils_for_training import ATEPCProcessor, convert_examples_to_features
from pyabsa.utils.file_utils.file_utils import save_model
from pyabsa.utils.pyabsa_utils import print_args, init_optimizer, fprint, rprint

import pytorch_warmup as warmup


class ATEPCTrainingInstructor(BaseTrainingInstructor):

    def __init__(self, config):
        super().__init__(config)

        self._load_dataset_and_prepare_dataloader()

        self._init_misc()

    def _load_dataset_and_prepare_dataloader(self):

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_bert,
                                                       do_lower_case='uncased' in self.config.pretrained_bert)

        processor = ATEPCProcessor(self.tokenizer)
        cache_path = self.load_cache_dataset()
        if cache_path is None:
            self.train_examples = processor.get_train_examples(self.config.dataset_file['train'], 'train')
            train_features = convert_examples_to_features(self.train_examples, self.config.max_seq_len, self.tokenizer,
                                                          self.config)
            self.config.label_list = sorted(list(self.config.IOB_label_to_index.keys()))
            self.config.num_labels = len(self.config.label_list) + 1
            all_spc_input_ids = torch.tensor([f.input_ids_spc for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
            all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
            all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
            all_polarities = torch.tensor([f.polarity for f in train_features], dtype=torch.long)
            lcf_cdm_vec = torch.tensor([f.lcf_cdm_vec for f in train_features], dtype=torch.float32)
            lcf_cdw_vec = torch.tensor([f.lcf_cdw_vec for f in train_features], dtype=torch.float32)

            self.train_set = TensorDataset(all_spc_input_ids, all_segment_ids, all_input_mask, all_label_ids,
                                           all_polarities, all_valid_ids, all_lmask_ids, lcf_cdm_vec, lcf_cdw_vec)

            if self.config.dataset_file['valid']:
                self.valid_examples = processor.get_valid_examples(self.config.dataset_file['valid'], 'valid')
                valid_features = convert_examples_to_features(self.valid_examples, self.config.max_seq_len,
                                                              self.tokenizer, self.config)
                all_spc_input_ids = torch.tensor([f.input_ids_spc for f in valid_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
                all_polarities = torch.tensor([f.polarity for f in valid_features], dtype=torch.long)
                all_valid_ids = torch.tensor([f.valid_ids for f in valid_features], dtype=torch.long)
                all_lmask_ids = torch.tensor([f.label_mask for f in valid_features], dtype=torch.long)
                lcf_cdm_vec = torch.tensor([f.lcf_cdm_vec for f in valid_features], dtype=torch.float32)
                lcf_cdw_vec = torch.tensor([f.lcf_cdw_vec for f in valid_features], dtype=torch.float32)
                self.valid_set = TensorDataset(all_spc_input_ids, all_segment_ids, all_input_mask, all_label_ids,
                                               all_polarities, all_valid_ids, all_lmask_ids, lcf_cdm_vec, lcf_cdw_vec)
            else:
                self.valid_set = None

            if self.config.dataset_file['test']:
                self.test_examples = processor.get_test_examples(self.config.dataset_file['test'], 'test')
                test_features = convert_examples_to_features(self.test_examples, self.config.max_seq_len,
                                                             self.tokenizer, self.config)
                all_spc_input_ids = torch.tensor([f.input_ids_spc for f in test_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
                all_polarities = torch.tensor([f.polarity for f in test_features], dtype=torch.long)
                all_valid_ids = torch.tensor([f.valid_ids for f in test_features], dtype=torch.long)
                all_lmask_ids = torch.tensor([f.label_mask for f in test_features], dtype=torch.long)
                lcf_cdm_vec = torch.tensor([f.lcf_cdm_vec for f in test_features], dtype=torch.float32)
                lcf_cdw_vec = torch.tensor([f.lcf_cdw_vec for f in test_features], dtype=torch.float32)

                self.test_set = TensorDataset(all_spc_input_ids, all_segment_ids, all_input_mask, all_label_ids,
                                              all_polarities, all_valid_ids, all_lmask_ids, lcf_cdm_vec, lcf_cdw_vec)
            else:
                self.test_set = None

        self.num_train_optimization_steps = int(
            len(self.train_set) / self.config.batch_size / self.config.gradient_accumulation_steps) * self.config.num_epoch

        train_sampler = RandomSampler(self.train_set)
        self.train_dataloader = DataLoader(self.train_set, sampler=train_sampler, pin_memory=True,
                                           batch_size=self.config.batch_size)

        if self.valid_set:
            valid_sampler = SequentialSampler(self.valid_set)
            self.valid_dataloader = DataLoader(self.valid_set, sampler=valid_sampler, pin_memory=True,
                                               batch_size=self.config.batch_size)

        if self.test_set:
            test_sampler = SequentialSampler(self.test_set)
            self.test_dataloader = DataLoader(self.test_set, sampler=test_sampler, pin_memory=True,
                                              batch_size=self.config.batch_size)

        self.bert_base_model = AutoModel.from_pretrained(self.config.pretrained_bert)
        self.config.sep_indices = self.tokenizer.sep_token_id
        self.bert_base_model.config.num_labels = self.config.num_labels

        self.save_cache_dataset()

        self.model = self.config.model(self.bert_base_model, config=self.config)

    def _train_and_evaluate(self, criterion):
        losses = []

        patience = self.config.patience + self.config.evaluate_begin
        if self.config.log_step < 0:
            self.config.log_step = len(self.train_dataloader) if self.config.log_step < 0 else self.config.log_step

        if self.config.warmup_step >= 0:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(
                self.train_dataloader) * self.config.num_epoch)
            self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)

        self.logger.info("***** Running training for {} *****".format(self.config.task_name))
        self.logger.info("  Num examples = %d", len(self.train_set))
        self.logger.info("  Batch size = %d", self.config.batch_size)
        self.logger.info("  Num steps = %d", self.num_train_optimization_steps)
        sum_loss = 0
        sum_apc_test_acc = 0
        sum_apc_test_f1 = 0
        sum_ate_test_f1 = 0
        self.config.max_test_metrics = {'max_apc_test_acc': 0, 'max_apc_test_f1': 0, 'max_ate_test_f1': 0}
        self.config.metrics_of_this_checkpoint = {'apc_acc': 0, 'apc_f1': 0, 'ate_f1': 0}
        global_step = 0
        save_path = '{0}/{1}_{2}'.format(self.config.model_path_to_save,
                                         self.config.model_name,
                                         self.config.dataset_name
                                         )
        for epoch in range(int(self.config.num_epoch)):
            nb_tr_examples, nb_tr_steps = 0, 0
            iterator = tqdm.tqdm(self.train_dataloader, postfix='Epoch:{}'.format(epoch))
            description = ''
            patience -= 1
            for step, batch in enumerate(iterator):
                self.model.train()
                input_ids_spc, segment_ids, input_mask, label_ids, polarity, \
                    valid_ids, l_mask, lcf_cdm_vec, lcf_cdw_vec = batch
                input_ids_spc = input_ids_spc.to(self.config.device)
                segment_ids = segment_ids.to(self.config.device)
                input_mask = input_mask.to(self.config.device)
                label_ids = label_ids.to(self.config.device)
                polarity = polarity.to(self.config.device)
                valid_ids = valid_ids.to(self.config.device)
                l_mask = l_mask.to(self.config.device)
                lcf_cdm_vec = lcf_cdm_vec.to(self.config.device)
                lcf_cdw_vec = lcf_cdw_vec.to(self.config.device)
                if self.config.use_amp:
                    with torch.cuda.amp.autocast():
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
                else:
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
                if self.config.auto_device == DeviceTypeOption.ALL_CUDA:
                    loss_ate, loss_apc = loss_ate.mean(), loss_apc.mean()
                ate_loss_weight = self.config.args.get('ate_loss_weight', 1.0)

                loss = loss_ate + ate_loss_weight * loss_apc  # the optimal weight of loss may be different according to dataset

                sum_loss += loss.item()

                losses.append(loss.item())

                if self.config.use_amp and self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                if self.config.warmup_step >= 0:
                    with self.warmup_scheduler.dampening():
                        self.lr_scheduler.step()

                nb_tr_examples += input_ids_spc.size(0)
                nb_tr_steps += 1
                self.optimizer.zero_grad()
                global_step += 1
                global_step += 1
                if global_step % self.config.log_step == 0:
                    if self.test_dataloader and epoch >= self.config.evaluate_begin:
                        if self.valid_set:
                            apc_result, ate_result = self._evaluate_acc_f1(self.valid_dataloader)
                        else:
                            apc_result, ate_result = self._evaluate_acc_f1(self.test_dataloader)
                        sum_apc_test_acc += apc_result['apc_test_acc']
                        sum_apc_test_f1 += apc_result['apc_test_f1']
                        sum_ate_test_f1 += ate_result
                        self.config.metrics_of_this_checkpoint['apc_acc'] = apc_result['apc_test_acc']
                        self.config.metrics_of_this_checkpoint['apc_f1'] = apc_result['apc_test_f1']
                        self.config.metrics_of_this_checkpoint['ate_f1'] = ate_result

                        if apc_result['apc_test_acc'] > self.config.max_test_metrics['max_apc_test_acc'] or \
                                apc_result['apc_test_f1'] > self.config.max_test_metrics['max_apc_test_f1'] or \
                                ate_result > self.config.max_test_metrics['max_ate_test_f1']:
                            patience = self.config.patience
                            if apc_result['apc_test_acc'] > self.config.max_test_metrics['max_apc_test_acc']:
                                self.config.max_test_metrics['max_apc_test_acc'] = apc_result['apc_test_acc']
                            if apc_result['apc_test_f1'] > self.config.max_test_metrics['max_apc_test_f1']:
                                self.config.max_test_metrics['max_apc_test_f1'] = apc_result['apc_test_f1']
                            if ate_result > self.config.max_test_metrics['max_ate_test_f1']:
                                self.config.max_test_metrics['max_ate_test_f1'] = ate_result

                            if self.config.model_path_to_save:
                                # if save_path:
                                #     try:
                                #         shutil.rmtree(save_path)
                                #         # self.logger.info('Remove sub-self.configimal trained model:', save_path)
                                #     except:
                                #         self.logger.info('Can not remove sub-self.configimal trained model:', save_path)

                                save_path = '{0}/{1}_{2}_{3}_apcacc_{4}_apcf1_{5}_atef1_{6}/'.format(
                                    self.config.model_path_to_save,
                                    self.config.model_name,
                                    self.config.dataset_name,
                                    self.config.lcf,
                                    round(apc_result['apc_test_acc'], 2),
                                    round(apc_result['apc_test_f1'], 2),
                                    round(ate_result, 2)
                                )

                                save_model(self.config, self.model, self.tokenizer, save_path)

                        current_apc_test_acc = apc_result['apc_test_acc']
                        current_apc_test_f1 = apc_result['apc_test_f1']
                        current_ate_test_f1 = round(ate_result, 2)

                        description = 'Epoch:{} | '.format(epoch)

                        postfix += 'loss_apc:{:.4f} | loss_ate:{:.4f} |'.format(loss_apc.item(), loss_ate.item())

                        postfix += ' APC_ACC: {}(max:{}) | APC_F1: {}(max:{}) | '.format(current_apc_test_acc,
                                                                                         self.config.max_test_metrics[
                                                                                             'max_apc_test_acc'],
                                                                                         current_apc_test_f1,
                                                                                         self.config.max_test_metrics[
                                                                                             'max_apc_test_f1']
                                                                                         )
                        postfix += 'ATE_F1: {}(max:{})'.format(current_ate_test_f1, self.config.max_test_metrics[
                            'max_ate_test_f1'])
                    else:
                        if self.config.save_mode and epoch >= self.config.evaluate_begin:
                            save_model(self.config, self.model, self.tokenizer, save_path + '_{}/'.format(loss.item()))
                        description = 'Epoch:{} | Loss: {} | No evaluation until epoch:{}'.format(epoch,
                                                                                                  round(loss.item(), 8),
                                                                                                  self.config.evaluate_begin)

                iterator.set_description(description)
                iterator.refresh()

            if patience < 0:
                break

        apc_result, ate_result = self._evaluate_acc_f1(self.test_dataloader)

        if self.valid_set and self.test_set:
            self.config.MV.add_metric('Test-APC-Acc', apc_result['apc_test_acc'])
            self.config.MV.add_metric('Test-APC-F1', apc_result['apc_test_f1'])
            self.config.MV.add_metric('Test-ATE-F1', ate_result)

        else:
            self.config.MV.add_metric('Max-APC-Test-Acc w/o Valid Set',
                                      self.config.max_test_metrics['max_apc_test_acc'])
            self.config.MV.add_metric('Max-APC-Test-F1 w/o Valid Set', self.config.max_test_metrics['max_apc_test_f1'])
            self.config.MV.add_metric('Max-ATE-Test-F1 w/o Valid Set', self.config.max_test_metrics['max_ate_test_f1'])

        self.config.MV.summary(no_print=True)
        self.logger.info(self.config.MV.summary(no_print=True))

        rolling_intv = 5
        df = pandas.DataFrame(losses)
        losses = list(numpy.hstack(df.rolling(rolling_intv, min_periods=1).mean().values))
        self.config.loss = losses[-1]
        # self.config.loss = np.average(losses)

        print_args(self.config, self.logger)

        # return the model paths of multiple trainer
        # in case of loading the best model after trainer
        if self.config.save_mode:
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
            return self.model, self.config, self.tokenizer, sum_apc_test_acc, sum_apc_test_f1, sum_ate_test_f1

    def _k_fold_train_and_evaluate(self, criterion):
        pass

    def _evaluate_acc_f1(self, test_dataloader, eval_ATE=True, eval_APC=True):
        if test_dataloader is None:
            test_dataloader = self.test_dataloader
        apc_result = {'apc_test_acc': 0, 'apc_test_f1': 0}
        ate_result = 0
        y_true = []
        y_pred = []
        n_test_correct, n_test_total = 0, 0
        test_apc_logits_all, test_polarities_all = None, None
        self.model.eval()
        label_map = {i: label for i, label in enumerate(self.config.label_list, 1)}

        for i_batch, batch in enumerate(test_dataloader):
            input_ids_spc, segment_ids, input_mask, label_ids, polarity, \
                valid_ids, l_mask, lcf_cdm_vec, lcf_cdw_vec = batch

            input_ids_spc = input_ids_spc.to(self.config.device)
            segment_ids = segment_ids.to(self.config.device)
            input_mask = input_mask.to(self.config.device)
            label_ids = label_ids.to(self.config.device)
            polarity = polarity.to(self.config.device)
            valid_ids = valid_ids.to(self.config.device)
            l_mask = l_mask.to(self.config.device)
            lcf_cdm_vec = lcf_cdm_vec.to(self.config.device)
            lcf_cdw_vec = lcf_cdw_vec.to(self.config.device)

            with torch.no_grad():
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
                input_ids = input_ids_spc
                if self.config.use_bert_spc:
                    label_ids = self.model.get_batch_token_labels_bert_base_indices(label_ids)
                    input_ids = self.model.get_ids_for_local_context_extractor(input_ids_spc)
                ate_logits = torch.argmax(F.log_softmax(ate_logits, dim=2), dim=2)
                ate_logits = ate_logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                for i, label in enumerate(label_ids):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(label):
                        if j == 0:
                            continue
                        elif len(temp_1) == torch.count_nonzero(input_ids[i]):
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                            break
                        else:
                            temp_1.append(label_map.get(label_ids[i][j], 'O'))
                            temp_2.append(label_map.get(ate_logits[i][j], 'O'))
        if eval_APC:
            test_acc = n_test_correct / n_test_total

            test_f1 = f1_score(torch.argmax(test_apc_logits_all, -1).cpu(), test_polarities_all.cpu(),
                               labels=list(range(self.config.output_dim)), average='macro')

            test_acc = round(test_acc * 100, 2)
            test_f1 = round(test_f1 * 100, 2)
            apc_result = {'apc_test_acc': test_acc, 'apc_test_f1': test_f1}
            if self.config.args.get('show_metric', False):
                try:
                    apc_report = metrics.classification_report(test_apc_logits_all.cpu(),
                                                               torch.argmax(test_polarities_all, -1).cpu(),
                                                               target_names=[self.config.index_to_label[x] for x in
                                                                             self.config.index_to_label])
                    fprint('\n---------------------------- APC Classification Report ----------------------------\n')
                    fprint(apc_report)
                    fprint('\n---------------------------- APC Classification Report ----------------------------\n')
                except Exception as e:
                    # No enough raw_data to calculate the report
                    pass
        if eval_ATE:
            try:
                report = classification_report(y_true, y_pred, digits=4)
                tmps = report.split()
                ate_result = round(float(tmps[7]) * 100, 2)
                if self.config.args.get('show_metric', False):
                    fprint('\n---------------------------- ATE Classification Report ----------------------------\n')
                    rprint(report)
                    fprint('\n---------------------------- ATE Classification Report ----------------------------\n')
            except Exception as e:
                # No enough raw_data to calculate the report
                pass

        return apc_result, ate_result

    def _init_misc(self):
        if self.config.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                self.config.gradient_accumulation_steps))

        self.config.batch_size = self.config.batch_size // self.config.gradient_accumulation_steps

        if self.config.model_path_to_save and not os.path.exists(self.config.model_path_to_save):
            os.makedirs(self.config.model_path_to_save)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config.l2reg},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0}
        ]

        if self.config.auto_device == DeviceTypeOption.ALL_CUDA:
            self.model.to(self.config.device)
            self.model = torch.nn.parallel.DataParallel(self.model)
            self.model = self.model.module
        else:
            self.model.to(self.config.device)

        if isinstance(self.config.optimizer, str):
            self.optimizer = init_optimizer(self.config.optimizer)(
                self.optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.l2reg,

            )
        self.config.device = torch.device(self.config.device)
        if self.config.device.type == 'cuda':
            self.logger.info(
                "cuda memory allocated:{}".format(torch.cuda.memory_allocated(device=self.config.device)))

        print_args(self.config, self.logger)

    def _cache_or_load_dataset(self):
        pass

    def run(self):
        return self._train(criterion=None)
