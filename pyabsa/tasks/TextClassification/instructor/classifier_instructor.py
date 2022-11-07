# -*- coding: utf-8 -*-
# file: classifier_instructor.py
# time: 2021/4/22 0022
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os
import pickle
import random
import re
import shutil
import time
from hashlib import sha256

import numpy
import torch
import torch.nn as nn
from findfile import find_file
from sklearn import metrics
from torch import cuda
from torch.utils.data import DataLoader, random_split, ConcatDataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoModel

from pyabsa import DeviceTypeOption
from pyabsa.framework.instructor_class.instructor_template import BaseTrainingInstructor
from pyabsa.tasks.TextClassification.dataset_utils.__classic__.data_utils_for_training import GloVeTCDataset
from pyabsa.tasks.TextClassification.dataset_utils.__plm__.data_utils_for_training import BERTTCDataset
from pyabsa.tasks.TextClassification.models import GloVeTCModelList, BERTTCModelList

import pytorch_warmup as warmup

from pyabsa.utils.file_utils.file_utils import save_model
from pyabsa.utils.pyabsa_utils import init_optimizer, print_args
from pyabsa.framework.tokenizer_class.tokenizer_class import PretrainedTokenizer, Tokenizer, build_embedding_matrix


class TCTrainingInstructor(BaseTrainingInstructor):

    def __init__(self, config):
        super().__init__(config)

        self._load_dataset_and_prepare_dataloader()

        self._init_misc()

    def reload_model(self, ckpt='./init_state_dict.bin'):
        if os.path.exists(ckpt):
            self.model.load_state_dict(torch.load(find_file(ckpt, or_key=['.bin', 'state_dict'])))

    def prepare_dataloader(self, train_set):
        if self.config.cross_validate_fold < 1:
            train_sampler = RandomSampler(self.train_set if not self.train_set else self.train_set)
            self.train_dataloaders.append(DataLoader(dataset=train_set,
                                                     batch_size=self.config.batch_size,
                                                     sampler=train_sampler,
                                                     pin_memory=True))
            if self.test_set:
                self.test_dataloader = DataLoader(dataset=self.test_set, batch_size=self.config.batch_size, shuffle=False)

            if self.valid_set:
                self.valid_dataloader = DataLoader(dataset=self.valid_set, batch_size=self.config.batch_size, shuffle=False)
        else:
            split_dataset = train_set
            len_per_fold = len(split_dataset) // self.config.cross_validate_fold + 1
            folds = random_split(split_dataset, tuple([len_per_fold] * (self.config.cross_validate_fold - 1) + [
                len(split_dataset) - len_per_fold * (self.config.cross_validate_fold - 1)]))

            for f_idx in range(self.config.cross_validate_fold):
                train_set = ConcatDataset([x for i, x in enumerate(folds) if i != f_idx])
                val_set = folds[f_idx]
                train_sampler = RandomSampler(train_set if not train_set else train_set)
                val_sampler = SequentialSampler(val_set if not val_set else val_set)
                self.train_dataloaders.append(
                    DataLoader(dataset=train_set, batch_size=self.config.batch_size, sampler=train_sampler))
                self.valid_dataloaders.append(
                    DataLoader(dataset=val_set, batch_size=self.config.batch_size, sampler=val_sampler))
                if self.test_set:
                    self.test_dataloader = DataLoader(dataset=self.test_set, batch_size=self.config.batch_size, shuffle=False)

    def _train(self, criterion):
        self.prepare_dataloader(self.train_set)

        if self.config.warmup_step >= 0:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_dataloaders[0]) * self.config.num_epoch)
            self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)

        if self.valid_dataloaders:
            return self._k_fold_train_and_evaluate(criterion)
        else:
            return self._train_and_evaluate(criterion)

    def _train_and_evaluate(self, criterion):
        global_step = 0
        max_fold_acc = 0
        max_fold_f1 = 0
        save_path = '{0}/{1}_{2}'.format(self.config.model_path_to_save,
                                         self.config.model_name,
                                         self.config.dataset_name
                                         )

        losses = []

        self.config.metrics_of_this_checkpoint = {'acc': 0, 'f1': 0}
        self.config.max_test_metrics = {'max_test_acc': 0, 'max_test_f1': 0}

        self.logger.info("***** Running trainer for Text Classification *****")
        self.logger.info("Training set examples = %d", len(self.train_set))
        if self.test_set:
            self.logger.info("Test set examples = %d", len(self.test_set))
        self.logger.info("Batch size = %d", self.config.batch_size)
        self.logger.info("Num steps = %d", len(self.train_dataloaders[0]) // self.config.batch_size * self.config.num_epoch)
        patience = self.config.patience + self.config.evaluate_begin
        if self.config.log_step < 0:
            self.config.log_step = len(self.train_dataloaders[0]) if self.config.log_step < 0 else self.config.log_step

        for epoch in range(self.config.num_epoch):
            patience -= 1
            iterator = tqdm(self.train_dataloaders[0], postfix='Epoch:{}'.format(epoch))
            for i_batch, sample_batched in enumerate(iterator):
                global_step += 1
                # switch model to train mode, clear gradient accumulators
                self.model.train()
                self.optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.config.device) for col in self.config.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['label'].to(self.config.device)

                if isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    loss = criterion(outputs, targets)

                losses.append(loss.item())
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                if self.config.warmup_step >= 0:
                    with self.warmup_scheduler.dampening():
                        self.lr_scheduler.step()

                # evaluate if test set is available
                if global_step % self.config.log_step == 0:
                    if self.test_dataloader and epoch >= self.config.evaluate_begin:

                        if self.valid_dataloader:
                            test_acc, f1 = self._evaluate_acc_f1(self.valid_dataloader)
                        else:
                            test_acc, f1 = self._evaluate_acc_f1(self.test_dataloader)

                        self.config.metrics_of_this_checkpoint['acc'] = test_acc
                        self.config.metrics_of_this_checkpoint['f1'] = f1

                        if test_acc > max_fold_acc or f1 > max_fold_f1:

                            if test_acc > max_fold_acc:
                                patience = self.config.patience
                                max_fold_acc = test_acc

                            if f1 > max_fold_f1:
                                max_fold_f1 = f1
                                patience = self.config.patience

                            if self.config.model_path_to_save:
                                if not os.path.exists(self.config.model_path_to_save):
                                    os.makedirs(self.config.model_path_to_save)
                                if save_path:
                                    try:
                                        shutil.rmtree(save_path)
                                        # logger.info('Remove sub-optimal trained model:', save_path)
                                    except:
                                        # logger.info('Can not remove sub-optimal trained model:', save_path)
                                        pass
                                save_path = '{0}/{1}_{2}_acc_{3}_f1_{4}/'.format(self.config.model_path_to_save,
                                                                                 self.config.model_name,
                                                                                 self.config.dataset_name,
                                                                                 round(test_acc * 100, 2),
                                                                                 round(f1 * 100, 2)
                                                                                 )

                                if test_acc > self.config.max_test_metrics['max_test_acc']:
                                    self.config.max_test_metrics['max_test_acc'] = test_acc
                                if f1 > self.config.max_test_metrics['max_test_f1']:
                                    self.config.max_test_metrics['max_test_f1'] = f1

                                save_model(self.config, self.model, self.tokenizer, save_path)

                        postfix = ('Epoch:{} | Loss:{:.4f} | Test Acc:{:.2f}(max:{:.2f}) |'
                                   ' Test F1:{:.2f}(max:{:.2f})'.format(epoch,
                                                                        loss.item(),
                                                                        test_acc * 100,
                                                                        max_fold_acc * 100,
                                                                        f1 * 100,
                                                                        max_fold_f1 * 100))
                    else:
                        if self.config.save_mode and epoch >= self.config.evaluate_begin:
                            save_model(self.config, self.model, self.tokenizer, save_path + '_{}/'.format(loss.item()))
                        postfix = 'Epoch:{} | Loss: {} |No evaluation until epoch:{}'.format(epoch, round(loss.item(), 8), self.config.evaluate_begin)

                    iterator.postfix = postfix
                    iterator.refresh()
            if patience < 0:
                break

        if not self.valid_dataloader:
            self.config.MV.add_metric('Max-Test-Acc w/o Valid Set', max_fold_acc * 100)
            self.config.MV.add_metric('Max-Test-F1 w/o Valid Set', max_fold_f1 * 100)

        if self.valid_dataloader:
            print('Loading best model: {} and evaluating on test set ...'.format(save_path))
            self.reload_model(find_file(save_path, '.state_dict'))
            max_fold_acc, max_fold_f1 = self._evaluate_acc_f1(self.test_dataloader)

            self.config.MV.add_metric('Max-Test-Acc', max_fold_acc * 100)
            self.config.MV.add_metric('Max-Test-F1', max_fold_f1 * 100)

        self.logger.info(self.config.MV.summary(no_print=True))

        print('Training finished, we hope you can share your checkpoint_class with everybody, please see:',
              'https://github.com/yangheng95/PyABSA#how-to-share-checkpoints-eg-checkpoints-trained-on-your-custom-dataset-with-community')

        print_args(self.config, self.logger)

        if self.valid_dataloader or self.config.save_mode:
            del self.train_dataloaders
            del self.test_dataloader
            del self.valid_dataloader
            del self.model
            cuda.empty_cache()
            time.sleep(3)
            return save_path
        else:
            # direct return model if do not evaluate
            # if self.config.model_path_to_save:
            #     save_path = '{0}/{1}/'.format(self.config.model_path_to_save,
            #                                   self.config.model_name
            #                                   )
            #     save_model(self.config, self.model, self.tokenizer, save_path)
            del self.train_dataloaders
            del self.test_dataloader
            del self.valid_dataloader
            cuda.empty_cache()
            time.sleep(3)
            return self.model, self.config, self.tokenizer

    def _k_fold_train_and_evaluate(self, criterion):
        fold_test_acc = []
        fold_test_f1 = []

        save_path_k_fold = ''
        max_fold_acc_k_fold = 0

        self.config.metrics_of_this_checkpoint = {'acc': 0, 'f1': 0}
        self.config.max_test_metrics = {'max_test_acc': 0, 'max_test_f1': 0}

        for f, (train_dataloader, valid_dataloader) in enumerate(zip(self.train_dataloaders, self.valid_dataloaders)):
            patience = self.config.patience + self.config.evaluate_begin
            if self.config.log_step < 0:
                self.config.log_step = len(self.train_dataloaders[0]) if self.config.log_step < 0 else self.config.log_step

            self.logger.info("***** Running trainer for Text Classification *****")
            self.logger.info("Training set examples = %d", len(self.train_set))
            if self.test_set:
                self.logger.info("Test set examples = %d", len(self.test_set))
            self.logger.info("Batch size = %d", self.config.batch_size)
            self.logger.info("Num steps = %d", len(train_dataloader) // self.config.batch_size * self.config.num_epoch)
            if len(self.train_dataloaders) > 1:
                self.logger.info('No. {} trainer in {} folds...'.format(f + 1, self.config.cross_validate_fold))
            global_step = 0
            max_fold_acc = 0
            max_fold_f1 = 0
            save_path = '{0}/{1}_{2}'.format(self.config.model_path_to_save,
                                             self.config.model_name,
                                             self.config.dataset_name
                                             )
            for epoch in range(self.config.num_epoch):
                patience -= 1
                iterator = tqdm(train_dataloader, postfix='Epoch:{}'.format(epoch))
                postfix = ''
                for i_batch, sample_batched in enumerate(iterator):
                    global_step += 1
                    # switch model to train mode, clear gradient accumulators
                    self.model.train()
                    self.optimizer.zero_grad()
                    inputs = [sample_batched[col].to(self.config.device) for col in self.config.inputs_cols]
                    with torch.cuda.amp.autocast():

                        if self.config.use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(inputs)
                        else:
                            outputs = self.model(inputs)
                    targets = sample_batched['label'].to(self.config.device)

                    if isinstance(outputs, dict) and 'loss' in outputs:
                            loss = outputs['loss']
                    else:
                            loss = criterion(outputs, targets)

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

                    # evaluate if test set is available
                    if global_step % self.config.log_step == 0:
                        if self.test_dataloader and epoch >= self.config.evaluate_begin:

                            test_acc, f1 = self._evaluate_acc_f1(valid_dataloader)

                            self.config.metrics_of_this_checkpoint['acc'] = test_acc
                            self.config.metrics_of_this_checkpoint['f1'] = f1
                            if test_acc > max_fold_acc or f1 > max_fold_f1:

                                if test_acc > max_fold_acc:
                                    patience = self.config.patience
                                    max_fold_acc = test_acc

                                if f1 > max_fold_f1:
                                    max_fold_f1 = f1
                                    patience = self.config.patience

                                if self.config.model_path_to_save:
                                    if not os.path.exists(self.config.model_path_to_save):
                                        os.makedirs(self.config.model_path_to_save)
                                    if save_path:
                                        try:
                                            shutil.rmtree(save_path)
                                            # logger.info('Remove sub-optimal trained model:', save_path)
                                        except:
                                            # logger.info('Can not remove sub-optimal trained model:', save_path)
                                            pass
                                    save_path = '{0}/{1}_{2}_acc_{3}_f1_{4}/'.format(self.config.model_path_to_save,
                                                                                     self.config.model_name,
                                                                                     self.config.dataset_name,
                                                                                     round(test_acc * 100, 2),
                                                                                     round(f1 * 100, 2)
                                                                                     )

                                    if test_acc > self.config.max_test_metrics['max_test_acc']:
                                        self.config.max_test_metrics['max_test_acc'] = test_acc
                                    if f1 > self.config.max_test_metrics['max_test_f1']:
                                        self.config.max_test_metrics['max_test_f1'] = f1

                                    save_model(self.config, self.model, self.tokenizer, save_path)

                            postfix = ('Epoch:{} | Loss:{:.4f} | Test Acc:{:.2f}(max:{:.2f}) |'
                                       ' Test F1:{:.2f}(max:{:.2f})'.format(epoch,
                                                                            loss.item(),
                                                                            test_acc * 100,
                                                                            max_fold_acc * 100,
                                                                            f1 * 100,
                                                                            max_fold_f1 * 100))
                        else:
                            postfix = 'Epoch:{} | Loss:{} | No evaluation until epoch:{}'.format(epoch, round(loss.item(), 8), self.config.evaluate_begin)

                    iterator.postfix = postfix
                    iterator.refresh()
                if patience < 0:
                    break

            max_fold_acc, max_fold_f1 = self._evaluate_acc_f1(self.test_dataloader)
            if max_fold_acc > max_fold_acc_k_fold:
                save_path_k_fold = save_path
            fold_test_acc.append(max_fold_acc)
            fold_test_f1.append(max_fold_f1)

            self.config.MV.add_metric('Fold{}-Max-Valid-Acc'.format(f), max_fold_acc * 100)
            self.config.MV.add_metric('Fold{}-Max-Valid-F1'.format(f), max_fold_f1 * 100)

            self.logger.info(self.config.MV.summary(no_print=True))
            if os.path.exists('./init_state_dict.bin'):
                self.reload_model()

        max_test_acc = numpy.max(fold_test_acc)
        max_test_f1 = numpy.mean(fold_test_f1)

        self.config.MV.add_metric('Max-Test-Acc', max_test_acc * 100)
        self.config.MV.add_metric('Max-Test-F1', max_test_f1 * 100)

        if self.config.cross_validate_fold > 0:
            self.logger.info(self.config.MV.summary(no_print=True))
        # self.config.MV.summary()

        print('Training finished, we hope you can share your checkpoint_class with community, please see:',
              'https://github.com/yangheng95/PyABSA/blob/release/demos/documents/share-checkpoint.md')

        print_args(self.config, self.logger)

        self.reload_model(save_path_k_fold)

        if self.valid_dataloader or self.config.save_mode:
            del self.train_dataloaders
            del self.test_dataloader
            del self.valid_dataloaders
            del self.model
            cuda.empty_cache()
            time.sleep(3)
            return save_path_k_fold
        else:
            # direct return model if do not evaluate
            if self.config.model_path_to_save:
                save_path_k_fold = '{0}/{1}/'.format(self.config.model_path_to_save,
                                                     self.config.model_name,
                                                     )
                save_model(self.config, self.model, self.tokenizer, save_path_k_fold)
            del self.train_dataloaders
            del self.test_dataloader
            del self.valid_dataloaders
            cuda.empty_cache()
            time.sleep(3)
            return self.model, self.config, self.tokenizer

    def _evaluate_acc_f1(self, test_dataloader):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(test_dataloader):

                t_inputs = [t_sample_batched[col].to(self.config.device) for col in self.config.inputs_cols]
                t_targets = t_sample_batched['label'].to(self.config.device)

                sen_outputs = self.model(t_inputs)
                n_test_correct += (torch.argmax(sen_outputs, -1) == t_targets).sum().item()
                n_test_total += len(sen_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = sen_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, sen_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(),
                              labels=list(range(self.config.output_dim)), average='macro')
        if self.config.args.get('show_metric', False):
            print('\n---------------------------- Classification Report ----------------------------\n')
            print(metrics.classification_report(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), target_names=[self.config.index_to_label[x] for x in self.config.index_to_label]))
            print('\n---------------------------- Classification Report ----------------------------\n')
        return test_acc, f1

    def _init_misc(self):
        random.seed(self.config.seed)
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)

        self.config.inputs_cols = self.model.inputs

        self.config.device = torch.device(self.config.device)

        # use DataParallel for trainer if device count larger than 1
        if self.config.auto_device == DeviceTypeOption.ALL_CUDA:
            self.model.to(self.config.device)
            self.model = torch.nn.parallel.DataParallel(self.model).module
        else:
            self.model.to(self.config.device)

        self.optimizer = init_optimizer(self.config.optimizer)(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2reg
        )

        self.train_dataloaders = []
        self.valid_dataloaders = []

        if os.path.exists('./init_state_dict.bin'):
            os.remove('./init_state_dict.bin')
        if self.config.cross_validate_fold > 0:
            torch.save(self.model.state_dict(), './init_state_dict.bin')

        self.config.device = torch.device(self.config.device)
        if self.config.device.type == 'cuda':
            self.logger.info("cuda memory allocated:{}".format(torch.cuda.memory_allocated(device=self.config.device)))

        print_args(self.config, self.logger)

    def _cache_or_load_dataset(self):
        pass

    def _load_dataset_and_prepare_dataloader(self):

        cache_path = self.load_cache_dataset()

        # init BERT-based model and dataset
        if hasattr(BERTTCModelList, self.config.model.__name__):
            self.tokenizer = PretrainedTokenizer(self.config)
            if cache_path is None or self.config.overwrite_cache:
                self.train_set = BERTTCDataset(self.config, self.tokenizer, dataset_type='train')
                self.test_set = BERTTCDataset(self.config, self.tokenizer, dataset_type='test')
                self.valid_set = BERTTCDataset(self.config, self.tokenizer, dataset_type='valid')
            try:
                self.bert = AutoModel.from_pretrained(self.config.pretrained_bert)
            except ValueError as e:
                print('Init pretrained model failed, exception: {}'.format(e))

            # init the model behind the construction of datasets in case of updating output_dim
            self.model = self.config.model(self.bert, self.config).to(self.config.device)

        elif hasattr(GloVeTCModelList, self.config.model.__name__):
            # init GloVe-based model and dataset
            self.tokenizer = Tokenizer.build_tokenizer(
                config=self.config,
                cache_path='{0}_tokenizer.dat'.format(os.path.basename(self.config.dataset_name))
            )
            self.embedding_matrix = build_embedding_matrix(
                config=self.config,
                tokenizer=self.tokenizer,
                cache_path='{0}_{1}_embedding_matrix.dat'.format(str(self.config.embed_dim), os.path.basename(self.config.dataset_name)),
            )
            self.train_set = GloVeTCDataset(self.config, self.tokenizer, dataset_type='train')
            self.test_set = GloVeTCDataset(self.config, self.tokenizer, dataset_type='test')
            self.valid_set = GloVeTCDataset(self.config, self.tokenizer, dataset_type='valid')

            self.model = self.config.model(self.embedding_matrix, self.config).to(self.config.device)

        self.save_cache_dataset()

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()

        return self._train(criterion)
