# -*- coding: utf-8 -*-
# file: rnac_instructor.py
# time: 03/11/2022 19:46
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import os
import shutil
import time

import numpy
import numpy as np
import sklearn
import torch
import torch.nn as nn
from sklearn import metrics
from torch import cuda
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from pyabsa import DeviceTypeOption
from pyabsa.framework.instructor_class.instructor_template import BaseTrainingInstructor
from pyabsa.utils.file_utils.file_utils import save_model
from pyabsa.utils.pyabsa_utils import init_optimizer, print_args
from ..dataset_utils.data_utils_for_training import GloVeRNACDataset
from ..dataset_utils.data_utils_for_training import BERTRNACDataset
from ..models import GloVeRNACModelList, BERTRNACModelList

from pyabsa.framework.tokenizer_class.tokenizer_class import Tokenizer, build_embedding_matrix, PretrainedTokenizer


class RNACTrainingInstructor(BaseTrainingInstructor):

    def __init__(self, config):
        super().__init__(config)

        self._load_dataset_and_prepare_dataloader()

        self._init_misc()

    def _init_misc(self):
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

        torch.save(self.model.state_dict(), './init_state_dict.bin')

        self.config.device = torch.device(self.config.device)
        if self.config.device.type == 'cuda':
            self.logger.info("cuda memory allocated:{}".format(torch.cuda.memory_allocated(device=self.config.device)))

        print_args(self.config, self.logger)

    def _cache_or_load_dataset(self):
        pass

    def _train_and_evaluate(self, criterion):
        global_step = 0
        max_fold_acc1 = 0
        max_fold_acc2 = 0
        save_path = '{0}/{1}_{2}'.format(self.config.model_path_to_save,
                                         self.config.model_name,
                                         self.config.dataset_name
                                         )

        losses = []

        self.config.metrics_of_this_checkpoint = {'acc': 0, 'f1': 0}
        self.config.max_test_metrics = {'max_test_acc': 0, 'max_test_f1': 0}

        self.logger.info("***** Running trainer for RNA Classification *****")
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

                if self.config.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                label1_targets = sample_batched['label1'].to(self.config.device)
                label2_targets = sample_batched['label2'].to(self.config.device)

                if isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    loss1 = criterion(outputs[0], label1_targets)
                    loss2 = criterion(outputs[1], label2_targets)
                    loss = loss1 + loss2

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

                # evaluate if test set is available
                if global_step % self.config.log_step == 0:
                    if self.test_dataloader and epoch >= self.config.evaluate_begin:

                        if self.valid_dataloader:
                            acc1, f1_1, acc2, f1_2 = self._evaluate_acc_f1(self.valid_dataloader)
                        else:
                            acc1, f1_1, acc2, f1_2 = self._evaluate_acc_f1(self.test_dataloader)

                        self.config.metrics_of_this_checkpoint['decay_acc'] = acc1
                        self.config.metrics_of_this_checkpoint['rna_cls_acc'] = acc2

                        if acc1 > max_fold_acc1 or acc2 > max_fold_acc2:

                            if acc1 > max_fold_acc1:
                                patience = self.config.patience
                                max_fold_acc1 = acc1

                            if acc2 > max_fold_acc2:
                                max_fold_acc2 = acc2
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
                                                                                 round(acc1 * 100, 2),
                                                                                 round(acc2 * 100, 2)
                                                                                 )

                                save_model(self.config, self.model, self.tokenizer, save_path)

                        postfix = ('Epoch:{} | Loss:{:.4f} | DecayAcc:{:.2f}(max:{:.2f}) | RNACLS Acc:{:.2f}(max:{:.2f})'.format(epoch,
                                                                                                                                            loss.item(),
                                                                                                                                            acc1 * 100,
                                                                                                                                            max_fold_acc1 * 100,
                                                                                                                                            acc2 * 100,
                                                                                                                                            max_fold_acc2 * 100))
                    else:
                        if self.config.save_mode and epoch >= self.config.evaluate_begin:
                            save_model(self.config, self.model, self.tokenizer, save_path + '_{}/'.format(loss.item()))
                        postfix = 'Epoch:{} | Loss: {} |No evaluation until epoch:{}'.format(epoch, round(loss.item(), 8), self.config.evaluate_begin)

                    iterator.postfix = postfix
                    iterator.refresh()
            if patience < 0:
                break

        if not self.valid_dataloader:
            self.config.MV.add_metric('Max-Decay-Acc w/o Valid Set', max_fold_acc1 * 100)
            self.config.MV.add_metric('Max-RNACLS-Acc w/o Valid Set', max_fold_acc2 * 100)

        if self.valid_dataloader:
            print('Loading best model: {} and evaluating on test set ...'.format(save_path))
            self._reload_model_state_dict(save_path)
            max_fold_acc_1, max_fold_f1_1, max_fold_acc_2, max_fold_f1_2 = self._evaluate_acc_f1(self.test_dataloader)

            self.config.MV.add_metric('Max-Test-Decay-Acc', max_fold_acc_1 * 100)
            self.config.MV.add_metric('Max-Test-RNACLS-Acc', max_fold_acc_2 * 100)

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
                    if self.config.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(inputs)
                    else:
                        outputs = self.model(inputs)

                    label1_targets = sample_batched['label1'].to(self.config.device)
                    label2_targets = sample_batched['label2'].to(self.config.device)

                    if isinstance(outputs, dict) and 'loss' in outputs:
                        loss = outputs['loss']
                    else:
                        loss = criterion(outputs[0], label1_targets) + criterion(outputs[0], label2_targets)

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
                        if self.valid_dataloader and epoch >= self.config.evaluate_begin:

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
            self._reload_model_state_dict('./init_state_dict.bin')

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

        self._reload_model_state_dict(save_path_k_fold)

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
        t_targets_all1, t_outputs_all1 = [], []
        t_outputs_all2, t_targets_all2 = [], []
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(test_dataloader):
                t_inputs = [t_sample_batched[col].to(self.config.device) for col in self.config.inputs_cols]
                t_targets1 = t_sample_batched['label1'].to(self.config.device)
                t_targets2 = t_sample_batched['label2'].to(self.config.device)

                outputs = self.model(t_inputs)

                t_outputs_all1.extend(outputs[0].cpu().numpy().tolist())
                t_outputs_all2.extend(outputs[1].cpu().numpy().tolist())
                t_targets_all1.extend(t_targets1.cpu().numpy().tolist())
                t_targets_all2.extend(t_targets2.cpu().numpy().tolist())

        f1_1 = metrics.f1_score(t_targets_all1, np.argmax(t_outputs_all1, axis=-1), average='macro')

        acc_1 = metrics.accuracy_score(t_targets_all1, np.argmax(t_outputs_all1, axis=-1))

        f1_2 = metrics.f1_score(t_targets_all2, np.argmax(t_outputs_all2, axis=-1), average='macro')

        acc_2 = metrics.accuracy_score(t_targets_all2, np.argmax(t_outputs_all2, axis=-1))

        if self.config.args.get('show_metric', False):
            print('\n---------------------------- Classification Report ----------------------------\n')
            print(metrics.classification_report(t_targets_all1, np.argmax(t_outputs_all1, -1), digits=4,
                                                target_names=[str(self.config.index_to_label1[x]) for x in self.config.index_to_label1]))
            print('\n-------------------------------------------------------------------------------\n')
            print(metrics.classification_report(t_targets_all2, np.argmax(t_outputs_all2, -1), digits=4,
                                                target_names=[str(self.config.index_to_label2[x]) for x in self.config.index_to_label2]))

            print('\n---------------------------- Classification Report ----------------------------\n')
        return acc_1, f1_1, acc_2, f1_2

    def _load_dataset_and_prepare_dataloader(self):
        self.config.inputs_cols = self.config.model.inputs

        cache_path = self.load_cache_dataset()
        # init BERT-based model and dataset
        if hasattr(BERTRNACModelList, self.config.model.__name__):
            self.tokenizer = PretrainedTokenizer(self.config)
            if cache_path is None or self.config.overwrite_cache:
                self.train_set = BERTRNACDataset(self.config, self.tokenizer, dataset_type='train')
                self.test_set = BERTRNACDataset(self.config, self.tokenizer, dataset_type='test')
                self.valid_set = BERTRNACDataset(self.config, self.tokenizer, dataset_type='valid')
            try:
                self.bert = AutoModel.from_pretrained(self.config.pretrained_bert)
            except ValueError as e:
                print('Init pretrained model failed, exception: {}'.format(e))

            # init the model behind the construction of datasets in case of updating output_dim
            self.model = self.config.model(self.bert, self.config).to(self.config.device)

        elif hasattr(GloVeRNACModelList, self.config.model.__name__):
            # init GloVe-based model and dataset
            self.tokenizer = Tokenizer.build_tokenizer(
                config=self.config,
                cache_path='{0}_tokenizer.dat'.format(os.path.basename(self.config.dataset_name)),
                pre_tokenizer=AutoTokenizer.from_pretrained(self.config.pretrained_bert)
            )
            self.embedding_matrix = build_embedding_matrix(
                config=self.config,
                tokenizer=self.tokenizer,
                cache_path='{0}_{1}_embedding_matrix.dat'.format(str(self.config.embed_dim), os.path.basename(self.config.dataset_name)),
            )
            self.train_set = GloVeRNACDataset(self.config, self.tokenizer, dataset_type='train')
            self.test_set = GloVeRNACDataset(self.config, self.tokenizer, dataset_type='test')
            self.valid_set = GloVeRNACDataset(self.config, self.tokenizer, dataset_type='valid')

            self.model = self.config.model(self.embedding_matrix, self.config).to(self.config.device)
            self.config.tokenizer = self.tokenizer
            self.config.embedding_matrix = self.embedding_matrix

        self.save_cache_dataset()

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()

        return self._train(criterion)
