# -*- coding: utf-8 -*-
# file: apc_trainer.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import math
import os
import random
import shutil
import time

import numpy
import torch
import torch.nn as nn
from findfile import find_file
from sklearn import metrics
from torch import cuda
from torch.utils.data import DataLoader, random_split, ConcatDataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import BertModel

from pyabsa.utils.file_utils import save_model
from pyabsa.utils.pyabsa_utils import print_args, optimizers, resume_from_checkpoint, retry

from ..models.ensembler import APCEnsembler

try:
    import apex.amp as amp

    assert torch.version.__version__ < '1.10.0'
    print('Use FP16 via Apex!')
except Exception:
    amp = None


class Instructor:
    def __init__(self, opt, logger):
        self.logger = logger
        self.opt = opt

        self.logger = logger

        self.model = APCEnsembler(self.opt)
        self.opt = self.model.opt
        self.train_set = self.model.train_set
        self.test_set = self.model.test_set
        self.test_dataloader = self.model.test_dataloader
        self.val_dataloader = self.model.val_dataloader
        self.train_dataloader = self.model.train_dataloader
        self.tokenizer = self.model.tokenizer

        if 'patience' in self.opt.args and self.opt.patience:
            self.opt.patience = self.opt.patience
        else:
            self.opt.patience = 999999999

        # use DataParallel for training if device count larger than 1
        if self.opt.auto_device == 'allcuda':
            self.model.to(self.opt.device)
            self.model = torch.nn.parallel.DataParallel(self.model)
        else:
            self.model.to(self.opt.device)

        self.opt.device = torch.device(self.opt.device)
        if self.opt.device.type == 'cuda':
            self.logger.info("cuda memory allocated:{}".format(torch.cuda.memory_allocated(device=self.opt.device)))

        print_args(self.opt, self.logger)

        initializers = {
            'xavier_uniform_': torch.nn.init.xavier_uniform_,
            'xavier_normal_': torch.nn.init.xavier_normal_,
            'orthogonal_': torch.nn.init.orthogonal_,
        }
        self.initializer = initializers[self.opt.initializer]

        self.optimizer = optimizers[self.opt.optimizer](
            self.model.parameters(),
            lr=self.opt.learning_rate,
            weight_decay=self.opt.l2reg
        )
        self.train_dataloaders = []
        self.val_dataloaders = []

        if amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

        if os.path.exists('init_state_dict.bin'):
            os.remove('init_state_dict.bin')
        if self.opt.cross_validate_fold > 0:
            torch.save(self.model.state_dict(), 'init_state_dict.bin')

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def reload_model(self):
        self.model.load_state_dict(torch.load('./init_state_dict.bin'))
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optimizers[self.opt.optimizer](_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

    def prepare_dataloader(self, train_set):
        if self.train_dataloader and self.val_dataloader:
            self.val_dataloaders = [self.val_dataloader]
            self.train_dataloaders = [self.train_dataloader]

        elif self.opt.cross_validate_fold < 1:
            train_sampler = RandomSampler(self.train_set if not self.train_set else self.train_set)
            self.train_dataloaders.append(DataLoader(dataset=train_set,
                                                     batch_size=self.opt.batch_size,
                                                     sampler=train_sampler,
                                                     pin_memory=True))

        else:
            split_dataset = train_set
            len_per_fold = len(split_dataset) // self.opt.cross_validate_fold + 1
            folds = random_split(split_dataset, tuple([len_per_fold] * (self.opt.cross_validate_fold - 1) + [
                len(split_dataset) - len_per_fold * (self.opt.cross_validate_fold - 1)]))

            for f_idx in range(self.opt.cross_validate_fold):
                train_set = ConcatDataset([x for i, x in enumerate(folds) if i != f_idx])
                val_set = folds[f_idx]
                train_sampler = RandomSampler(train_set if not train_set else train_set)
                val_sampler = SequentialSampler(val_set if not val_set else val_set)
                self.train_dataloaders.append(
                    DataLoader(dataset=train_set, batch_size=self.opt.batch_size, sampler=train_sampler))
                self.val_dataloaders.append(
                    DataLoader(dataset=val_set, batch_size=self.opt.batch_size, sampler=val_sampler))

    def _train(self, criterion):
        self.prepare_dataloader(self.train_set)
        if len(self.val_dataloaders) > 1:
            return self._k_fold_train_and_evaluate(criterion)
        else:
            return self._train_and_evaluate(criterion)

    def _train_and_evaluate(self, criterion):
        sum_loss = 0
        sum_acc = 0
        sum_f1 = 0

        global_step = 0
        max_fold_acc = 0
        max_fold_f1 = 0
        save_path = ''
        self.opt.metrics_of_this_checkpoint = {'acc': 0, 'f1': 0}
        self.opt.max_test_metrics = {'max_apc_test_acc': 0, 'max_apc_test_f1': 0}

        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0

        for param in self.model.parameters():
            mulValue = numpy.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
            Total_params += mulValue  # 总参数量
            if param.requires_grad:
                Trainable_params += mulValue  # 可训练参数量
            else:
                NonTrainable_params += mulValue  # 非可训练参数量

        patience = self.opt.patience + self.opt.evaluate_begin
        if self.opt.log_step < 0:
            self.opt.log_step = len(self.train_dataloaders[0]) if self.opt.log_step < 0 else self.opt.log_step

        self.logger.info("***** Running training for Aspect Polarity Classification *****")
        self.logger.info("Training set examples = %d", len(self.train_set))
        if self.test_set:
            self.logger.info("Test set examples = %d", len(self.test_set))
        self.logger.info("Total params = %d, Trainable params = %d, Non-trainable params = %d", Total_params,
                         Trainable_params, NonTrainable_params)
        self.logger.info("Batch size = %d", self.opt.batch_size)
        self.logger.info("Num steps = %d", len(self.train_dataloaders[0]) // self.opt.batch_size * self.opt.num_epoch)
        postfix = ''
        for epoch in range(self.opt.num_epoch):
            patience -= 1
            iterator = tqdm(self.train_dataloaders[0])
            for i_batch, sample_batched in enumerate(iterator):
                global_step += 1
                # switch model to training_tutorials mode, clear gradient accumulators
                self.model.train()
                self.optimizer.zero_grad()
                inputs = {col: sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols}
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)

                sen_logits = outputs['logits']
                loss = criterion(sen_logits, targets)
                if isinstance(outputs, dict) and 'loss' in outputs:
                    loss += torch.sum(outputs['loss'])

                if self.opt.auto_device == 'allcuda':
                    loss = loss.mean()

                sum_loss += loss.item()
                if amp:
                    self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                    self.optimizer.step()

                # evaluate if test set is available
                if self.opt.dataset_file['test'] and global_step % self.opt.log_step == 0:
                    if epoch >= self.opt.evaluate_begin:
                        if self.val_dataloaders:
                            test_acc, f1 = self._evaluate_acc_f1(self.val_dataloaders[0])
                        else:
                            test_acc, f1 = self._evaluate_acc_f1(self.test_dataloader)
                        self.opt.metrics_of_this_checkpoint['acc'] = test_acc
                        self.opt.metrics_of_this_checkpoint['f1'] = f1

                        sum_acc += test_acc
                        sum_f1 += f1

                        if test_acc > max_fold_acc or f1 > max_fold_f1:

                            if test_acc > max_fold_acc:
                                patience = self.opt.patience
                                max_fold_acc = test_acc

                            if f1 > max_fold_f1:
                                max_fold_f1 = f1
                                patience = self.opt.patience

                            if self.opt.model_path_to_save:
                                if not os.path.exists(self.opt.model_path_to_save):
                                    os.makedirs(self.opt.model_path_to_save)
                                if save_path:
                                    try:
                                        shutil.rmtree(save_path)
                                        # logger.info('Remove sub-optimal trained model:', save_path)
                                    except:
                                        # logger.info('Can not remove sub-optimal trained model:', save_path)
                                        pass
                                save_path = '{0}/{1}_{2}_acc_{3}_f1_{4}/'.format(self.opt.model_path_to_save,
                                                                                 self.opt.model_name,
                                                                                 self.opt.dataset_name,
                                                                                 round(test_acc * 100, 2),
                                                                                 round(f1 * 100, 2)
                                                                                 )

                                if test_acc > self.opt.max_test_metrics['max_apc_test_acc']:
                                    self.opt.max_test_metrics['max_apc_test_acc'] = test_acc
                                if f1 > self.opt.max_test_metrics['max_apc_test_f1']:
                                    self.opt.max_test_metrics['max_apc_test_f1'] = f1

                                save_model(self.opt, self.model, self.tokenizer, save_path)

                        postfix = ('Epoch:{} | Loss:{:.4f} | Test Acc:{:.2f}(max:{:.2f}) |'
                                   ' Test F1:{:.2f}(max:{:.2f})'.format(epoch,
                                                                        loss.item(),
                                                                        test_acc * 100,
                                                                        max_fold_acc * 100,
                                                                        f1 * 100,
                                                                        max_fold_f1 * 100))
                    else:
                        postfix = 'Epoch:{} | Loss: {} | No evaluation until epoch:{}'.format(epoch, round(loss.item(), 8), self.opt.evaluate_begin)

                iterator.postfix = postfix
                iterator.refresh()
            if patience < 0:
                break

        if not self.val_dataloaders:
            self.opt.MV.add_metric('Max-Test-Acc w/o Valid Set', max_fold_acc * 100)
            self.opt.MV.add_metric('Max-Test-F1 w/o Valid Set', max_fold_f1 * 100)

        if self.val_dataloaders:
            print('Loading best model: {} and evaluating on test set ...'.format(save_path))
            self.model.load_state_dict(torch.load(find_file(save_path, '.state_dict')))
            max_fold_acc, max_fold_f1 = self._evaluate_acc_f1(self.test_dataloader)

            self.opt.MV.add_metric('Max-Test-Acc', max_fold_acc * 100)
            self.opt.MV.add_metric('Max-Test-F1', max_fold_f1 * 100)
            # shutil.rmtree(save_path)

        self.logger.info(self.opt.MV.summary(no_print=True))
        # self.logger.info('-------------------------- Training Summary --------------------------')
        # self.logger.info('Acc: {:.8f} F1: {:.8f} Accumulated Loss: {:.8f}'.format(
        #     max_fold_acc * 100,
        #     max_fold_f1 * 100,
        #     sum_loss)
        # )
        # self.logger.info('-------------------------- Training Summary --------------------------')

        print('Training finished, we hope you can share your checkpoint with community, please see:',
              'https://github.com/yangheng95/PyABSA/blob/release/demos/documents/share-checkpoint.md')

        if save_path:
            del self.train_dataloaders
            del self.test_dataloader
            del self.val_dataloaders
            del self.model
            cuda.empty_cache()
            time.sleep(3)
            return save_path
        else:
            # direct return model if do not evaluate
            if self.opt.model_path_to_save:
                save_path = '{0}/{1}/'.format(self.opt.model_path_to_save,
                                              self.opt.model_name
                                              )
                save_model(self.opt, self.model, self.tokenizer, save_path)
            del self.train_dataloaders
            del self.test_dataloader
            del self.val_dataloaders
            cuda.empty_cache()
            time.sleep(3)
            return self.model, self.opt, self.tokenizer, sum_acc, sum_f1

    def _k_fold_train_and_evaluate(self, criterion):
        sum_loss = 0
        sum_acc = 0
        sum_f1 = 0

        fold_test_acc = []
        fold_test_f1 = []

        save_path_k_fold = ''
        max_fold_acc_k_fold = 0

        self.opt.metrics_of_this_checkpoint = {'acc': 0, 'f1': 0}
        self.opt.max_test_metrics = {'max_apc_test_acc': 0, 'max_apc_test_f1': 0}

        for f, (train_dataloader, val_dataloader) in enumerate(zip(self.train_dataloaders, self.val_dataloaders)):
            patience = self.opt.patience + self.opt.evaluate_begin
            if self.opt.log_step < 0:
                self.opt.log_step = len(self.train_dataloaders[0]) if self.opt.log_step < 0 else self.opt.log_step

            self.logger.info("***** Running training for Aspect Polarity Classification *****")
            self.logger.info("Training set examples = %d", len(self.train_set))
            if self.test_set:
                self.logger.info("Test set examples = %d", len(self.test_set))
            self.logger.info("Batch size = %d", self.opt.batch_size)
            self.logger.info("Num steps = %d", len(train_dataloader) // self.opt.batch_size * self.opt.num_epoch)
            if len(self.train_dataloaders) > 1:
                self.logger.info('No. {} training in {} folds...'.format(f + 1, self.opt.cross_validate_fold))
            global_step = 0
            max_fold_acc = 0
            max_fold_f1 = 0
            save_path = ''
            for epoch in range(self.opt.num_epoch):
                patience -= 1
                iterator = tqdm(train_dataloader)
                postfix = ''
                for i_batch, sample_batched in enumerate(iterator):
                    global_step += 1
                    # switch model to train mode, clear gradient accumulators
                    self.model.train()
                    self.optimizer.zero_grad()
                    inputs = {col: sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols}
                    outputs = self.model(inputs)
                    targets = sample_batched['polarity'].to(self.opt.device)

                    sen_logits = outputs['logits']
                    loss = criterion(sen_logits, targets)
                    if 'loss' in outputs:
                        loss += torch.sum(outputs['loss'])

                    if self.opt.auto_device == 'allcuda':
                        loss = loss.mean()

                    sum_loss += loss.item()
                    if amp:
                        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                        self.optimizer.step()

                    # evaluate if test set is available
                    if self.opt.dataset_file['test'] and global_step % self.opt.log_step == 0:
                        if epoch >= self.opt.evaluate_begin:

                            test_acc, f1 = self._evaluate_acc_f1(val_dataloader)

                            self.opt.metrics_of_this_checkpoint['acc'] = test_acc
                            self.opt.metrics_of_this_checkpoint['f1'] = f1

                            sum_acc += test_acc
                            sum_f1 += f1

                            if test_acc > max_fold_acc or f1 > max_fold_f1:

                                if test_acc > max_fold_acc:
                                    patience = self.opt.patience
                                    max_fold_acc = test_acc

                                if f1 > max_fold_f1:
                                    max_fold_f1 = f1
                                    patience = self.opt.patience

                                if self.opt.model_path_to_save:
                                    if not os.path.exists(self.opt.model_path_to_save):
                                        os.makedirs(self.opt.model_path_to_save)
                                    if save_path:
                                        try:
                                            shutil.rmtree(save_path)
                                            # logger.info('Remove sub-optimal trained model:', save_path)
                                        except:
                                            # logger.info('Can not remove sub-optimal trained model:', save_path)
                                            pass
                                    save_path = '{0}/{1}_{2}_acc_{3}_f1_{4}/'.format(self.opt.model_path_to_save,
                                                                                     self.opt.model_name,
                                                                                     self.opt.dataset_name,
                                                                                     round(test_acc * 100, 2),
                                                                                     round(f1 * 100, 2)
                                                                                     )

                                    if test_acc > self.opt.max_test_metrics['max_apc_test_acc']:
                                        self.opt.max_test_metrics['max_apc_test_acc'] = test_acc
                                    if f1 > self.opt.max_test_metrics['max_apc_test_f1']:
                                        self.opt.max_test_metrics['max_apc_test_f1'] = f1

                                    save_model(self.opt, self.model, self.tokenizer, save_path)
                            postfix = ('Epoch:{} | Loss:{:.4f} | Test Acc:{:.2f}(max:{:.2f}) |'
                                       ' Test F1:{:.2f}(max:{:.2f})'.format(epoch,
                                                                            loss.item(),
                                                                            test_acc * 100,
                                                                            max_fold_acc * 100,
                                                                            f1 * 100,
                                                                            max_fold_f1 * 100))
                        else:
                            postfix = 'Epoch:{} | Loss: {} | No evaluation until epoch:{}'.format(epoch, round(loss.item(), 8), self.opt.evaluate_begin)

                    iterator.postfix = postfix
                    iterator.refresh()
                if patience < 0:
                    break
            self.model.load_state_dict(torch.load(find_file(save_path, '.state_dict')))
            max_fold_acc, max_fold_f1 = self._evaluate_acc_f1(self.test_dataloader)
            if max_fold_acc > max_fold_acc_k_fold:
                save_path_k_fold = save_path
            fold_test_acc.append(max_fold_acc)
            fold_test_f1.append(max_fold_f1)

            self.opt.MV.add_metric('Fold{}-Max-Test-Acc'.format(f), max_fold_acc * 100)
            self.opt.MV.add_metric('Fold{}-Max-Test-F1'.format(f), max_fold_f1 * 100)

            # self.logger.info('-------------------------- Training Summary --------------------------')
            # self.logger.info('Acc: {:.8f} F1: {:.8f} Accumulated Loss: {:.8f}'.format(
            #     max_fold_acc * 100,
            #     max_fold_f1 * 100,
            #     sum_loss)
            # )
            # self.logger.info('-------------------------- Training Summary --------------------------')

            self.logger.info(self.opt.MV.summary(no_print=True))

            if os.path.exists('./init_state_dict.bin'):
                self.reload_model()

        max_test_acc = numpy.max(fold_test_acc)
        max_test_f1 = numpy.max(fold_test_f1)

        self.opt.MV.add_metric('Max-Test-Acc', max_test_acc * 100)
        self.opt.MV.add_metric('Max-Test-F1', max_test_f1 * 100)

        # if self.opt.cross_validate_fold > 0:
        #     self.logger.info('-------------------------- Training Summary --------------------------')
        #     self.logger.info('{}-fold Best Test Acc: {:.8f} Avg F1: {:.8f} Accumulated Loss: {:.8f}'.format(
        #         self.opt.cross_validate_fold,
        #         mean_test_acc * 100,
        #         mean_test_f1 * 100,
        #         sum_loss)
        #     )
        #     self.logger.info('-------------------------- Training Summary --------------------------')

        self.logger.info(self.opt.MV.summary(no_print=True))

        print('Training finished, we hope you can share your checkpoint with everybody, please see:',
              'https://github.com/yangheng95/PyABSA#how-to-share-checkpoints-eg-checkpoints-trained-on-your-custom-dataset-with-community')

        if os.path.exists('./init_state_dict.bin'):
            self.reload_model()
            os.remove('./init_state_dict.bin')
        if save_path_k_fold:
            del self.train_dataloaders
            del self.test_dataloader
            del self.val_dataloaders
            del self.model
            cuda.empty_cache()
            time.sleep(3)
            return save_path_k_fold
        else:
            # direct return model if do not evaluate
            if self.opt.model_path_to_save:
                save_path_k_fold = '{0}/{1}/'.format(self.opt.model_path_to_save,
                                                     self.opt.model_name
                                                     )
                save_model(self.opt, self.model, self.tokenizer, save_path_k_fold)
                del self.train_dataloaders
                del self.test_dataloader
                del self.val_dataloaders
                cuda.empty_cache()
                time.sleep(3)
            return self.model, self.opt, self.tokenizer, sum_acc, sum_f1

    def _evaluate_acc_f1(self, test_dataloader):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(test_dataloader):

                t_inputs = {col: t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols}

                t_targets = t_sample_batched['polarity'].to(self.opt.device)

                t_outputs = self.model(t_inputs)

                if isinstance(t_outputs, dict):
                    sen_outputs = t_outputs['logits']
                else:
                    sen_outputs = t_outputs

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
                              labels=list(range(self.opt.polarities_dim)), average='macro')
        return test_acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        return self._train(criterion)


@retry
def train4apc(opt, from_checkpoint_path, logger):
    random.seed(opt.seed)
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    opt.device = torch.device(opt.device)

    # in case of handling ConnectionError exception
    trainer = Instructor(opt, logger)
    resume_from_checkpoint(trainer, from_checkpoint_path)

    return trainer.run()
