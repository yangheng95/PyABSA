# -*- coding: utf-8 -*-
# file: classifier_trainer.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import math
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

from pyabsa.functional.dataset import ClassificationDatasetList
from ..models import GloVeClassificationModelList, BERTClassificationModelList
from ..classic.__bert__.dataset_utils.data_utils_for_training import (Tokenizer4Pretraining,
                                                                      BERTClassificationDataset)
from ..classic.__glove__.dataset_utils.data_utils_for_training import (build_tokenizer,
                                                                       build_embedding_matrix,
                                                                       GloVeClassificationDataset)
from pyabsa.utils.file_utils import save_model
from pyabsa.utils.pyabsa_utils import print_args, resume_from_checkpoint, retry, TransformerConnectionError, optimizers

import pytorch_warmup as warmup

try:
    import apex.amp as amp

    # assert torch.version.__version__ < '1.10.0'
    print('Use FP16 via Apex!')
except Exception:
    amp = None


class Instructor:
    def __init__(self, opt, logger):
        self.logger = logger
        self.opt = opt

        config_str = re.sub(r'<.*?>', '', str(sorted([str(self.opt.args[k]) for k in self.opt.args if k != 'seed'])))
        hash_tag = sha256(config_str.encode()).hexdigest()
        cache_path = '{}.{}.dataset.{}.cache'.format(self.opt.model_name, self.opt.dataset_name, hash_tag)

        if os.path.exists(cache_path):
            print('Loading dataset cache:', cache_path)
            if self.opt.dataset_file['test']:
                self.train_set, self.valid_set, self.test_set, opt = pickle.load(open(cache_path, mode='rb'))
            else:
                self.train_set, opt = pickle.load(open(cache_path, mode='rb'))
            # reset output dim according to dataset labels
            self.opt.polarities_dim = opt.polarities_dim
            self.opt.label_to_index = opt.label_to_index
            self.opt.index_to_label = opt.index_to_label

        # init BERT-based model and dataset
        if hasattr(BERTClassificationModelList, opt.model.__name__):
            self.tokenizer = Tokenizer4Pretraining(self.opt.max_seq_len, self.opt)
            if not os.path.exists(cache_path):
                self.train_set = BERTClassificationDataset(self.opt.dataset_file['train'], self.tokenizer, self.opt)
                if self.opt.dataset_file['test']:
                    self.test_set = BERTClassificationDataset(self.opt.dataset_file['test'], self.tokenizer, self.opt)
                else:
                    self.test_set = None
                if self.opt.dataset_file['valid']:
                    self.valid_set = BERTClassificationDataset(self.opt.dataset_file['valid'], self.tokenizer, self.opt)
                else:
                    self.valid_set = None
            try:
                self.bert = AutoModel.from_pretrained(self.opt.pretrained_bert)
            except ValueError as e:
                print('Init pretrained model failed, exception: {}'.format(e))
                raise TransformerConnectionError()

            # init the model behind the construction of datasets in case of updating polarities_dim
            self.model = self.opt.model(self.bert, self.opt).to(self.opt.device)

        elif hasattr(GloVeClassificationModelList, opt.model.__name__):
            # init GloVe-based model and dataset
            if hasattr(ClassificationDatasetList, opt.dataset_name):
                if not os.path.exists(os.path.join(os.getcwd(), opt.dataset_name)):
                    os.makedirs(os.path.join(os.getcwd(), opt.dataset_name))

            self.tokenizer = build_tokenizer(
                dataset_list=opt.dataset_file,
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(os.path.basename(opt.dataset_name)),
                opt=self.opt
            )
            self.embedding_matrix = build_embedding_matrix(
                word2idx=self.tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), os.path.basename(opt.dataset_name)),
                opt=self.opt
            )
            self.train_set = GloVeClassificationDataset(self.opt.dataset_file['train'], self.tokenizer, self.opt)

            if self.opt.dataset_file['test']:
                self.test_set = GloVeClassificationDataset(self.opt.dataset_file['test'], self.tokenizer, self.opt)
            else:
                self.test_set = None
            if self.opt.dataset_file['valid']:
                self.valid_set = GloVeClassificationDataset(self.opt.dataset_file['valid'], self.tokenizer, self.opt)
            else:
                self.valid_set = None
            self.model = opt.model(self.embedding_matrix, opt).to(opt.device)

        if self.test_set:
            self.test_dataloader = DataLoader(dataset=self.test_set, batch_size=self.opt.batch_size, shuffle=False)
        else:
            self.test_dataloader = None

        if self.valid_set:
            self.valid_dataloader = DataLoader(dataset=self.valid_set, batch_size=self.opt.batch_size, shuffle=False)
        else:
            self.valid_dataloader = None

        if self.opt.cache_dataset and not os.path.exists(cache_path):
            print('Caching dataset... please remove cached dataset if change model or dataset')
            if self.opt.dataset_file['test']:
                pickle.dump((self.train_set, self.valid_set, self.test_set, self.opt), open(cache_path, mode='wb'))
            else:
                pickle.dump((self.train_set, self.opt), open(cache_path, mode='wb'))

        self.optimizer = optimizers[self.opt.optimizer](
            self.model.parameters(),
            lr=self.opt.learning_rate,
            weight_decay=self.opt.l2reg
        )

        self.train_dataloaders = []
        self.val_dataloaders = []

        if os.path.exists('./init_state_dict.bin'):
            os.remove('./init_state_dict.bin')
        if self.opt.cross_validate_fold > 0:
            torch.save(self.model.state_dict(), './init_state_dict.bin')

        if amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

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

    def reload_model(self):
        self.model.load_state_dict(torch.load('./init_state_dict.bin'))
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optimizers[self.opt.optimizer](_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

    def prepare_dataloader(self, train_set):
        if self.opt.cross_validate_fold < 1:
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

        if self.opt.warmup_step >= 0:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_dataloaders[0]) * self.opt.num_epoch)
            self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)

        if self.val_dataloaders:
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
        save_path = '{0}/{1}_{2}'.format(self.opt.model_path_to_save,
                                         self.opt.model_name,
                                         self.opt.dataset_name
                                         )
        self.opt.metrics_of_this_checkpoint = {'acc': 0, 'f1': 0}
        self.opt.max_test_metrics = {'max_test_acc': 0, 'max_test_f1': 0}

        self.logger.info("***** Running training for Text Classification *****")
        self.logger.info("Training set examples = %d", len(self.train_set))
        if self.test_set:
            self.logger.info("Test set examples = %d", len(self.test_set))
        self.logger.info("Batch size = %d", self.opt.batch_size)
        self.logger.info("Num steps = %d", len(self.train_dataloaders[0]) // self.opt.batch_size * self.opt.num_epoch)
        patience = self.opt.patience + self.opt.evaluate_begin
        if self.opt.log_step < 0:
            self.opt.log_step = len(self.train_dataloaders[0]) if self.opt.log_step < 0 else self.opt.log_step

        for epoch in range(self.opt.num_epoch):
            patience -= 1
            iterator = tqdm(self.train_dataloaders[0], postfix='Epoch:{}'.format(epoch))
            for i_batch, sample_batched in enumerate(iterator):
                global_step += 1
                # switch model to training_tutorials mode, clear gradient accumulators
                self.model.train()
                self.optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['label'].to(self.opt.device)

                sen_logits = outputs
                loss = criterion(sen_logits, targets)
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

                # evaluate if test set is available
                if global_step % self.opt.log_step == 0:
                    if self.opt.dataset_file['test'] and epoch >= self.opt.evaluate_begin:

                        if self.valid_dataloader:
                            test_acc, f1 = self._evaluate_acc_f1(self.valid_dataloader)
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

                                if test_acc > self.opt.max_test_metrics['max_test_acc']:
                                    self.opt.max_test_metrics['max_test_acc'] = test_acc
                                if f1 > self.opt.max_test_metrics['max_test_f1']:
                                    self.opt.max_test_metrics['max_test_f1'] = f1

                                save_model(self.opt, self.model, self.tokenizer, save_path)

                        postfix = ('Epoch:{} | Loss:{:.4f} | Test Acc:{:.2f}(max:{:.2f}) |'
                                   ' Test F1:{:.2f}(max:{:.2f})'.format(epoch,
                                                                        loss.item(),
                                                                        test_acc * 100,
                                                                        max_fold_acc * 100,
                                                                        f1 * 100,
                                                                        max_fold_f1 * 100))
                    else:
                        if self.opt.save_mode:
                            save_model(self.opt, self.model, self.tokenizer, save_path + '_{}/'.format(loss.item()))
                        postfix = 'Epoch:{} | Loss: {} |No evaluation until epoch:{}'.format(epoch, round(loss.item(), 8), self.opt.evaluate_begin)

                    iterator.postfix = postfix
                    iterator.refresh()
            if patience < 0:
                break

        if not self.valid_dataloader:
            self.opt.MV.add_metric('Max-Test-Acc w/o Valid Set', max_fold_acc * 100)
            self.opt.MV.add_metric('Max-Test-F1 w/o Valid Set', max_fold_f1 * 100)

        if self.valid_dataloader:
            print('Loading best model: {} and evaluating on test set ...'.format(save_path))
            self.model.load_state_dict(torch.load(find_file(save_path, '.state_dict')))
            max_fold_acc, max_fold_f1 = self._evaluate_acc_f1(self.test_dataloader)

            self.opt.MV.add_metric('Max-Test-Acc', max_fold_acc * 100)
            self.opt.MV.add_metric('Max-Test-F1', max_fold_f1 * 100)

        self.logger.info(self.opt.MV.summary(no_print=True))

        print('Training finished, we hope you can share your checkpoint with everybody, please see:',
              'https://github.com/yangheng95/PyABSA#how-to-share-checkpoints-eg-checkpoints-trained-on-your-custom-dataset-with-community')
        print_args(self.opt, self.logger)

        if self.opt.save_mode:
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
                                              self.opt.model_name,
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
        self.opt.max_test_metrics = {'max_test_acc': 0, 'max_test_f1': 0}

        for f, (train_dataloader, val_dataloader) in enumerate(zip(self.train_dataloaders, self.val_dataloaders)):
            patience = self.opt.patience + self.opt.evaluate_begin
            if self.opt.log_step < 0:
                self.opt.log_step = len(self.train_dataloaders[0]) if self.opt.log_step < 0 else self.opt.log_step

            self.logger.info("***** Running training for Text Classification *****")
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
            save_path = '{0}/{1}_{2}'.format(self.opt.model_path_to_save,
                                             self.opt.model_name,
                                             self.opt.dataset_name
                                             )
            for epoch in range(self.opt.num_epoch):
                patience -= 1
                iterator = tqdm(train_dataloader, postfix='Epoch:{}'.format(epoch))
                postfix = ''
                for i_batch, sample_batched in enumerate(iterator):
                    global_step += 1
                    # switch model to training_tutorials mode, clear gradient accumulators
                    self.model.train()
                    self.optimizer.zero_grad()
                    inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                    outputs = self.model(inputs)
                    targets = sample_batched['label'].to(self.opt.device)

                    sen_logits = outputs
                    loss = criterion(sen_logits, targets)
                    sum_loss += loss.item()
                    if amp:
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                        self.optimizer.step()

                    if self.opt.warm_step >= 0:
                        with self.warmup_scheduler.dampening():
                            self.lr_scheduler.step()

                    # evaluate if test set is available
                    if global_step % self.opt.log_step == 0:
                        if self.opt.dataset_file['test'] and epoch >= self.opt.evaluate_begin:

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

                                    if test_acc > self.opt.max_test_metrics['max_test_acc']:
                                        self.opt.max_test_metrics['max_test_acc'] = test_acc
                                    if f1 > self.opt.max_test_metrics['max_test_f1']:
                                        self.opt.max_test_metrics['max_test_f1'] = f1

                                    save_model(self.opt, self.model, self.tokenizer, save_path)

                            postfix = ('Epoch:{} | Loss:{:.4f} | Test Acc:{:.2f}(max:{:.2f}) |'
                                       ' Test F1:{:.2f}(max:{:.2f})'.format(epoch,
                                                                            loss.item(),
                                                                            test_acc * 100,
                                                                            max_fold_acc * 100,
                                                                            f1 * 100,
                                                                            max_fold_f1 * 100))
                        else:
                            postfix = 'Epoch:{} | Loss:{} | No evaluation until epoch:{}'.format(epoch, round(loss.item(), 8), self.opt.evaluate_begin)

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

            self.opt.MV.add_metric('Fold{}-Max-Valid-Acc'.format(f), max_fold_acc * 100)
            self.opt.MV.add_metric('Fold{}-Max-Valid-F1'.format(f), max_fold_f1 * 100)

            self.logger.info(self.opt.MV.summary(no_print=True))
            if os.path.exists('./init_state_dict.bin'):
                self.reload_model()

        max_test_acc = numpy.max(fold_test_acc)
        max_test_f1 = numpy.mean(fold_test_f1)

        self.opt.MV.add_metric('Max-Test-Acc', max_test_acc * 100)
        self.opt.MV.add_metric('Max-Test-F1', max_test_f1 * 100)

        if self.opt.cross_validate_fold > 0:
            self.logger.info(self.opt.MV.summary(no_print=True))
        # self.opt.MV.summary()

        print('Training finished, we hope you can share your checkpoint with community, please see:',
              'https://github.com/yangheng95/PyABSA/blob/release/demos/documents/share-checkpoint.md')

        print_args(self.opt, self.logger)

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
                                                     self.opt.model_name,
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

                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['label'].to(self.opt.device)

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
                              labels=list(range(self.opt.polarities_dim)), average='macro')
        return test_acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()

        return self._train(criterion)


@retry
def train4classification(opt, from_checkpoint_path, logger):
    random.seed(opt.seed)
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    if hasattr(BERTClassificationModelList, opt.model.__name__):
        opt.inputs_cols = BERTClassificationDataset.bert_baseline_input_colses[opt.model.__name__.lower()]

    elif hasattr(GloVeClassificationModelList, opt.model.__name__):
        opt.inputs_cols = GloVeClassificationDataset.glove_input_colses[opt.model.__name__.lower()]

    opt.device = torch.device(opt.device)

    trainer = Instructor(opt, logger)
    resume_from_checkpoint(trainer, from_checkpoint_path)

    return trainer.run()
