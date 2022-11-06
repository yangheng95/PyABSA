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
import pandas
import torch
import torch.nn as nn
from findfile import find_file
from sklearn import metrics
from torch import cuda
from torch.utils.data import DataLoader, random_split, ConcatDataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoModel

import pytorch_warmup as warmup

from pyabsa import DeviceTypeOption
from pyabsa.framework.instructor_class.instructor_template import BaseTrainingInstructor
from pyabsa.tasks.TextAdversarialDefense.dataset_utils.__classic__.data_utils_for_training import GloVeTADDataset
from pyabsa.tasks.TextAdversarialDefense.dataset_utils.__plm__.data_utils_for_training import BERTTADDataset
from pyabsa.tasks.TextAdversarialDefense.models import BERTTADModelList, GloVeTADModelList
from pyabsa.utils.file_utils.file_utils import save_model
from pyabsa.utils.pyabsa_utils import print_args, init_optimizer
from pyabsa.framework.tokenizer_class.tokenizer_class import PretrainedTokenizer, Tokenizer, build_embedding_matrix


class TADTrainingInstructor(BaseTrainingInstructor):
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
            self.config.logger.info("cuda memory allocated:{}".format(torch.cuda.memory_allocated(device=self.config.device)))

        print_args(self.config, self.config.logger)

    def _cache_or_load_dataset(self):
        pass

    def _load_dataset_and_prepare_dataloader(self):
        config_str = re.sub(r'<.*?>', '', str(sorted([str(self.config.args[k]) for k in self.config.args if k != 'seed'])))
        hash_tag = sha256(config_str.encode()).hexdigest()
        cache_path = '{}.{}.dataset.{}.cache'.format(self.config.model_name, self.config.dataset_name, hash_tag)

        if os.path.exists(cache_path) and not self.config.overwrite_cache:
            print('Loading dataset cache:', cache_path)
            with open(cache_path, mode='rb') as f_cache:
                self.train_set, self.valid_set, self.test_set, self.config = pickle.load(f_cache)

        # init BERT-based model and dataset
        if hasattr(BERTTADModelList, self.config.model.__name__):
            self.tokenizer = PretrainedTokenizer(self.config)
            if not os.path.exists(cache_path) or self.config.overwrite_cache:
                self.train_set = BERTTADDataset(self.config, self.tokenizer, dataset_type='train')
                self.test_set = BERTTADDataset(self.config, self.tokenizer, dataset_type='test')
                self.valid_set = BERTTADDataset(self.config, self.tokenizer, dataset_type='valid')

            try:
                self.bert = AutoModel.from_pretrained(self.config.pretrained_bert)
            except ValueError as e:
                print('Init pretrained model failed, exception: {}'.format(e))

            # init the model behind the construction of datasets in case of updating output_dim
            self.model = self.config.model(self.bert, self.config).to(self.config.device)

        elif hasattr(GloVeTADModelList, self.config.model.__name__):
            # init GloVe-based model and dataset
            self.tokenizer = Tokenizer.build_tokenizer(
                config=self.config,
                cache_path='{0}_tokenizer.dat'.format(os.path.basename(self.config.dataset_name)),
            )
            self.embedding_matrix = build_embedding_matrix(
                config=self.config,
                tokenizer=self.tokenizer,
                cache_path='{0}_{1}_embedding_matrix.dat'.format(str(self.config.embed_dim), os.path.basename(self.config.dataset_name)),
            )
            self.train_set = GloVeTADDataset(self.config, self.tokenizer, dataset_type='train')
            self.test_set = GloVeTADDataset(self.config, self.tokenizer, dataset_type='test')
            self.valid_set = GloVeTADDataset(self.config, self.tokenizer, dataset_type='valid')

            self.model = self.config.model(self.embedding_matrix, self.config).to(self.config.device)

        if self.config.cache_dataset and not os.path.exists(cache_path) or self.config.overwrite_cache:
            print('Caching dataset... please remove cached dataset if change model or dataset')
            with open(cache_path, mode='wb') as f_cache:
                pickle.dump((self.train_set, self.valid_set, self.test_set, self.config), f_cache)

    def __init__(self, config):

        super().__init__(config)

        self._load_dataset_and_prepare_dataloader()

        self._init_misc()

        self.config.pop('dataset_dict', None)

    def reload_model_state_dict(self, ckpt='./init_state_dict.bin'):
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
        max_label_fold_acc = 0
        max_label_fold_f1 = 0
        max_adv_det_fold_acc = 0
        max_adv_det_fold_f1 = 0
        max_adv_tr_fold_acc = 0
        max_adv_tr_fold_f1 = 0

        losses = []

        save_path = '{0}/{1}_{2}'.format(self.config.model_path_to_save,
                                         self.config.model_name,
                                         self.config.dataset_name
                                         )
        self.config.metrics_of_this_checkpoint = {'acc': 0, 'f1': 0}
        self.config.max_test_metrics = {'max_cls_test_acc': 0,
                                        'max_cls_test_f1': 0,
                                        'max_adv_det_test_acc': 0,
                                        'max_adv_det_test_f1': 0,
                                        'max_adv_tr_test_acc': 0,
                                        'max_adv_tr_test_f1': 0,
                                        }

        self.config.logger.info("***** Running trainer for Text Classification with Adversarial Attack Defense *****")
        self.config.logger.info("Training set examples = %d", len(self.train_set))
        if self.test_set:
            self.config.logger.info("Test set examples = %d", len(self.test_set))
        self.config.logger.info("Batch size = %d", self.config.batch_size)
        self.config.logger.info("Num steps = %d", len(self.train_dataloaders[0]) // self.config.batch_size * self.config.num_epoch)
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
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    label_targets = sample_batched['label'].to(self.config.device)
                    adv_tr_targets = sample_batched['adv_train_label'].to(self.config.device)
                    adv_det_targets = sample_batched['is_adv'].to(self.config.device)

                    sen_logits, advdet_logits, adv_tr_logits = outputs['sent_logits'], outputs['advdet_logits'], outputs['adv_tr_logits']
                    sen_loss = criterion(sen_logits, label_targets)
                    adv_det_loss = criterion(advdet_logits, adv_det_targets)
                    adv_train_loss = criterion(adv_tr_logits, adv_tr_targets)
                    loss = sen_loss + self.config.args.get('adv_det_weight', 5) * adv_det_loss + self.config.args.get('adv_train_weight', 5) * adv_train_loss
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
                    if self.test_dataloder and epoch >= self.config.evaluate_begin:

                        if self.valid_dataloader:
                            test_label_acc, test_label_f1, test_adv_det_acc, test_adv_det_f1, test_adv_tr_acc, test_adv_tr_f1 = \
                                self._evaluate_acc_f1(self.valid_dataloader)
                        else:
                            test_label_acc, test_label_f1, test_adv_det_acc, test_adv_det_f1, test_adv_tr_acc, test_adv_tr_f1 = \
                                self._evaluate_acc_f1(self.test_dataloader)

                        self.config.metrics_of_this_checkpoint['max_cls_test_acc'] = test_label_acc
                        self.config.metrics_of_this_checkpoint['max_cls_test_f1'] = test_label_f1
                        self.config.metrics_of_this_checkpoint['max_adv_det_test_acc'] = test_adv_det_acc
                        self.config.metrics_of_this_checkpoint['max_adv_det_test_f1'] = test_adv_det_f1
                        self.config.metrics_of_this_checkpoint['max_adv_tr_test_acc'] = test_adv_tr_acc
                        self.config.metrics_of_this_checkpoint['max_adv_tr_test_f1'] = test_adv_tr_f1

                        if test_label_acc > max_label_fold_acc or test_label_acc > max_label_fold_f1 \
                            or test_adv_det_acc > max_adv_det_fold_acc or test_adv_det_f1 > max_adv_det_fold_f1 \
                            or test_adv_tr_acc > max_adv_tr_fold_acc or test_adv_tr_f1 > max_adv_tr_fold_f1:

                            if test_label_acc > max_label_fold_acc:
                                patience = self.config.patience
                                max_label_fold_acc = test_label_acc

                            if test_label_f1 > max_label_fold_f1:
                                patience = self.config.patience
                                max_label_fold_f1 = test_label_f1

                            if test_adv_det_acc > max_adv_det_fold_acc:
                                patience = self.config.patience
                                max_adv_det_fold_acc = test_adv_det_acc

                            if test_adv_det_f1 > max_adv_det_fold_f1:
                                patience = self.config.patience
                                max_adv_det_fold_f1 = test_adv_det_f1

                            if test_adv_tr_acc > max_adv_tr_fold_acc:
                                patience = self.config.patience
                                max_adv_tr_fold_acc = test_adv_tr_acc

                            if test_adv_tr_f1 > max_adv_tr_fold_f1:
                                patience = self.config.patience
                                max_adv_tr_fold_f1 = test_adv_tr_f1

                            if self.config.model_path_to_save:
                                if not os.path.exists(self.config.model_path_to_save):
                                    os.makedirs(self.config.model_path_to_save)
                                if save_path:
                                    try:
                                        shutil.rmtree(save_path)
                                        # logger.info('Remove sub-configimal trained model:', save_path)
                                    except:
                                        # logger.info('Can not remove sub-configimal trained model:', save_path)
                                        pass
                                save_path = '{0}/{1}_{2}_cls_acc_{3}_cls_f1_{4}_adv_det_acc_{5}_adv_det_f1_{6}' \
                                            '_adv_training_acc_{7}_adv_training_f1_{8}/'.format(self.config.model_path_to_save,
                                                                                                self.config.model_name,
                                                                                                self.config.dataset_name,
                                                                                                round(test_label_acc * 100, 2),
                                                                                                round(test_label_f1 * 100, 2),
                                                                                                round(test_adv_det_acc * 100, 2),
                                                                                                round(test_adv_det_f1 * 100, 2),
                                                                                                round(test_adv_tr_acc * 100, 2),
                                                                                                round(test_adv_tr_f1 * 100, 2),
                                                                                                )

                                if test_label_acc > self.config.max_test_metrics['max_cls_test_acc']:
                                    self.config.max_test_metrics['max_cls_test_acc'] = test_label_acc
                                if test_label_f1 > self.config.max_test_metrics['max_cls_test_f1']:
                                    self.config.max_test_metrics['max_cls_test_f1'] = test_label_f1

                                if test_adv_det_acc > self.config.max_test_metrics['max_adv_det_test_acc']:
                                    self.config.max_test_metrics['max_adv_det_test_acc'] = test_adv_det_acc
                                if test_adv_det_f1 > self.config.max_test_metrics['max_adv_det_test_f1']:
                                    self.config.max_test_metrics['max_adv_det_test_f1'] = test_adv_det_f1

                                if test_adv_tr_acc > self.config.max_test_metrics['max_adv_tr_test_acc']:
                                    self.config.max_test_metrics['max_adv_tr_test_acc'] = test_adv_tr_acc
                                if test_adv_tr_f1 > self.config.max_test_metrics['max_adv_tr_test_f1']:
                                    self.config.max_test_metrics['max_adv_tr_test_f1'] = test_adv_tr_f1

                                save_model(self.config, self.model, self.tokenizer, save_path)

                        postfix = ('Epoch:{} | Loss:{:.4f} | CLS ACC:{:.2f}(max:{:.2f}) | AdvDet ACC:{:.2f}(max:{:.2f})'
                                   ' | AdvCLS ACC:{:.2f}(max:{:.2f})'.format(epoch,
                                                                             sen_loss.item() + adv_det_loss.item() + adv_train_loss.item(),
                                                                             test_label_acc * 100,
                                                                             max_label_fold_acc * 100,
                                                                             test_adv_det_acc * 100,
                                                                             max_adv_det_fold_acc * 100,
                                                                             test_adv_tr_acc * 100,
                                                                             max_adv_tr_fold_acc * 100,
                                                                             ))
                    else:
                        if self.config.save_mode and epoch >= self.config.evaluate_begin:
                            save_model(self.config, self.model, self.tokenizer, save_path + '_{}/'.format(loss.item()))
                        postfix = 'Epoch:{} | Loss: {} |No evaluation until epoch:{}'.format(epoch, round(loss.item(), 8), self.config.evaluate_begin)

                    iterator.postfix = postfix
                    iterator.refresh()
            if patience < 0:
                break

        if not self.valid_dataloader:
            self.config.MV.add_metric('Max-CLS-Acc w/o Valid Set', max_label_fold_acc * 100)
            self.config.MV.add_metric('Max-CLS-F1 w/o Valid Set', max_label_fold_f1 * 100)
            self.config.MV.add_metric('Max-AdvDet-Acc w/o Valid Set', max_adv_det_fold_acc * 100)
            self.config.MV.add_metric('Max-AdvDet-F1 w/o Valid Set', max_adv_det_fold_f1 * 100)
        if self.valid_dataloader:
            print('Loading best model: {} and evaluating on test set ...'.format(save_path))
            self.reload_model_state_dict(find_file(save_path, '.state_dict'))
            max_label_fold_acc, max_label_fold_f1, max_adv_det_fold_acc, max_adv_det_fold_f1, max_adv_tr_fold_acc, max_adv_tr_fold_f1 = \
                self._evaluate_acc_f1(self.test_dataloader)

            self.config.MV.add_metric('Max-CLS-Acc', max_label_fold_acc * 100)
            self.config.MV.add_metric('Max-CLS-F1', max_label_fold_f1 * 100)
            self.config.MV.add_metric('Max-AdvDet-Acc', max_adv_det_fold_acc * 100)
            self.config.MV.add_metric('Max-AdvDet-F1', max_adv_det_fold_f1 * 100)
            self.config.MV.add_metric('Max-AdvCLS-Acc', max_adv_tr_fold_acc * 100)
            self.config.MV.add_metric('Max-AdvCLS-F1', max_adv_tr_fold_f1 * 100)

        self.config.logger.info(self.config.MV.summary(no_print=True))

        print('Training finished, we hope you can share your checkpoint_class with everybody, please see:',
              'https://github.com/yangheng95/PyABSA#how-to-share-checkpoints-eg-checkpoints-trained-on-your-custom-dataset-with-community')

        rolling_intv = 5
        df = pandas.DataFrame(losses)
        losses = list(numpy.hstack(df.rolling(rolling_intv, min_periods=1).mean().values))
        self.config.loss = losses[-1]
        # self.config.loss = np.average(losses)

        print_args(self.config, self.config.logger)

        if self.valid_dataloader or self.config.save_mode:
            del self.train_dataloaders
            del self.test_dataloader
            del self.valid_dataloader
            del self.model
            cuda.empty_cache()
            time.sleep(3)
            return save_path
        else:
            del self.train_dataloaders
            del self.test_dataloader
            del self.valid_dataloader
            cuda.empty_cache()
            time.sleep(3)
            return self.model, self.config, self.tokenizer

    def _k_fold_train_and_evaluate(self, criterion):
        raise NotImplementedError()

    def _evaluate_acc_f1(self, test_dataloader):
        # switch model to evaluation mode
        self.model.eval()
        n_label_test_correct, n_label_test_total = 0, 0
        n_adv_det_test_correct, n_adv_det_test_total = 0, 0
        n_adv_tr_test_correct, n_adv_tr_test_total = 1e-10, 1e-10
        t_label_targets_all, t_label_outputs_all = None, None
        t_adv_det_targets_all, t_adv_det_outputs_all = None, None
        t_adv_tr_targets_all, t_adv_tr_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(test_dataloader):

                t_inputs = [t_sample_batched[col].to(self.config.device) for col in self.config.inputs_cols]
                t_label_targets = t_sample_batched['label'].to(self.config.device)
                t_adv_tr_targets = t_sample_batched['adv_train_label'].to(self.config.device)
                t_adv_det_targets = t_sample_batched['is_adv'].to(self.config.device)

                t_outputs = self.model(t_inputs)
                sent_logits, advdet_logits, adv_tr_logits = t_outputs['sent_logits'], t_outputs['advdet_logits'], t_outputs['adv_tr_logits']

                # --------------------------------------------------------------------------------------------#
                valid_label_targets = torch.tensor([x for x in t_label_targets.cpu() if x != -100]).to(self.config.device)
                if any(valid_label_targets):
                    valid_label_logit_ids = [True if x != -100 else False for x in t_label_targets.cpu()]
                    valid_label_logits = sent_logits[valid_label_logit_ids]

                    n_label_test_correct += (torch.argmax(valid_label_logits, -1) == valid_label_targets).sum().item()
                    n_label_test_total += len(valid_label_logits)

                    if t_label_targets_all is None:
                        t_label_targets_all = valid_label_targets
                        t_label_outputs_all = valid_label_logits
                    else:
                        t_label_targets_all = torch.cat((t_label_targets_all, valid_label_targets), dim=0)
                        t_label_outputs_all = torch.cat((t_label_outputs_all, valid_label_logits), dim=0)

                # --------------------------------------------------------------------------------------------#
                n_adv_det_test_correct += (torch.argmax(advdet_logits, -1) == t_adv_det_targets).sum().item()
                n_adv_det_test_total += len(advdet_logits)

                if t_adv_det_targets_all is None:
                    t_adv_det_targets_all = t_adv_det_targets
                    t_adv_det_outputs_all = advdet_logits
                else:
                    t_adv_det_targets_all = torch.cat((t_adv_det_targets_all, t_adv_det_targets), dim=0)
                    t_adv_det_outputs_all = torch.cat((t_adv_det_outputs_all, advdet_logits), dim=0)

                # --------------------------------------------------------------------------------------------#
                valid_adv_tr_targets = torch.tensor([x for x in t_adv_tr_targets.cpu() if x != -100]).to(self.config.device)
                if any(t_adv_tr_targets):
                    valid_adv_tr_logit_ids = [True if x != -100 else False for x in t_adv_tr_targets.cpu()]
                    valid_adv_tr_logits = adv_tr_logits[valid_adv_tr_logit_ids]

                    n_adv_tr_test_correct += (torch.argmax(valid_adv_tr_logits, -1) == valid_adv_tr_targets).sum().item()
                    n_adv_tr_test_total += len(valid_adv_tr_logits)

                    if t_adv_tr_targets_all is None:
                        t_adv_tr_targets_all = valid_adv_tr_targets
                        t_adv_tr_outputs_all = valid_adv_tr_logits
                    else:
                        t_adv_tr_targets_all = torch.cat((t_adv_tr_targets_all, valid_adv_tr_targets), dim=0)
                        t_adv_tr_outputs_all = torch.cat((t_adv_tr_outputs_all, valid_adv_tr_logits), dim=0)

        label_test_acc = n_label_test_correct / n_label_test_total
        label_test_f1 = metrics.f1_score(t_label_targets_all.cpu(), torch.argmax(t_label_outputs_all, -1).cpu(),
                                         labels=list(range(self.config.class_dim)), average='macro')
        if self.config.args.get('show_metric', False):
            print('\n---------------------------- Standard Classification Report ----------------------------\n')
            print(
                metrics.classification_report(t_label_targets_all.cpu(), torch.argmax(t_label_outputs_all, -1).cpu(), target_names=[self.config.index_to_label[x] for x in self.config.index_to_label]))
            print('\n---------------------------- Standard Classification Report ----------------------------\n')

        adv_det_test_acc = n_adv_det_test_correct / n_adv_det_test_total
        adv_det_test_f1 = metrics.f1_score(t_adv_det_targets_all.cpu(), torch.argmax(t_adv_det_outputs_all, -1).cpu(),
                                           labels=list(range(self.config.adv_det_dim)), average='macro')

        adv_tr_test_acc = n_adv_tr_test_correct / n_adv_tr_test_total
        adv_tr_test_f1 = metrics.f1_score(t_adv_tr_targets_all.cpu(), torch.argmax(t_adv_tr_outputs_all, -1).cpu(),
                                          labels=list(range(self.config.class_dim)), average='macro')

        return label_test_acc, label_test_f1, adv_det_test_acc, adv_det_test_f1, adv_tr_test_acc, adv_tr_test_f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()

        return self._train(criterion)
