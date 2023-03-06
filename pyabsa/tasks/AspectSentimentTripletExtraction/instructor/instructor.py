# -*- coding: utf-8 -*-
# file: apc_instructor.py
# time: 2021/4/22 0022
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os
import pickle
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
from torch import cuda
from torch.utils.data import random_split, ConcatDataset

from pyabsa.tasks.AspectSentimentTripletExtraction.dataset_utils.aste_utils import (
    DataIterator,
    Metric,
)
from pyabsa.utils.pyabsa_utils import fprint, init_optimizer

from pyabsa.utils.file_utils.file_utils import save_model

from pyabsa.framework.flag_class import DeviceTypeOption
from tqdm import tqdm

from pyabsa.framework.tokenizer_class.tokenizer_class import PretrainedTokenizer

from pyabsa.framework.instructor_class.instructor_template import BaseTrainingInstructor
from pyabsa.tasks.AspectSentimentTripletExtraction.dataset_utils.data_utils_for_training import (
    ASTEDataset,
)
from pyabsa.tasks.AspectSentimentTripletExtraction.models.model import EMCGCN
import pytorch_warmup as warmup


class ASTETrainingInstructor(BaseTrainingInstructor):
    def _load_dataset_and_prepare_dataloader(self):
        cache_path = self.load_cache_dataset()

        # init BERT-based model and dataset
        self.tokenizer = PretrainedTokenizer(self.config)

        if not os.path.exists(cache_path) or self.config.overwrite_cache:
            self.train_set = ASTEDataset(
                self.config, self.tokenizer, dataset_type="train"
            )
            self.test_set = ASTEDataset(
                self.config, self.tokenizer, dataset_type="test"
            )
            self.valid_set = ASTEDataset(
                self.config, self.tokenizer, dataset_type="valid"
            )
            # You can not call convert_examples_to_features unless train_set, test_set, valid_set are initialized
            self.train_set.convert_examples_to_features()
            self.test_set.convert_examples_to_features()
            self.valid_set.convert_examples_to_features()

            self.save_cache_dataset(cache_path)
        else:
            fprint("Loading dataset from cache file: %s" % cache_path)
            with open(cache_path, "rb") as cache_path:
                (
                    self.train_set,
                    self.test_set,
                    self.valid_set,
                    self.config,
                ) = pickle.load(cache_path)

        self.model = self.config.model(config=self.config).to(self.config.device)

        self.config.tokenizer = self.tokenizer

    def __init__(self, config):
        super().__init__(config)

        self._load_dataset_and_prepare_dataloader()

        self._init_misc()

    def _prepare_dataloader(self):
        """
        Prepares the data loaders for training, validation, and testing.
        Special for ASTE, do not use the default data loader
        """
        random.shuffle(self.train_set.data)

        if self.config.cross_validate_fold < 1:
            # Single dataset, no cross-validation
            self.train_dataloaders.append(
                DataIterator(
                    self.train_set,
                    config=self.config,
                )
            )

            # Set up the validation dataloader
            if self.valid_set and not self.valid_dataloader:
                self.valid_dataloaders.append(
                    DataIterator(
                        self.valid_set,
                        config=self.config,
                    )
                )

            # Set up the testing dataloader
            if self.test_set and not self.test_dataloader:
                self.test_dataloader = DataIterator(
                    self.test_set,
                    config=self.config,
                )

        # Cross-validation
        else:
            split_dataset = self.train_set
            len_per_fold = len(split_dataset) // self.config.cross_validate_fold + 1
            # Split the dataset into folds
            folds = random_split(
                split_dataset,
                tuple(
                    [len_per_fold] * (self.config.cross_validate_fold - 1)
                    + [
                        len(split_dataset)
                        - len_per_fold * (self.config.cross_validate_fold - 1)
                    ]
                ),
            )

            # Set up dataloaders for each fold
            for f_idx in range(self.config.cross_validate_fold):
                train_set = ConcatDataset(
                    [x for i, x in enumerate(folds) if i != f_idx]
                )
                val_set = folds[f_idx]
                self.train_dataloaders.append(
                    DataIterator(
                        train_set,
                        config=self.config,
                    )
                )
                self.valid_dataloaders.append(
                    DataIterator(
                        val_set,
                        config=self.config,
                    )
                )

    def _train_and_evaluate(self, criterion):
        global_step = 0
        max_fold_f1 = -1
        save_path = "{0}/{1}_{2}".format(
            self.config.model_path_to_save,
            self.config.model_name,
            self.config.dataset_name,
        )

        self.config.metrics_of_this_checkpoint = {"f1": 0}
        self.config.max_test_metrics = {"max_apc_test_f1": 0}

        losses = []

        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0

        for param in self.model.parameters():
            mulValue = np.prod(param.size())  # 使用np prod接口计算参数数组所有元素之积
            Total_params += mulValue  # 总参数量
            if param.requires_grad:
                Trainable_params += mulValue  # 可训练参数量
            else:
                NonTrainable_params += mulValue  # 非可训练参数量

        patience = self.config.patience + self.config.evaluate_begin
        if self.config.log_step < 0:
            self.config.log_step = (
                len(self.train_dataloaders[0])
                if self.config.log_step < 0
                else self.config.log_step
            )

        self.logger.info(
            "***** Running training for {} *****".format(self.config.task_name)
        )
        self.logger.info("Training set examples = %d", len(self.train_set))
        if self.valid_set:
            self.logger.info("Valid set examples = %d", len(self.valid_set))
        if self.test_set:
            self.logger.info("Test set examples = %d", len(self.test_set))
        self.logger.info(
            "Total params = %d, Trainable params = %d, Non-trainable params = %d",
            Total_params,
            Trainable_params,
            NonTrainable_params,
        )
        self.logger.info("Batch size = %d", self.config.batch_size)
        self.logger.info(
            "Num steps = %d",
            len(self.train_dataloaders[0]) * self.config.num_epoch,
        )
        weight = (
            torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
            .float()
            .cuda()
        )

        for epoch in range(self.config.num_epoch):
            patience -= 1
            description = "Epoch:{} | Loss:{}".format(epoch, 0)
            iterator = tqdm(self.train_dataloaders[0], desc=description)
            for i_batch, sample_batched in enumerate(iterator):
                global_step += 1
                # switch model to trainer mode, clear gradient accumulators
                self.model.train()
                self.optimizer.zero_grad()

                if self.config.use_amp:
                    with torch.cuda.amp.autocast():
                        (
                            _,
                            sentences,
                            token_ids,
                            lengths,
                            masks,
                            _,
                            _,
                            aspect_tags,
                            tags,
                            word_pair_position,
                            word_pair_deprel,
                            word_pair_pos,
                            word_pair_synpost,
                            tags_symmetry,
                        ) = sample_batched

                else:
                    (
                        _,
                        sentences,
                        token_ids,
                        lengths,
                        masks,
                        _,
                        _,
                        aspect_tags,
                        tags,
                        word_pair_position,
                        word_pair_deprel,
                        word_pair_pos,
                        word_pair_synpost,
                        tags_symmetry,
                    ) = sample_batched

                inputs = {
                    "token_ids": token_ids,
                    "masks": masks,
                    "word_pair_position": word_pair_position,
                    "word_pair_deprel": word_pair_deprel,
                    "word_pair_pos": word_pair_pos,
                    "word_pair_synpost": word_pair_synpost,
                }

                tags_flatten = tags.reshape([-1])
                tags_symmetry_flatten = tags_symmetry.reshape([-1])
                if self.config.get("relation_constraint", True):
                    predictions = self.model(inputs)
                    (
                        biaffine_pred,
                        post_pred,
                        deprel_pred,
                        postag,
                        synpost,
                        final_pred,
                    ) = (
                        predictions[0],
                        predictions[1],
                        predictions[2],
                        predictions[3],
                        predictions[4],
                        predictions[5],
                    )
                    l_ba = 0.10 * criterion(
                        biaffine_pred.reshape([-1, biaffine_pred.shape[3]]),
                        tags_symmetry_flatten,
                    )
                    l_rpd = 0.01 * criterion(
                        post_pred.reshape([-1, post_pred.shape[3]]),
                        tags_symmetry_flatten,
                    )
                    l_dep = 0.01 * criterion(
                        deprel_pred.reshape([-1, deprel_pred.shape[3]]),
                        tags_symmetry_flatten,
                    )
                    l_psc = 0.01 * criterion(
                        postag.reshape([-1, postag.shape[3]]), tags_symmetry_flatten
                    )
                    l_tbd = 0.01 * criterion(
                        synpost.reshape([-1, synpost.shape[3]]), tags_symmetry_flatten
                    )

                    if self.config.get("symmetry_decoding", False):
                        l_p = torch.nn.functional.cross_entropy(
                            final_pred.reshape([-1, final_pred.shape[3]]),
                            tags_symmetry_flatten,
                            weight=weight,
                            ignore_index=-1,
                        )
                    else:
                        l_p = torch.nn.functional.cross_entropy(
                            final_pred.reshape([-1, final_pred.shape[3]]),
                            tags_flatten,
                            weight=weight,
                            ignore_index=-1,
                        )

                    loss = l_ba + l_rpd + l_dep + l_psc + l_tbd + l_p
                else:
                    preds = self.model(inputs)[-1]
                    preds_flatten = preds.reshape([-1, preds.shape[3]])
                    if self.config.symmetry_decoding:
                        loss = torch.nn.functional.cross_entropy(
                            preds_flatten,
                            tags_symmetry_flatten,
                            weight=weight,
                            ignore_index=-1,
                        )
                    else:
                        loss = torch.nn.functional.cross_entropy(
                            preds_flatten, tags_flatten, weight=weight, ignore_index=-1
                        )

                if self.config.auto_device == DeviceTypeOption.ALL_CUDA:
                    loss = loss.mean()

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
                        if len(self.valid_dataloaders) > 1:
                            joint_precision, joint_recall, joint_f1 = self._evaluate_f1(
                                self.valid_dataloaders[0]
                            )

                        else:
                            joint_precision, joint_recall, joint_f1 = self._evaluate_f1(
                                self.test_dataloader
                            )
                        self.config.metrics_of_this_checkpoint["f1"] = joint_f1

                        if joint_f1 > max_fold_f1:
                            if joint_f1 > max_fold_f1:
                                max_fold_f1 = joint_f1
                                patience = self.config.patience - 1

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
                                save_path = "{0}/{1}_{2}_f1_{3}/".format(
                                    self.config.model_path_to_save,
                                    self.config.model_name,
                                    self.config.dataset_name,
                                    round(joint_f1 * 100, 2),
                                )

                                if (
                                    joint_f1
                                    > self.config.max_test_metrics["max_apc_test_f1"]
                                ):
                                    self.config.max_test_metrics[
                                        "max_apc_test_f1"
                                    ] = joint_f1

                                save_model(
                                    self.config, self.model, self.tokenizer, save_path
                                )

                        postfix = "Dev F1:{:>.2f}(max:{:>.2f})".format(
                            joint_f1 * 100,
                            max_fold_f1 * 100,
                        )
                        iterator.set_postfix_str(postfix)
                    elif self.config.save_mode and epoch >= self.config.evaluate_begin:
                        save_model(
                            self.config,
                            self.model,
                            self.tokenizer,
                            save_path + "_{}/".format(loss.item()),
                        )
                else:
                    if self.config.get("loss_display", "smooth") == "smooth":
                        description = "Epoch:{:>3d} | Smooth Loss: {:>.4f}".format(
                            epoch, round(np.nanmean(losses), 4)
                        )
                    else:
                        description = "Epoch:{:>3d} | Batch Loss: {:>.4f}".format(
                            epoch, round(loss.item(), 4)
                        )
                iterator.set_description(description)
                iterator.refresh()
            if patience == 0:
                break

        if not self.valid_dataloaders:
            self.config.MV.log_metric(
                self.config.model_name
                + "-"
                + self.config.dataset_name
                + "-"
                + self.config.pretrained_bert,
                "Max-Test-F1 w/o Valid Set",
                max_fold_f1 * 100,
            )

        if self.test_dataloader:
            fprint(
                "Loading best model: {} and evaluating on test set ".format(save_path)
            )
            self._reload_model_state_dict(save_path)
            joint_precision, joint_recall, joint_f1 = self._evaluate_f1(
                self.test_dataloader
            )

            self.config.MV.log_metric(
                self.config.model_name
                + "-"
                + self.config.dataset_name
                + "-"
                + self.config.pretrained_bert,
                "Max-Test-F1",
                joint_f1 * 100,
            )
            # shutil.rmtree(save_path)

        self.logger.info(self.config.MV.summary(no_print=True))
        # self.logger.info(self.config.MV.short_summary(no_print=True))

        if self.valid_dataloader or self.config.save_mode:
            del self.train_dataloaders
            del self.test_dataloader
            del self.valid_dataloaders
            del self.model
            cuda.empty_cache()
            time.sleep(3)
            return save_path
        else:
            del self.train_dataloaders
            del self.test_dataloader
            del self.valid_dataloaders
            cuda.empty_cache()
            time.sleep(3)
            return self.model, self.config, self.tokenizer

    def _k_fold_train_and_evaluate(self, criterion):
        fold_test_acc = []
        fold_test_f1 = []

        save_path_k_fold = ""
        max_fold_acc_k_fold = 0

        losses = []

        self.config.metrics_of_this_checkpoint = {"acc": 0, "f1": 0}
        self.config.max_test_metrics = {"max_apc_test_acc": 0, "max_apc_test_f1": 0}

        for f, (train_dataloader, valid_dataloader) in enumerate(
            zip(self.train_dataloaders, self.valid_dataloaders)
        ):
            patience = self.config.patience + self.config.evaluate_begin
            if self.config.log_step < 0:
                self.config.log_step = (
                    len(self.train_dataloaders[0])
                    if self.config.log_step < 0
                    else self.config.log_step
                )

            self.logger.info(
                "***** Running training for {} *****".format(self.config.task_name)
            )
            self.logger.info("Training set examples = %d", len(self.train_set))
            if self.valid_set:
                self.logger.info("Valid set examples = %d", len(self.valid_set))
            if self.test_set:
                self.logger.info("Test set examples = %d", len(self.test_set))
            self.logger.info("Batch size = %d", self.config.batch_size)
            self.logger.info(
                "Num steps = %d",
                len(train_dataloader) // self.config.batch_size * self.config.num_epoch,
            )
            if len(self.train_dataloaders) > 1:
                self.logger.info(
                    "No. {} trainer in {} folds".format(
                        f + 1, self.config.cross_validate_fold
                    )
                )
            global_step = 0
            max_fold_acc = 0
            max_fold_f1 = 0
            save_path = "{0}/{1}_{2}".format(
                self.config.model_path_to_save,
                self.config.model_name,
                self.config.dataset_name,
            )
            for epoch in range(self.config.num_epoch):
                patience -= 1
                description = "Epoch:{} | Loss:{}".format(epoch, 0)
                iterator = tqdm(train_dataloader, desc=description)
                for i_batch, sample_batched in enumerate(iterator):
                    global_step += 1
                    # switch model to train mode, clear gradient accumulators
                    self.model.train()
                    self.optimizer.zero_grad()
                    inputs = {
                        col: sample_batched[col].to(self.config.device)
                        for col in self.config.inputs
                    }

                    if self.config.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(inputs)
                    else:
                        outputs = self.model(inputs)

                    targets = sample_batched["polarity"].to(self.config.device)

                    if (
                        isinstance(outputs, dict)
                        and "loss" in outputs
                        and outputs["loss"] != 0
                    ):
                        loss = outputs["loss"]
                    else:
                        loss = criterion(outputs["logits"], targets)

                    if self.config.auto_device == DeviceTypeOption.ALL_CUDA:
                        loss = loss.mean()

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
                            test_acc, f1 = self._evaluate_acc_f1(valid_dataloader)

                            self.config.metrics_of_this_checkpoint["acc"] = test_acc
                            self.config.metrics_of_this_checkpoint["f1"] = f1

                            if test_acc > max_fold_acc or f1 > max_fold_f1:
                                if test_acc > max_fold_acc:
                                    patience = self.config.patience - 1
                                    max_fold_acc = test_acc

                                if f1 > max_fold_f1:
                                    max_fold_f1 = f1
                                    patience = self.config.patience - 1

                                if self.config.model_path_to_save:
                                    if not os.path.exists(
                                        self.config.model_path_to_save
                                    ):
                                        os.makedirs(self.config.model_path_to_save)
                                    if save_path:
                                        try:
                                            shutil.rmtree(save_path)
                                            # logger.info('Remove sub-optimal trained model:', save_path)
                                        except:
                                            # logger.info('Can not remove sub-optimal trained model:', save_path)
                                            pass
                                    save_path = "{0}/{1}_{2}_acc_{3}_f1_{4}/".format(
                                        self.config.model_path_to_save,
                                        self.config.model_name,
                                        self.config.dataset_name,
                                        round(test_acc * 100, 2),
                                        round(f1 * 100, 2),
                                    )

                                    if (
                                        test_acc
                                        > self.config.max_test_metrics[
                                            "max_apc_test_acc"
                                        ]
                                    ):
                                        self.config.max_test_metrics[
                                            "max_apc_test_acc"
                                        ] = test_acc
                                    if (
                                        f1
                                        > self.config.max_test_metrics[
                                            "max_apc_test_f1"
                                        ]
                                    ):
                                        self.config.max_test_metrics[
                                            "max_apc_test_f1"
                                        ] = f1

                                    save_model(
                                        self.config,
                                        self.model,
                                        self.tokenizer,
                                        save_path,
                                    )

                            postfix = "Dev Acc:{:>.2f}(max:{:>.2f}) Dev F1:{:>.2f}(max:{:>.2f})".format(
                                test_acc * 100,
                                max_fold_acc * 100,
                                f1 * 100,
                                max_fold_f1 * 100,
                            )
                            iterator.set_postfix_str(postfix)
                        if (
                            self.config.save_mode
                            and epoch >= self.config.evaluate_begin
                        ):
                            save_model(
                                self.config,
                                self.model,
                                self.tokenizer,
                                save_path + "_{}/".format(loss.item()),
                            )
                    else:
                        if self.config.get("loss_display", "smooth") == "smooth":
                            description = "Epoch:{:>3d} | Smooth Loss: {:>.4f}".format(
                                epoch, round(np.nanmean(losses), 4)
                            )
                        else:
                            description = "Epoch:{:>3d} | Batch Loss: {:>.4f}".format(
                                epoch, round(loss.item(), 4)
                            )

                    iterator.set_description(description)
                    iterator.refresh()
                if patience == 0:
                    break
            max_fold_acc, max_fold_f1 = self._evaluate_acc_f1(self.test_dataloader)
            if max_fold_acc > max_fold_acc_k_fold:
                save_path_k_fold = save_path
            fold_test_acc.append(max_fold_acc)
            fold_test_f1.append(max_fold_f1)

            self.config.MV.log_metric(
                self.config.model_name,
                "Fold{}-Max-Test-Acc".format(f),
                max_fold_acc * 100,
            )
            self.config.MV.log_metric(
                self.config.model_name,
                "Fold{}-Max-Test-F1".format(f),
                max_fold_f1 * 100,
            )

            # self.logger.info(self.config.MV.summary(no_print=True))
            self.logger.info(self.config.MV.raw_summary(no_print=True))

            self._reload_model_state_dict(save_path_k_fold)

        max_test_acc = np.max(fold_test_acc)
        max_test_f1 = np.max(fold_test_f1)

        self.config.MV.log_metric(
            self.config.model_name
            + "-"
            + self.config.dataset_name
            + "-"
            + self.config.pretrained_bert,
            "Max-Test-Acc",
            max_test_acc * 100,
        )
        self.config.MV.log_metric(
            self.config.model_name
            + "-"
            + self.config.dataset_name
            + "-"
            + self.config.pretrained_bert,
            "Max-Test-F1",
            max_test_f1 * 100,
        )

        self.logger.info(self.config.MV.summary(no_print=True))
        # self.logger.info(self.config.MV.short_summary(no_print=True))
        self._reload_model_state_dict(save_path_k_fold)

        if os.path.exists("./init_state_dict.bin"):
            os.remove("./init_state_dict.bin")
        if self.valid_dataloaders or self.config.save_mode:
            del self.train_dataloaders
            del self.test_dataloader
            del self.valid_dataloaders
            del self.model
            cuda.empty_cache()
            time.sleep(3)
            return save_path
        else:
            del self.train_dataloaders
            del self.test_dataloader
            del self.valid_dataloaders
            cuda.empty_cache()
            time.sleep(3)
            return self.model, self.config, self.tokenizer

    def _evaluate_f1(self, data_loader, FLAG=False):
        self.model.eval()
        with torch.no_grad():
            all_ids = []
            all_sentences = []
            all_preds = []
            all_labels = []
            all_lengths = []
            all_sens_lengths = []
            all_token_ranges = []
            for eval_batch in data_loader:
                (
                    sentence_ids,
                    sentences,
                    token_ids,
                    lengths,
                    masks,
                    sens_lens,
                    token_ranges,
                    aspect_tags,
                    tags,
                    word_pair_position,
                    word_pair_deprel,
                    word_pair_pos,
                    word_pair_synpost,
                    tags_symmetry,
                ) = eval_batch

                inputs = {
                    "token_ids": token_ids,
                    "masks": masks,
                    "word_pair_position": word_pair_position,
                    "word_pair_deprel": word_pair_deprel,
                    "word_pair_pos": word_pair_pos,
                    "word_pair_synpost": word_pair_synpost,
                }

                preds = self.model(inputs)[-1]

                preds = nn.functional.softmax(preds, dim=-1)
                preds = torch.argmax(preds, dim=3)
                all_preds.append(preds)
                all_labels.append(tags)
                all_lengths.append(lengths)
                all_sens_lengths.extend(sens_lens)
                all_token_ranges.extend(token_ranges)
                all_ids.extend(sentence_ids)
                all_sentences.extend(sentences)

            all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
            all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
            all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()

            metric = Metric(
                self.config,
                all_preds,
                all_labels,
                all_lengths,
                all_sens_lengths,
                all_token_ranges,
            )
            precision, recall, f1 = metric.score_uniontags()
            aspect_results = metric.score_aspect()
            opinion_results = metric.score_opinion()
            # print('Aspect term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(aspect_results[0], aspect_results[1],
            #                                                           aspect_results[2]))
            # print('Opinion term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(opinion_results[0], opinion_results[1],
            #                                                            opinion_results[2]))
            # print(self.config.task + '\t\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

            if FLAG:
                metric.tagReport()

        return precision, recall, f1

    def _init_misc(self):
        # # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        diff_part = ["bert.embeddings", "bert.encoder"]

        if isinstance(self.model, EMCGCN):
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                        and any(nd in n for nd in diff_part)
                    ],
                    "weight_decay": self.config.l2reg,
                    "lr": self.config.learning_rate,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                        and any(nd in n for nd in diff_part)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.config.learning_rate,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                        and not any(nd in n for nd in diff_part)
                    ],
                    "weight_decay": self.config.l2reg,
                    "lr": 1e-3,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                        and not any(nd in n for nd in diff_part)
                    ],
                    "weight_decay": 0.0,
                    "lr": 1e-3,
                },
            ]
            self.optimizer = init_optimizer(self.config.optimizer)(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.l2reg,
                eps=self.config.get("adam_epsilon", 1e-8),
            )
        else:
            raise NotImplementedError("Please implement this method in your subclass!")

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        return self._train(criterion)

    def _train(self, criterion):
        self._prepare_dataloader()

        if self.config.warmup_step >= 0:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=len(self.train_dataloaders[0]) * self.config.num_epoch,
            )
            self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)

        if len(self.valid_dataloaders) > 1:
            return self._k_fold_train_and_evaluate(criterion)
        else:
            return self._train_and_evaluate(criterion)
