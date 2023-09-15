# -*- coding: utf-8 -*-
# file: classifier_trainer.py
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
from torch.utils.data import (
    DataLoader,
    random_split,
    ConcatDataset,
    RandomSampler,
    SequentialSampler,
)
from tqdm import tqdm
from transformers import AutoModel

from ..classic.__bert__.dataset_utils.data_utils_for_training import (
    Tokenizer4Pretraining,
    BERTTADDataset,
)
from ..classic.__glove__.dataset_utils.data_utils_for_training import (
    build_tokenizer,
    GloVeTADDataset,
)
from pyabsa.utils.file_utils import save_model
from pyabsa.utils.pyabsa_utils import (
    print_args,
    resume_from_checkpoint,
    retry,
    TransformerConnectionError,
    init_optimizer,
    build_embedding_matrix,
)

import pytorch_warmup as warmup

from ..models import BERTTADModelList, GloVeTADModelList


class Instructor:
    def __init__(self, opt, logger):
        if opt.use_amp:
            try:
                self.scaler = torch.cuda.amp.GradScaler()
                print("Use AMP for training!")
            except Exception:
                self.scaler = None
        else:
            self.scaler = None

        self.val_dataloader = None
        self.test_dataloader = None
        self.logger = logger
        self.opt = opt
        self.test_set = None
        self.train_set = None
        self.valid_set = None

        config_str = re.sub(
            r"<.*?>",
            "",
            str(sorted([str(self.opt.args[k]) for k in self.opt.args if k != "seed"])),
        )
        hash_tag = sha256(config_str.encode()).hexdigest()
        cache_path = "{}.{}.dataset.{}.cache".format(
            self.opt.model_name, self.opt.dataset_name, hash_tag
        )

        if os.path.exists(cache_path):
            print("Loading dataset cache:", cache_path)
            if self.opt.dataset_file["test"]:
                self.train_set, self.valid_set, self.test_set, opt = pickle.load(
                    open(cache_path, mode="rb")
                )
            else:
                self.train_set, opt = pickle.load(open(cache_path, mode="rb"))
            # reset output dim according to dataset labels
            self.opt.polarities_dim1 = opt.polarities_dim1
            self.opt.label_to_index = opt.label_to_index
            self.opt.index_to_label = opt.index_to_label

            self.opt.polarities_dim2 = opt.polarities_dim2
            self.opt.adv_label_to_index = opt.adv_label_to_index
            self.opt.index_to_adv_label = opt.index_to_adv_label

            self.opt.polarities_dim3 = opt.polarities_dim3
            self.opt.ood_label_to_index = opt.ood_label_to_index
            self.opt.index_to_ood_label = opt.index_to_ood_label

        # init BERT-based model and dataset
        if hasattr(BERTTADModelList, opt.model.__name__):
            self.tokenizer = Tokenizer4Pretraining(self.opt.max_seq_len, self.opt)
            if not os.path.exists(cache_path):
                self.train_set = BERTTADDataset(
                    self.opt.dataset_file["train"], self.tokenizer, self.opt
                )
                if self.opt.dataset_file["test"]:
                    self.test_set = BERTTADDataset(
                        self.opt.dataset_file["test"], self.tokenizer, self.opt
                    )
                else:
                    self.test_set = None
                if self.opt.dataset_file["valid"]:
                    self.valid_set = BERTTADDataset(
                        self.opt.dataset_file["valid"], self.tokenizer, self.opt
                    )
                else:
                    self.valid_set = None
            try:
                self.bert = AutoModel.from_pretrained(self.opt.pretrained_bert)
            except ValueError as e:
                print("Init pretrained model failed, exception: {}".format(e))
                raise TransformerConnectionError()

            # init the model behind the construction of datasets in case of updating polarities_dim
            self.model = self.opt.model(self.bert, self.opt).to(self.opt.device)

        elif hasattr(GloVeTADModelList, opt.model.__name__):
            # init GloVe-based model and dataset
            self.tokenizer = build_tokenizer(
                dataset_list=opt.dataset_file,
                max_seq_len=opt.max_seq_len,
                dat_fname="{0}_tokenizer.dat".format(
                    os.path.basename(opt.dataset_name)
                ),
                opt=self.opt,
            )
            self.embedding_matrix = build_embedding_matrix(
                word2idx=self.tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname="{0}_{1}_embedding_matrix.dat".format(
                    str(opt.embed_dim), os.path.basename(opt.dataset_name)
                ),
                opt=self.opt,
            )
            self.train_set = GloVeTADDataset(
                self.opt.dataset_file["train"], self.tokenizer, self.opt
            )

            if self.opt.dataset_file["test"]:
                self.test_set = GloVeTADDataset(
                    self.opt.dataset_file["test"], self.tokenizer, self.opt
                )
            else:
                self.test_set = None
            if self.opt.dataset_file["valid"]:
                self.valid_set = GloVeTADDataset(
                    self.opt.dataset_file["valid"], self.tokenizer, self.opt
                )
            else:
                self.valid_set = None
            self.model = opt.model(self.embedding_matrix, opt).to(opt.device)

        if self.opt.cache_dataset and not os.path.exists(cache_path):
            print(
                "Caching dataset... please remove cached dataset if change model or dataset"
            )
            if self.opt.dataset_file["test"]:
                pickle.dump(
                    (self.train_set, self.valid_set, self.test_set, self.opt),
                    open(cache_path, mode="wb"),
                )
            else:
                pickle.dump((self.train_set, self.opt), open(cache_path, mode="wb"))

        # use DataParallel for training if device count larger than 1
        if self.opt.auto_device == "allcuda":
            self.model.to(self.opt.device)
            self.model = torch.nn.parallel.DataParallel(self.model).module
        else:
            self.model.to(self.opt.device)

        self.optimizer = init_optimizer(self.opt.optimizer)(
            self.model.parameters(),
            lr=self.opt.learning_rate,
            weight_decay=self.opt.l2reg,
        )

        self.train_dataloaders = []
        self.val_dataloaders = []

        if os.path.exists("./init_state_dict.bin"):
            os.remove("./init_state_dict.bin")
        if self.opt.cross_validate_fold > 0:
            torch.save(self.model.state_dict(), "./init_state_dict.bin")

        self.opt.device = torch.device(self.opt.device)
        if self.opt.device.type == "cuda":
            self.logger.info(
                "cuda memory allocated:{}".format(
                    torch.cuda.memory_allocated(device=self.opt.device)
                )
            )

        print_args(self.opt, self.logger)

    def reload_model(self, ckpt="./init_state_dict.bin"):
        if os.path.exists(ckpt):
            self.model.load_state_dict(
                torch.load(find_file(ckpt, or_key=[".bin", "state_dict"]))
            )

    def prepare_dataloader(self, train_set):
        if self.opt.cross_validate_fold < 1:
            train_sampler = RandomSampler(
                self.train_set if not self.train_set else self.train_set
            )
            self.train_dataloaders.append(
                DataLoader(
                    dataset=train_set,
                    batch_size=self.opt.batch_size,
                    sampler=train_sampler,
                    pin_memory=True,
                )
            )
            if self.test_set:
                self.test_dataloader = DataLoader(
                    dataset=self.test_set, batch_size=self.opt.batch_size, shuffle=False
                )

            if self.valid_set:
                self.val_dataloader = DataLoader(
                    dataset=self.valid_set,
                    batch_size=self.opt.batch_size,
                    shuffle=False,
                )
        else:
            split_dataset = train_set
            len_per_fold = len(split_dataset) // self.opt.cross_validate_fold + 1
            folds = random_split(
                split_dataset,
                tuple(
                    [len_per_fold] * (self.opt.cross_validate_fold - 1)
                    + [
                        len(split_dataset)
                        - len_per_fold * (self.opt.cross_validate_fold - 1)
                    ]
                ),
            )

            for f_idx in range(self.opt.cross_validate_fold):
                train_set = ConcatDataset(
                    [x for i, x in enumerate(folds) if i != f_idx]
                )
                val_set = folds[f_idx]
                train_sampler = RandomSampler(train_set if not train_set else train_set)
                val_sampler = SequentialSampler(val_set if not val_set else val_set)
                self.train_dataloaders.append(
                    DataLoader(
                        dataset=train_set,
                        batch_size=self.opt.batch_size,
                        sampler=train_sampler,
                    )
                )
                self.val_dataloaders.append(
                    DataLoader(
                        dataset=val_set,
                        batch_size=self.opt.batch_size,
                        sampler=val_sampler,
                    )
                )
                if self.test_set:
                    self.test_dataloader = DataLoader(
                        dataset=self.test_set,
                        batch_size=self.opt.batch_size,
                        shuffle=False,
                    )

    def _train(self, criterion):
        self.prepare_dataloader(self.train_set)

        if self.opt.warmup_step >= 0:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=len(self.train_dataloaders[0]) * self.opt.num_epoch,
            )
            self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)

        if self.val_dataloaders:
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

        save_path = "{0}/{1}_{2}".format(
            self.opt.model_path_to_save, self.opt.model_name, self.opt.dataset_name
        )
        self.opt.metrics_of_this_checkpoint = {"acc": 0, "f1": 0}
        self.opt.max_test_metrics = {
            "max_cls_test_acc": 0,
            "max_cls_test_f1": 0,
            "max_adv_det_test_acc": 0,
            "max_adv_det_test_f1": 0,
            "max_adv_tr_test_acc": 0,
            "max_adv_tr_test_f1": 0,
        }

        self.logger.info(
            "***** Running training for Text Classification with Adversarial Attack Defense *****"
        )
        self.logger.info("Training set examples = %d", len(self.train_set))
        if self.test_set:
            self.logger.info("Test set examples = %d", len(self.test_set))
        self.logger.info("Batch size = %d", self.opt.batch_size)
        self.logger.info(
            "Num steps = %d",
            len(self.train_dataloaders[0]) // self.opt.batch_size * self.opt.num_epoch,
        )
        patience = self.opt.patience + self.opt.evaluate_begin
        if self.opt.log_step < 0:
            self.opt.log_step = (
                len(self.train_dataloaders[0])
                if self.opt.log_step < 0
                else self.opt.log_step
            )

        for epoch in range(self.opt.num_epoch):
            patience -= 1
            iterator = tqdm(self.train_dataloaders[0], postfix="Epoch:{}".format(epoch))
            for i_batch, sample_batched in enumerate(iterator):
                global_step += 1
                # switch model to train mode, clear gradient accumulators
                self.model.train()
                self.optimizer.zero_grad()
                inputs = [
                    sample_batched[col].to(self.opt.device)
                    for col in self.opt.inputs_cols
                ]
                if self.opt.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                label_targets = sample_batched["label"].to(self.opt.device)
                adv_tr_targets = sample_batched["adv_train_label"].to(self.opt.device)
                adv_det_targets = sample_batched["is_adv"].to(self.opt.device)

                sen_logits, advdet_logits, adv_tr_logits = (
                    outputs["sent_logits"],
                    outputs["advdet_logits"],
                    outputs["adv_tr_logits"],
                )
                sen_loss = criterion(sen_logits, label_targets)
                adv_det_loss = criterion(advdet_logits, adv_det_targets)
                adv_train_loss = criterion(adv_tr_logits, adv_tr_targets)
                loss = (
                    sen_loss
                    + self.opt.args.get("adv_det_weight", 5) * adv_det_loss
                    + self.opt.args.get("adv_train_weight", 5) * adv_train_loss
                )
                losses.append(loss.item())

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                if self.opt.warmup_step >= 0:
                    with self.warmup_scheduler.dampening():
                        self.lr_scheduler.step()

                # evaluate if test set is available
                if global_step % self.opt.log_step == 0:
                    if (
                        self.opt.dataset_file["test"]
                        and epoch >= self.opt.evaluate_begin
                    ):
                        if self.val_dataloader:
                            (
                                test_label_acc,
                                test_label_f1,
                                test_adv_det_acc,
                                test_adv_det_f1,
                                test_adv_tr_acc,
                                test_adv_tr_f1,
                            ) = self._evaluate_acc_f1(self.val_dataloader)
                        else:
                            (
                                test_label_acc,
                                test_label_f1,
                                test_adv_det_acc,
                                test_adv_det_f1,
                                test_adv_tr_acc,
                                test_adv_tr_f1,
                            ) = self._evaluate_acc_f1(self.test_dataloader)

                        self.opt.metrics_of_this_checkpoint[
                            "max_cls_test_acc"
                        ] = test_label_acc
                        self.opt.metrics_of_this_checkpoint[
                            "max_cls_test_f1"
                        ] = test_label_f1
                        self.opt.metrics_of_this_checkpoint[
                            "max_adv_det_test_acc"
                        ] = test_adv_det_acc
                        self.opt.metrics_of_this_checkpoint[
                            "max_adv_det_test_f1"
                        ] = test_adv_det_f1
                        self.opt.metrics_of_this_checkpoint[
                            "max_adv_tr_test_acc"
                        ] = test_adv_tr_acc
                        self.opt.metrics_of_this_checkpoint[
                            "max_adv_tr_test_f1"
                        ] = test_adv_tr_f1

                        if (
                            test_label_acc > max_label_fold_acc
                            or test_label_acc > max_label_fold_f1
                            or test_adv_det_acc > max_adv_det_fold_acc
                            or test_adv_det_f1 > max_adv_det_fold_f1
                            or test_adv_tr_acc > max_adv_tr_fold_acc
                            or test_adv_tr_f1 > max_adv_tr_fold_f1
                        ):
                            if test_label_acc > max_label_fold_acc:
                                patience = self.opt.patience
                                max_label_fold_acc = test_label_acc

                            if test_label_f1 > max_label_fold_f1:
                                patience = self.opt.patience
                                max_label_fold_f1 = test_label_f1

                            if test_adv_det_acc > max_adv_det_fold_acc:
                                patience = self.opt.patience
                                max_adv_det_fold_acc = test_adv_det_acc

                            if test_adv_det_f1 > max_adv_det_fold_f1:
                                patience = self.opt.patience
                                max_adv_det_fold_f1 = test_adv_det_f1

                            if test_adv_tr_acc > max_adv_tr_fold_acc:
                                patience = self.opt.patience
                                max_adv_tr_fold_acc = test_adv_tr_acc

                            if test_adv_tr_f1 > max_adv_tr_fold_f1:
                                patience = self.opt.patience
                                max_adv_tr_fold_f1 = test_adv_tr_f1

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
                                save_path = (
                                    "{0}/{1}_{2}_cls_acc_{3}_cls_f1_{4}_adv_det_acc_{5}_adv_det_f1_{6}"
                                    "_adv_training_acc_{7}_adv_training_f1_{8}/".format(
                                        self.opt.model_path_to_save,
                                        self.opt.model_name,
                                        self.opt.dataset_name,
                                        round(test_label_acc * 100, 2),
                                        round(test_label_f1 * 100, 2),
                                        round(test_adv_det_acc * 100, 2),
                                        round(test_adv_det_f1 * 100, 2),
                                        round(test_adv_tr_acc * 100, 2),
                                        round(test_adv_tr_f1 * 100, 2),
                                    )
                                )

                                if (
                                    test_label_acc
                                    > self.opt.max_test_metrics["max_cls_test_acc"]
                                ):
                                    self.opt.max_test_metrics[
                                        "max_cls_test_acc"
                                    ] = test_label_acc
                                if (
                                    test_label_f1
                                    > self.opt.max_test_metrics["max_cls_test_f1"]
                                ):
                                    self.opt.max_test_metrics[
                                        "max_cls_test_f1"
                                    ] = test_label_f1

                                if (
                                    test_adv_det_acc
                                    > self.opt.max_test_metrics["max_adv_det_test_acc"]
                                ):
                                    self.opt.max_test_metrics[
                                        "max_adv_det_test_acc"
                                    ] = test_adv_det_acc
                                if (
                                    test_adv_det_f1
                                    > self.opt.max_test_metrics["max_adv_det_test_f1"]
                                ):
                                    self.opt.max_test_metrics[
                                        "max_adv_det_test_f1"
                                    ] = test_adv_det_f1

                                if (
                                    test_adv_tr_acc
                                    > self.opt.max_test_metrics["max_adv_tr_test_acc"]
                                ):
                                    self.opt.max_test_metrics[
                                        "max_adv_tr_test_acc"
                                    ] = test_adv_tr_acc
                                if (
                                    test_adv_tr_f1
                                    > self.opt.max_test_metrics["max_adv_tr_test_f1"]
                                ):
                                    self.opt.max_test_metrics[
                                        "max_adv_tr_test_f1"
                                    ] = test_adv_tr_f1

                                save_model(
                                    self.opt, self.model, self.tokenizer, save_path
                                )

                        postfix = (
                            "Epoch:{} | Loss:{:.4f} | CLS ACC:{:.2f}(max:{:.2f}) | AdvDet ACC:{:.2f}(max:{:.2f})"
                            " | AdvCLS ACC:{:.2f}(max:{:.2f})".format(
                                epoch,
                                sen_loss.item()
                                + adv_det_loss.item()
                                + adv_train_loss.item(),
                                test_label_acc * 100,
                                max_label_fold_acc * 100,
                                test_adv_det_acc * 100,
                                max_adv_det_fold_acc * 100,
                                test_adv_tr_acc * 100,
                                max_adv_tr_fold_acc * 100,
                            )
                        )
                    else:
                        if self.opt.save_mode and epoch >= self.opt.evaluate_begin:
                            save_model(
                                self.opt,
                                self.model,
                                self.tokenizer,
                                save_path + "_{}/".format(loss.item()),
                            )
                        postfix = (
                            "Epoch:{} | Loss: {} |No evaluation until epoch:{}".format(
                                epoch, round(loss.item(), 8), self.opt.evaluate_begin
                            )
                        )

                    iterator.postfix = postfix
                    iterator.refresh()
            if patience < 0:
                break

        if not self.val_dataloader:
            self.opt.MV.add_metric(
                "Max-CLS-Acc w/o Valid Set", max_label_fold_acc * 100
            )
            self.opt.MV.add_metric("Max-CLS-F1 w/o Valid Set", max_label_fold_f1 * 100)
            self.opt.MV.add_metric(
                "Max-AdvDet-Acc w/o Valid Set", max_adv_det_fold_acc * 100
            )
            self.opt.MV.add_metric(
                "Max-AdvDet-F1 w/o Valid Set", max_adv_det_fold_f1 * 100
            )
        if self.val_dataloader:
            print(
                "Loading best model: {} and evaluating on test set ...".format(
                    save_path
                )
            )
            self.reload_model(find_file(save_path, ".state_dict"))
            (
                max_label_fold_acc,
                max_label_fold_f1,
                max_adv_det_fold_acc,
                max_adv_det_fold_f1,
                max_adv_tr_fold_acc,
                max_adv_tr_fold_f1,
            ) = self._evaluate_acc_f1(self.test_dataloader)

            self.opt.MV.add_metric("Max-CLS-Acc", max_label_fold_acc * 100)
            self.opt.MV.add_metric("Max-CLS-F1", max_label_fold_f1 * 100)
            self.opt.MV.add_metric("Max-AdvDet-Acc", max_adv_det_fold_acc * 100)
            self.opt.MV.add_metric("Max-AdvDet-F1", max_adv_det_fold_f1 * 100)
            self.opt.MV.add_metric("Max-AdvCLS-Acc", max_adv_tr_fold_acc * 100)
            self.opt.MV.add_metric("Max-AdvCLS-F1", max_adv_tr_fold_f1 * 100)

        self.logger.info(self.opt.MV.summary(no_print=True))

        print(
            "Training finished, we hope you can share your checkpoint with everybody, please see:",
            "https://github.com/yangheng95/PyABSA#how-to-share-checkpoints-eg-checkpoints-trained-on-your-custom-dataset-with-community",
        )

        rolling_intv = 5
        df = pandas.DataFrame(losses)
        losses = list(
            numpy.hstack(df.rolling(rolling_intv, min_periods=1).mean().values)
        )
        self.opt.loss = losses[-1]
        # self.opt.loss = np.average(losses)

        print_args(self.opt, self.logger)

        if self.val_dataloader or self.opt.save_mode:
            del self.train_dataloaders
            del self.test_dataloader
            del self.val_dataloader
            del self.model
            cuda.empty_cache()
            time.sleep(3)
            return save_path
        else:
            del self.train_dataloaders
            del self.test_dataloader
            del self.val_dataloader
            cuda.empty_cache()
            time.sleep(3)
            return self.model, self.opt, self.tokenizer

    def _k_fold_train_and_evaluate(self):
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
                t_inputs = [
                    t_sample_batched[col].to(self.opt.device)
                    for col in self.opt.inputs_cols
                ]
                t_label_targets = t_sample_batched["label"].to(self.opt.device)
                t_adv_tr_targets = t_sample_batched["adv_train_label"].to(
                    self.opt.device
                )
                t_adv_det_targets = t_sample_batched["is_adv"].to(self.opt.device)

                t_outputs = self.model(t_inputs)
                sent_logits, advdet_logits, adv_tr_logits = (
                    t_outputs["sent_logits"],
                    t_outputs["advdet_logits"],
                    t_outputs["adv_tr_logits"],
                )

                # --------------------------------------------------------------------------------------------#
                valid_label_targets = torch.tensor(
                    [x for x in t_label_targets.cpu() if x != -100]
                ).to(self.opt.device)
                if any(valid_label_targets):
                    valid_label_logit_ids = [
                        True if x != -100 else False for x in t_label_targets.cpu()
                    ]
                    valid_label_logits = sent_logits[valid_label_logit_ids]

                    n_label_test_correct += (
                        (torch.argmax(valid_label_logits, -1) == valid_label_targets)
                        .sum()
                        .item()
                    )
                    n_label_test_total += len(valid_label_logits)

                    if t_label_targets_all is None:
                        t_label_targets_all = valid_label_targets
                        t_label_outputs_all = valid_label_logits
                    else:
                        t_label_targets_all = torch.cat(
                            (t_label_targets_all, valid_label_targets), dim=0
                        )
                        t_label_outputs_all = torch.cat(
                            (t_label_outputs_all, valid_label_logits), dim=0
                        )

                # --------------------------------------------------------------------------------------------#
                n_adv_det_test_correct += (
                    (torch.argmax(advdet_logits, -1) == t_adv_det_targets).sum().item()
                )
                n_adv_det_test_total += len(advdet_logits)

                if t_adv_det_targets_all is None:
                    t_adv_det_targets_all = t_adv_det_targets
                    t_adv_det_outputs_all = advdet_logits
                else:
                    t_adv_det_targets_all = torch.cat(
                        (t_adv_det_targets_all, t_adv_det_targets), dim=0
                    )
                    t_adv_det_outputs_all = torch.cat(
                        (t_adv_det_outputs_all, advdet_logits), dim=0
                    )

                # --------------------------------------------------------------------------------------------#
                valid_adv_tr_targets = torch.tensor(
                    [x for x in t_adv_tr_targets.cpu() if x != -100]
                ).to(self.opt.device)
                if any(t_adv_tr_targets):
                    valid_adv_tr_logit_ids = [
                        True if x != -100 else False for x in t_adv_tr_targets.cpu()
                    ]
                    valid_adv_tr_logits = adv_tr_logits[valid_adv_tr_logit_ids]

                    n_adv_tr_test_correct += (
                        (torch.argmax(valid_adv_tr_logits, -1) == valid_adv_tr_targets)
                        .sum()
                        .item()
                    )
                    n_adv_tr_test_total += len(valid_adv_tr_logits)

                    if t_adv_tr_targets_all is None:
                        t_adv_tr_targets_all = valid_adv_tr_targets
                        t_adv_tr_outputs_all = valid_adv_tr_logits
                    else:
                        t_adv_tr_targets_all = torch.cat(
                            (t_adv_tr_targets_all, valid_adv_tr_targets), dim=0
                        )
                        t_adv_tr_outputs_all = torch.cat(
                            (t_adv_tr_outputs_all, valid_adv_tr_logits), dim=0
                        )

        label_test_acc = n_label_test_correct / n_label_test_total
        label_test_f1 = metrics.f1_score(
            t_label_targets_all.cpu(),
            torch.argmax(t_label_outputs_all, -1).cpu(),
            labels=list(range(self.opt.class_dim)),
            average="macro",
        )
        if self.opt.args.get("show_metric", False):
            print(
                "\n---------------------------- Standard Classification Report ----------------------------\n"
            )
            print(
                metrics.classification_report(
                    t_label_targets_all.cpu(),
                    torch.argmax(t_label_outputs_all, -1).cpu(),
                    target_names=[
                        self.opt.index_to_label[x] for x in self.opt.index_to_label
                    ],
                )
            )
            print(
                "\n---------------------------- Standard Classification Report ----------------------------\n"
            )

        adv_det_test_acc = n_adv_det_test_correct / n_adv_det_test_total
        adv_det_test_f1 = metrics.f1_score(
            t_adv_det_targets_all.cpu(),
            torch.argmax(t_adv_det_outputs_all, -1).cpu(),
            labels=list(range(self.opt.adv_det_dim)),
            average="macro",
        )

        adv_tr_test_acc = n_adv_tr_test_correct / n_adv_tr_test_total
        adv_tr_test_f1 = metrics.f1_score(
            t_adv_tr_targets_all.cpu(),
            torch.argmax(t_adv_tr_outputs_all, -1).cpu(),
            labels=list(range(self.opt.class_dim)),
            average="macro",
        )

        return (
            label_test_acc,
            label_test_f1,
            adv_det_test_acc,
            adv_det_test_f1,
            adv_tr_test_acc,
            adv_tr_test_f1,
        )

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()

        return self._train(criterion)


@retry
def train4tad(opt, from_checkpoint_path, logger):
    random.seed(opt.seed)
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    if hasattr(BERTTADModelList, opt.model.__name__):
        opt.inputs_cols = BERTTADDataset.bert_baseline_input_colses[
            opt.model.__name__.lower()
        ]

    elif hasattr(GloVeTADModelList, opt.model.__name__):
        opt.inputs_cols = GloVeTADDataset.glove_input_colses[opt.model.__name__.lower()]

    opt.device = torch.device(opt.device)

    trainer = Instructor(opt, logger)
    resume_from_checkpoint(trainer, from_checkpoint_path)

    return trainer.run()
