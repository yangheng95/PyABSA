# -*- coding: utf-8 -*-
# file: apc_trainer.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


import os
import random
import pickle
import numpy
import torch
import torch.nn as nn
import shutil
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
from sklearn import metrics

from pyabsa.apc.models.bert_base import BERT_BASE
from pyabsa.apc.models.bert_spc import BERT_SPC
from pyabsa.apc.models.lcf_bert import LCF_BERT
from pyabsa.apc.models.slide_lcf_bert import SLIDE_LCF_BERT
from pyabsa.apc.dataset_utils.data_utils_for_training import ABSADataset
from pyabsa.apc.dataset_utils.apc_utils import Tokenizer4Bert
from pyabsa.apc.dataset_utils.apc_utils import get_polarities_dim

from pyabsa.logger import get_logger

logger = get_logger(os.getcwd())

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        self.bert = BertModel.from_pretrained(self.opt.pretrained_bert_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.opt.pretrained_bert_name, do_lower_case=True)
        self.tokenizer = Tokenizer4Bert(self.bert_tokenizer, self.opt.max_seq_len)
        self.model = self.opt.model_class(self.bert, self.opt).to(self.opt.device)

        trainset = ABSADataset(self.opt.dataset_file['train'], self.tokenizer, self.opt)
        self.train_data_loader = DataLoader(dataset=trainset,
                                            batch_size=self.opt.batch_size,
                                            shuffle=True,
                                            pin_memory=True)
        if 'test' in self.opt.dataset_file:
            testset = ABSADataset(self.opt.dataset_file['test'], self.tokenizer, self.opt)
            self.test_data_loader = DataLoader(dataset=testset,
                                               batch_size=self.opt.batch_size,
                                               shuffle=False,
                                               pin_memory=True)

        if self.opt.device.type == 'cuda':
            logger.info("cuda memory allocated:{}".format(torch.cuda.memory_allocated(device=self.opt.device.index)))

        self._log_write_args()

    def _log_write_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _save_model(self, model, save_path, mode=0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        if mode == 0 or 'bert' not in self.opt.model_name:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.model.cpu().state_dict(),
                       save_path + self.opt.model_name + '.state_dict')  # save the state dict
            pickle.dump(self.opt, open(save_path + 'model.config', 'wb'))
        else:
            # save the fine-tuned bert model
            model_output_dir = save_path + '_fine-tuned'
            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)
            output_model_file = os.path.join(model_output_dir, 'pytorch_model.bin')
            output_config_file = os.path.join(model_output_dir, 'bert_config.json')

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            self.bert_tokenizer.save_vocabulary(model_output_dir)

        # logger.info('trained model saved in: {}'.format(save_path))
        self.model.to(self.opt.device)

    def _train_and_evaluate(self, criterion, lca_criterion, optimizer):
        max_test_acc = 0
        max_f1 = 0
        test_acc = 0
        f1 = 0
        global_step = 0
        save_path = ''
        for epoch in range(self.opt.num_epoch):
            n_correct, n_total = 0, 0
            iterator = tqdm(self.train_data_loader)
            for i_batch, sample_batched in enumerate(iterator):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)
                if 'lca' in self.opt.model_name:
                    sen_logits, lca_logits, lca_ids = outputs
                    sen_loss = criterion(sen_logits, targets)
                    lcp_loss = lca_criterion(lca_logits, lca_ids)
                    loss = (1 - self.opt.sigma) * sen_loss + self.opt.sigma * lcp_loss
                else:
                    sen_logits = outputs
                    loss = criterion(sen_logits, targets)

                loss.backward()
                optimizer.step()

                # evaluate if test set is available
                if 'test' in self.opt.dataset_file and global_step % self.opt.log_step == 0:
                    # n_correct += (torch.argmax(sen_logits, -1) == targets).sum().item()
                    # n_total += len(sen_logits)
                    # train_acc = n_correct / n_total

                    test_acc, f1 = self._evaluate_acc_f1()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if self.opt.model_path_to_save:
                            if not os.path.exists(self.opt.model_path_to_save):
                                os.mkdir(self.opt.model_path_to_save)
                            if save_path:
                                try:
                                    shutil.rmtree(save_path)
                                    # logger.info('Remove sub-optimal trained model:', save_path)
                                except:
                                    # logger.info('Can not remove sub-optimal trained model:', save_path)
                                    pass
                            save_path = '{0}/{1}_{2}_acc{3}/'.format(self.opt.model_path_to_save,
                                                                     self.opt.model_name,
                                                                     self.opt.lcf,
                                                                     round(test_acc * 100, 2),
                                                                     )
                            self._save_model(self.model, save_path, mode=0)
                    if f1 > max_f1:
                        max_f1 = f1
                iterator.postfix = (
                    'Loss:{:.4f} | Test Acc:{:.2f}(max:{:.2f}) | Test F1:{:.2f}(max:{:.2f})'.format(loss.item(),
                                                                                                    test_acc * 100,
                                                                                                    max_test_acc * 100,
                                                                                                    f1 * 100,
                                                                                                    max_f1 * 100))
                iterator.refresh()
        # return the model paths of multiple training in case of loading the best model after training
        if save_path:
            logger.info('----------------------Training Summary----------------------')
            logger.info('Max Accuracy: {} Max F1: {}'.format(max_test_acc, max_f1))
            return save_path
        else:
            # direct return model if do not evaluate
            if self.opt.model_path_to_save:
                save_path = '{0}/{1}_{2}/'.format(self.opt.model_path_to_save,
                                                  self.opt.model_name,
                                                  self.opt.lcf,
                                                  )
                self._save_model(self.model, save_path, mode=0)
            return self.model, self.opt

    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):

                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)

                if 'lca' in self.opt.model_name:
                    sen_outputs, _, _ = self.model(t_inputs)
                else:
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
                              labels=list(range(get_polarities_dim(self.opt))), average='macro')
        return test_acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        lca_criterion = nn.CrossEntropyLoss()
        # # ---------------------------------------------------------------- #
        # bert_paprams = list(map(id, self.model.bert4global.parameters()))
        # _params = filter(lambda p: id(p) not in bert_paprams, self.model.parameters())
        # bert_params = filter(lambda p: p.requires_grad, self.model.bert4global.parameters())
        # params = [{'params': _params, 'lr': 0.0001},
        #           {'params': bert_params, 'lr': self.opt.learning_rate}
        #           ]
        # optimizer = self.opt.optimizer(params, weight_decay=self.opt.l2reg)
        # ---------------------------------------------------------------- #
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        return self._train_and_evaluate(criterion, lca_criterion, optimizer)


def train4apc(opt):

    if not isinstance(opt.seed, int):
        logger.info('Please do not use multiple random seeds without evaluating.')
        opt.seed = list(opt.seed)[0]
    random.seed(opt.seed)
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_classes = {
        'bert_base': BERT_BASE,
        'bert_spc': BERT_SPC,
        'lcf_bert': LCF_BERT,
        'lcfs_bert': LCF_BERT,
        'slide_lcf_bert': SLIDE_LCF_BERT,
        'slide_lcfs_bert': SLIDE_LCF_BERT
    }

    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
        'adamw': torch.optim.AdamW
    }

    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = ABSADataset.input_colses[opt.model_name]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device(opt.device)
    ins = Instructor(opt)
    return ins.run()
