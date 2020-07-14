# -*- coding: utf-8 -*-
# file: train_apc.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.

from pytorch_transformers import BertModel, BertTokenizer
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import math
import os
import numpy, random
from time import strftime, localtime

from utils.data_utils_apc import Tokenizer4Bert, ABSADataset, build_embedding_matrix, build_tokenizer, parse_experiments
from models.lc_apc import LCE_BERT, LCE_GLOVE, LCE_LSTM
from models.lc_apc import LCF_GLOVE, LCF_BERT
from models.lc_apc import HLCF_GLOVE, HLCF_BERT
from models.lc_apc import BERT_BASE, BERT_SPC
from models.apc import LSTM, IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, AEN_BERT
import logging, sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        if 'bert' in opt.model_name:
            # opt.learning_rate = 2e-5
            self.bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.bert_tokenizer = BertTokenizer.from_pretrained(opt.pretrained_bert_name, do_lower_case=True)
            tokenizer = Tokenizer4Bert(self.bert_tokenizer, opt.max_seq_len)
            self.model = opt.model_class(self.bert, opt).to(opt.device)
        else:
            # opt.learning_rate = 0.002
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        trainset = ABSADataset(opt.dataset_file['train'], tokenizer, opt)
        testset = ABSADataset(opt.dataset_file['test'], tokenizer, opt)
        self.train_data_loader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(dataset=testset, batch_size=opt.batch_size, shuffle=False)

        if opt.device.type == 'cuda':
            logging.info("cuda memory allocated:{}".format(torch.cuda.memory_allocated(device=opt.device.index)))

        self._log_write_args()

    def _log_write_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logging.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        for arg in vars(self.opt):
            logging.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params (with unfreezed bert)
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _save_model(self, model, save_path, mode=0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        model_output_dir = save_path+'_fine-tuned'
        if mode == 0 or 'bert' not in self.opt.model_name:
            torch.save(self.model.state_dict(), save_path+'.state_dict')  # save the state dict
        else:
            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)
            output_model_file = os.path.join(model_output_dir, 'pytorch_model.bin')
            output_config_file = os.path.join(model_output_dir, 'bert_config.json')

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            self.bert_tokenizer.save_vocabulary(model_output_dir)

    def _train(self, criterion, lce_criterion, optimizer, max_test_acc_overall=0):
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        loss, train_acc, test_acc, test_f1 = torch.tensor(0), 0, 0, 0
        for epoch in range(self.opt.num_epoch):
            logging.info('>' * 100)
            logging.info('epoch: {}'.format(epoch))
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_data_loader):

                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                if 'hlcf' not in self.opt.model_name:
                    inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                else:
                    inputs = [sample_batched[col] for col in self.opt.inputs_cols]

                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)
                if self.opt.lcp and 'lce' in self.opt.model_name:
                    sen_logits, lce_logits, lce_ids = outputs
                    sen_loss = criterion(sen_logits, targets)
                    lcp_loss = lce_criterion(lce_logits, lce_ids)
                    loss = 2 * (1 - self.opt.sigma) * sen_loss + self.opt.sigma * lcp_loss
                else:
                    sen_logits = outputs
                    loss = criterion(sen_logits, targets)
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(sen_logits, -1) == targets).sum().item()
                    n_total += len(sen_logits)
                    train_acc = n_correct / n_total
                    test_acc, f1 = self._evaluate_acc_f1()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('saved_models'):
                                os.mkdir('saved_models')
                            save_path = 'saved_models/{0}_{1}_acc{2}_seed{3}seed'.format(self.opt.model_name,
                                                        self.opt.dataset, round(test_acc * 100, 2), self.opt.seed)
                            # uncomment follow lines to save model during training
                            self._save_model(self.model, save_path, mode=0)
                            logging.info('saved: {}'.format(save_path))
                            logging.info('max_acc:{}, f1:{}'.format(round(test_acc * 100, 2), round(f1 * 100, 2)))
                    if f1 > max_f1:
                        max_f1 = f1
                    # # uncomment next line to monitor the training process
                    # logging.info('loss: {:.4f}, acc: {:.2f}, test_acc: {:.2f}, f1: {:.2f}'.
                    #              format(loss.item(), train_acc*100, test_acc*100, f1*100))

        return max_test_acc, max_f1

    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                if 'hlcf' not in self.opt.model_name:
                    t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                else:
                    t_inputs = [t_sample_batched[col] for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)

                if self.opt.lcp and 'lce' in self.opt.model_name:
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
        if self.opt.dataset in {'camera', 'notebook', 'car', 'phone'}:
            f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[1, 2],
                                  average='macro')
        else:
            f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                                  average='macro')
        return test_acc, f1

    def run(self, repeats=1):

        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        lce_criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        max_test_acc_overall = 0
        max_f1_overall = 0
        for i in range(repeats):
            logging.info('repeat: {}'.format(i))
            self._reset_params()
            max_test_acc, max_f1 = self._train(criterion, lce_criterion, optimizer,
                                               max_test_acc_overall=max_test_acc_overall)
            max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
            max_f1_overall = max(max_f1, max_f1_overall)
            logging.info('#' * 100)

        logging.info("max_test_acc_overall:{}".format(max_test_acc_overall * 100))
        logging.info("max_f1_overall:{}".format(max_f1_overall * 100))
        return max_test_acc_overall * 100, max_f1_overall * 100


def single_train(opt):
    if 'glove' in opt.model_name:
        logger.warning('Caution: The Chinese datasets are not available for GLoVe-based models.')

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'bert_base': BERT_BASE,
        'bert_spc': BERT_SPC,
        'lce_glove': LCE_GLOVE,
        'lce_bert': LCE_BERT,
        'lce_lstm': LCE_LSTM,
        'lcf_glove': LCF_GLOVE,
        'lcf_bert': LCF_BERT,
        'hlcf_glove': HLCF_GLOVE,
        'hlcf_bert': HLCF_BERT,
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'tc_lstm': TC_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'aen_bert': AEN_BERT,
    }

    dataset_files = {
        'twitter': {
            'train': './datasets/apc_datasets/acl-14-short-data/train.raw',
            'test': './datasets/apc_datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/apc_datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/apc_datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/apc_datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/apc_datasets/semeval14/Laptops_Test_Gold.xml.seg'
        },
        'car': {
            'train': './datasets/apc_datasets/Chinese/car/car.train.txt',
            'test': './datasets/apc_datasets/Chinese/car/car.test.txt'
        },
        'phone': {
            'train': './datasets/apc_datasets/Chinese/camera/camera.train.txt',
            'test': './datasets/apc_datasets/Chinese/camera/camera.test.txt'
        },
        'notebook': {
            'train': './datasets/apc_datasets/Chinese/notebook/notebook.train.txt',
            'test': './datasets/apc_datasets/Chinese/notebook/notebook.test.txt'
        },
        'camera': {
            'train': './datasets/apc_datasets/Chinese/phone/phone.train.txt',
            'test': './datasets/apc_datasets/Chinese/phone/phone.test.txt'
        },
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_
    }

    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = ABSADataset.input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]

    opt.device = torch.device(opt.device if 'cuda' in opt.device else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    ins = Instructor(opt)
    return ins.run()  # _reset_params in every repeat


def multi_train(config, n):
    import copy
    mean_test_acc_overall = 0
    mean_f1_overall = 0
    temp_test_acc_overall = 0
    temp_f1_overall = 0
    max_acc_overall = 0
    max_f1_overall = 0
    scores = []
    for t in range(n):
        logging.info('{} - {} - {} - No.{} in {}'.format(config.model_name, config.dataset, config.lcf, t + 1, n))
        config.seed = t
        test_acc_overall, f1_overall = single_train(copy.deepcopy(config))
        scores.append([test_acc_overall, f1_overall])
        temp_test_acc_overall += test_acc_overall
        temp_f1_overall += f1_overall
        logging.info("#" * 100)
        for i in range(len(scores)):
            if scores[i][0] > max_acc_overall:
                max_acc_overall = scores[i][0]
                max_f1_overall = scores[i][1]
            logging.info(
                "{} test_acc_overall: {}  f1_overall:{}".format(i + 1, round(scores[i][0], 2), round(scores[i][1], 2)))
        mean_test_acc_overall = temp_test_acc_overall / (t + 1)
        mean_f1_overall = temp_f1_overall / (t + 1)
        logging.info('max_acc_overall:{}  f1_overall:{}'.format(round(max_acc_overall, 2), round(max_f1_overall, 2)))
        logging.info("mean_acc_overall:{}  mean_f1_overall:{}".format(round(mean_test_acc_overall, 2),
                                                                      round(mean_f1_overall, 2)))
        logging.info("#" * 100)
    return mean_test_acc_overall, mean_f1_overall


if __name__ == '__main__':

    config_parser = argparse.ArgumentParser()
    config_parser.add_argument('--config', default='experiments_apc.json',
                        help='path of the experiments configuration', type=str)

    args = config_parser.parse_args()

    configs = parse_experiments(args.config)
    log_file = 'logs/{}.{}.log'.format(args.config, strftime("%y%m%d.%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    from utils.Pytorch_GPUManager import GPUManager

    GM = GPUManager()
    gpu = GM.auto_choice()

    for config in configs:
        config.device = 'cuda:' + str(gpu)
        # config.device = 'cpu'  # Uncomment this line to use CPU
        multi_train(config=config, n=config.repeat)
