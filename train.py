# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

# modified: yangheng<yangheng@m.scnu.edu.cn>

from pytorch_pretrained_bert import BertModel
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse
import math
import os

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset

from models import LCF_BERT, LCF_GLOVE
from models.bert_spc import BERT_SPC


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        if 'bert' in opt.model_name:
            opt.learning_rate = 2e-5
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)

            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            opt.learning_rate = 0.001
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        # self.model = nn.DataParallel(self.model)
        trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        testset = ABSADataset(opt.dataset_file['test'], tokenizer)
        self.train_data_loader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(dataset=testset, batch_size=opt.batch_size, shuffle=False)

        if opt.device.type == 'cuda':
            print("cuda memory allocated:", torch.cuda.memory_allocated(device=opt.device.index))

        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

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

    def _train(self, criterion, optimizer, max_test_acc_overall=0):
        writer = SummaryWriter(log_dir=self.opt.logdir)
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    test_acc, f1 = self._evaluate_acc_f1()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('state_dict'):
                                os.mkdir('state_dict')
                            path = 'state_dict/{0}_{1}_acc{2}'.format(self.opt.model_name, self.opt.dataset,
                                                                      round(test_acc * 100, 2))
                            # torch.save(self.model.state_dict(), path)
                            # print('>> saved: ' + path)
                            # torch.save(self.model.state_dict(),path)
                        print('max_acc:', round(test_acc * 100, 2), 'f1:', round(f1 * 100, 2))
                    if f1 > max_f1:
                        max_f1 = f1

                    writer.add_scalar('loss', loss, global_step)
                    writer.add_scalar('acc', train_acc, global_step)
                    writer.add_scalar('test_acc', test_acc, global_step)
                    # print('loss: {:.4f}, acc: {:.2f}, test_acc: {:.2f}, f1: {:.2f}'.format(loss.item(),train_acc*100,test_acc*100, f1*100))
        writer.close()
        return max_test_acc, max_f1

    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)
                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return test_acc, f1

    def run(self, repeats=1):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        max_test_acc_overall = 0
        max_f1_overall = 0
        for i in range(repeats):
            print('repeat: ', i)
            self._reset_params()
            max_test_acc, max_f1 = self._train(criterion, optimizer, max_test_acc_overall=max_test_acc_overall)
            # print('max_test_acc: {0}     max_f1: {1}'.format(max_test_acc, max_f1))

            max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
            max_f1_overall = max(max_f1, max_f1_overall)
            print('#' * 100)

        print("max_test_acc_overall:", max_test_acc_overall * 100)
        print("max_f1_overall:", max_f1_overall * 100)
        return max_test_acc_overall * 100, max_f1_overall * 100


def single_train(model, dataset, local_context_focus, SRD, n):

    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=model, type=str)
    parser.add_argument('--dataset', default=dataset, type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=5, type=int)
    # parser.add_argument('--num_epoch', default=7, type=int)
    parser.add_argument('--batch_size', default=16, type=int)  # try 16, 32, 64 for BERT models
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--logdir', default='log', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--SRD', default=SRD, type=int)
    parser.add_argument('--local_context_focus', default=local_context_focus, type=str)

    opt = parser.parse_args()

    model_classes = {
        'bert_spc': BERT_SPC,
        'lcf_glove': LCF_GLOVE,
        'lcf_bert': LCF_BERT,
    }

    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
            },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        }
    }

    input_colses = {
        'bert_spc': ['text_bert_indices', 'bert_segments_ids', 'aspect_indices'],
        'lcf_glove': ['text_raw_indices', 'text_raw_indices', 'aspect_indices'],
        'lcf_bert': ['text_bert_indices', 'bert_segments_ids', 'text_raw_bert_indices', 'aspect_bert_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
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
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    ins = Instructor(opt)
    return ins.run(1)  # _reset_params in every repeat


def avg_train(model, dataset, SRD, local_context_focus, n):
    mean_test_acc_overall = 0
    mean_f1_overall = 0
    temp_test_acc_overall = 0
    temp_f1_overall = 0
    max_acc_overall = 0
    max_f1_overall = 0
    scores = []
    for t in range(n):
        print(model, '-', dataset, '-', local_context_focus, ' No.', t + 1, ' in ', n)
        test_acc_overall, f1_overall = single_train(
                model=model, dataset=dataset,
                local_context_focus=local_context_focus,
                SRD=SRD,
                n=n)
        scores.append([test_acc_overall, f1_overall])
        temp_test_acc_overall += test_acc_overall
        temp_f1_overall += f1_overall
        print("\n\nResults:")
        for i in range(len(scores)):
            if scores[i][0] > max_acc_overall:
                max_acc_overall = scores[i][0]
                max_f1_overall = scores[i][1]
            print(i + 1, " test_acc_overall: ", round(scores[i][0], 2), "    f1_overall: ", round(scores[i][1], 2))
        mean_test_acc_overall = temp_test_acc_overall / (t + 1)
        mean_f1_overall = temp_f1_overall / (t + 1)
        print('\n\n\nmax_acc_overall:', round(max_acc_overall, 2), 'f1_overall:', round(max_f1_overall, 2), '\n')
        print("mean_acc_overall:", round(mean_test_acc_overall, 2), "mean_f1_overall:", round(mean_f1_overall, 2), "\n\n\n")

    return mean_test_acc_overall, mean_f1_overall


if __name__ == '__main__':

    model = 'lcf_bert'
    # model = 'lcf_glove'
    # model = 'bert_spc'


    # local_context_focus='cdm'
    local_context_focus='cdw'

    # n = 5
    n = 10
    # n = 50

    SRD = 3
    mean_test_acc_overall, mean_f1_overall = avg_train(
        model=model, dataset='laptop',
        local_context_focus=local_context_focus,
        SRD=SRD,
        n=n
    )

    SRD = 3
    mean_test_acc_overall, mean_f1_overall = avg_train(
        model=model, dataset='restaurant',
        local_context_focus=local_context_focus,
        SRD=SRD,
        n=n
    )

    SRD = 3
    mean_test_acc_overall, mean_f1_overall = avg_train(
        model=model, dataset='twitter',
        local_context_focus=local_context_focus,
        SRD=SRD,
        n=n
    )

