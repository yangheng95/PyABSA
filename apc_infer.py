# -*- coding: utf-8 -*-
# file: train_apc.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.

from pytorch_transformers import BertModel, BertTokenizer
import torch
from torch.utils.data import DataLoader
import argparse
import numpy, random

from utils.data_utils_apc import Tokenizer4Bert, ABSADataset, ABSAInferDataset, build_embedding_matrix,\
    build_tokenizer_for_inferring, parse_experiments
from models.lc_apc import LCE_BERT, LCE_GLOVE, LCE_LSTM
from models.lc_apc import LCF_GLOVE, LCF_BERT
from models.lc_apc import HLCF_GLOVE, HLCF_BERT
from models.lc_apc import BERT_BASE, BERT_SPC
from models.apc import LSTM, IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, AEN_BERT
import sys

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
            tokenizer = build_tokenizer_for_inferring(
                fnames=[opt.infer_data],
                max_seq_len=opt.max_seq_len,
                dat_fname=args.inferring_dataset + "/" + opt.tokenizer
            )
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname=args.inferring_dataset + "/" + opt.embedding
            )
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        infer_set = ABSAInferDataset(args.inferring_dataset+"/"+opt.infer_data, tokenizer, opt)
        self.train_data_loader = DataLoader(dataset=infer_set, batch_size=1, shuffle=False)

    def _infer(self):
        with torch.no_grad():
            self.model.eval()
            for _, sample in enumerate(self.train_data_loader):
                print(sample['sentence'][0])
                if 'hlcf' not in self.opt.model_name:
                    inputs = [sample[col].to(self.opt.device) for col in self.opt.inputs_cols]
                else:
                    inputs = [sample[col] for col in self.opt.inputs_cols]
                self.model.eval()
                outputs = self.model(inputs)
                if self.opt.lcp and 'lce' in self.opt.model_name:
                    sen_logits, _, _ = outputs

                else:
                    sen_logits = outputs
                t_probs = torch.softmax(sen_logits, dim=-1).cpu().numpy()
                sentiment = t_probs.argmax(axis=-1) - 1
                if self.opt.dataset in {'camera', 'notebook', 'car', 'phone'}:
                    print('polarity of {} = {}'.format(sample['aspect'], 'Negative' if sentiment == 1 else 'Positive'))
                # for English datasets
                elif sentiment == 0:
                    print('polarity of {} = Negative'.format(sample['aspect']))
                elif sentiment == 1:
                    print('polarity of {} = Neutral'.format(sample['aspect']))
                elif sentiment == 2:
                    print('polarity of {} = Positive'.format(sample['aspect']))

    def run(self):

        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        return self._infer()

def init_and_infer(opt):

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

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_
    }

    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = ABSADataset.input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]

    ins = Instructor(opt)
    return ins.run()  # _reset_params in every repeat

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--inferring_dataset', default='inferring_dataset', type=str)

    args = parser.parse_args()
    configs = parse_experiments(args.inferring_dataset+'/eval_config.json')

    from utils.Pytorch_GPUManager import GPUManager

    GM = GPUManager()
    gpu = GM.auto_choice()

    opt = configs[0]
    opt.device = 'cuda:' + str(gpu)
    # config.device = 'cpu'  # Uncomment this line to use CPU

    import os

    for file in os.listdir('inferring_dataset'):
        if 'acc' in file:
            opt.state_dict_path = file
        if 'infer' in file:
            opt.infer_data = file
        if 'config' in file:
            opt.config = file
        if 'embedding' in file:
            opt.embedding = file.split('/')[-1]
        if 'tokenizer' in file:
            opt.tokenizer = file.split('/')[-1]

    print('*'*80)
    print('Warning: Be sure the eval-config, eval-dataset, saved_models, seed are compatible! ')
    print('*' * 80)
    opt.seed = int(opt.state_dict_path.split('seed')[1])
    init_and_infer(opt)


