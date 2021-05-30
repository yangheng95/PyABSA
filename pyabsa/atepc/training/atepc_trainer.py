# -*- coding: utf-8 -*-
# file: train_atepc.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


import os
import shutil
import random
import pickle
import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from seqeval.metrics import classification_report
from sklearn.metrics import f1_score
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel
from transformers.optimization import AdamW

from ..dataset_utils.data_utils_for_training import ATEPCProcessor, convert_examples_to_features
from ..models.lcf_atepc import LCF_ATEPC

import warnings

warnings.filterwarnings('ignore')


def train4atepc(config):
    args = config

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.batch_size = args.batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model_path_to_save and not os.path.exists(args.model_path_to_save):
        os.makedirs(args.model_path_to_save)

    processor = ATEPCProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1
    model_classes = {
        'lcf_atepc': LCF_ATEPC,
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

    args.bert_model = args.pretrained_bert_name

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    train_examples = processor.get_train_examples(args.dataset_file['train'], 'train')
    eval_examples = processor.get_test_examples(args.dataset_file['test'], 'test')
    num_train_optimization_steps = int(
        len(train_examples) / args.batch_size / args.gradient_accumulation_steps) * args.num_epoch
    bert_base_model = BertModel.from_pretrained(args.bert_model)
    bert_base_model.config.num_labels = num_labels

    if args.polarities_dim == 2:
        convert_polarity(train_examples)
        convert_polarity(eval_examples)

    model = model_classes[args.model_name](bert_base_model, args=args)

    for arg in vars(args):
        print('>>> {0}: {1}'.format(arg, getattr(args, arg)))

    model.to(args.device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.l2reg},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': args.l2reg}
    ]

    optimizer = optimizers[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.l2reg)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_len, tokenizer)
    all_spc_input_ids = torch.tensor([f.input_ids_spc for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_polarities = torch.tensor([f.polarities for f in eval_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
    all_tokens = [f.tokens for f in eval_features]
    eval_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                              all_polarities, all_valid_ids, all_lmask_ids)
    # Run prediction for full data
    eval_sampler = RandomSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    def evaluate(eval_ATE=True, eval_APC=True):
        # evaluate
        apc_result = {'max_apc_test_acc': 0, 'max_apc_test_f1': 0}
        ate_result = 0
        y_true = []
        y_pred = []
        n_test_correct, n_test_total = 0, 0
        test_apc_logits_all, test_polarities_all = None, None
        model.eval()
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        for input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask in eval_dataloader:
            input_ids_spc = input_ids_spc.to(args.device)
            input_mask = input_mask.to(args.device)
            segment_ids = segment_ids.to(args.device)
            valid_ids = valid_ids.to(args.device)
            label_ids = label_ids.to(args.device)
            polarities = polarities.to(args.device)
            l_mask = l_mask.to(args.device)

            with torch.no_grad():
                ate_logits, apc_logits = model(input_ids_spc, segment_ids, input_mask,
                                               valid_ids=valid_ids, polarities=polarities, attention_mask_label=l_mask)
            if eval_APC:
                polarities = model.get_batch_polarities(polarities)
                n_test_correct += (torch.argmax(apc_logits, -1) == polarities).sum().item()
                n_test_total += len(polarities)

                if test_polarities_all is None:
                    test_polarities_all = polarities
                    test_apc_logits_all = apc_logits
                else:
                    test_polarities_all = torch.cat((test_polarities_all, polarities), dim=0)
                    test_apc_logits_all = torch.cat((test_apc_logits_all, apc_logits), dim=0)

            if eval_ATE:
                if not args.use_bert_spc:
                    label_ids = model.get_batch_token_labels_bert_base_indices(label_ids)
                ate_logits = torch.argmax(F.log_softmax(ate_logits, dim=2), dim=2)
                ate_logits = ate_logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()
                for i, label in enumerate(label_ids):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(label):
                        if j == 0:
                            continue
                        elif label_ids[i][j] == len(label_list):
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                            break
                        else:
                            temp_1.append(label_map.get(label_ids[i][j], 'O'))
                            temp_2.append(label_map.get(ate_logits[i][j], 'O'))
        if eval_APC:
            test_acc = n_test_correct / n_test_total

            test_f1 = f1_score(torch.argmax(test_apc_logits_all, -1).cpu(), test_polarities_all.cpu(),
                               labels=list(range(args.polarities_dim)), average='macro')

            test_acc = round(test_acc * 100, 2)
            test_f1 = round(test_f1 * 100, 2)
            apc_result = {'max_apc_test_acc': test_acc, 'max_apc_test_f1': test_f1}

        if eval_ATE:
            report = classification_report(y_true, y_pred, digits=4)
            tmps = report.split()
            ate_result = round(float(tmps[7]) * 100, 2)
        return apc_result, ate_result

    def _save_model(args_to_save, model_to_save, save_path, mode=0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model_to_save.module if hasattr(model_to_save,
                                                        'module') else model_to_save  # Only save the model it-self

        if mode == 0:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model_to_save.cpu().state_dict(),
                       save_path + args_to_save.model_name + '.state_dict')  # save the state dict
            pickle.dump(args_to_save, open(save_path + 'model.config', 'wb'))
        else:
            # save the fine-tuned bert model
            model_output_dir = save_path
            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)
            output_model_file = os.path.join(model_output_dir, 'pytorch_model.bin')
            output_config_file = os.path.join(model_output_dir, 'bert_config.json')

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(model_output_dir)

        # print('trained model saved in: {}'.format(save_path))
        model_to_save.to(args_to_save.device)

    def train():
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_len, tokenizer)
        print("***** Running training *****")
        print("  Num examples = %d", len(train_examples))
        print("  Batch size = %d", args.batch_size)
        print("  Num steps = %d", num_train_optimization_steps)
        all_spc_input_ids = torch.tensor([f.input_ids_spc for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        all_polarities = torch.tensor([f.polarities for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids,
                                   all_label_ids, all_polarities, all_valid_ids, all_lmask_ids)

        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        max_apc_test_acc = 0
        max_apc_test_f1 = 0
        max_ate_test_f1 = 0
        global_step = 0
        save_path = ''
        for epoch in range(int(args.num_epoch)):
            nb_tr_examples, nb_tr_steps = 0, 0
            iterator = tqdm.tqdm(train_dataloader)
            for step, batch in enumerate(iterator):
                postfix = ''
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask = batch
                loss_ate, loss_apc = model(input_ids_spc, segment_ids, input_mask, label_ids, polarities, valid_ids,
                                           l_mask)
                loss = loss_ate + loss_apc
                loss.backward()
                nb_tr_examples += input_ids_spc.size(0)
                nb_tr_steps += 1
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if epoch >= 2 or args.num_epoch <= 2:
                    if global_step % args.log_step == 0:
                        apc_result, ate_result = evaluate(eval_ATE=not args.use_bert_spc)
                        # if save_path:
                        #     try:
                        #         shutil.rmtree(save_path)
                        #         # print('Remove sub-optimal trained model:', save_path)
                        #     except:
                        #         print('Can not remove sub-optimal trained model:', save_path)
                        if args.model_path_to_save:
                            save_path = '{0}/{1}_{2}_apcacc_{3}_apcf1_{4}_atef1_{5}/'.format(
                                args.model_path_to_save,
                                args.model_name,
                                args.lcf,
                                round(apc_result['max_apc_test_acc'], 2),
                                round(apc_result['max_apc_test_f1'], 2),
                                round(ate_result, 2)
                            )
                            if apc_result['max_apc_test_acc'] > max_apc_test_acc or \
                                    apc_result['max_apc_test_f1'] > max_apc_test_f1 or \
                                    ate_result > max_ate_test_f1:
                                _save_model(args, model, save_path, mode=0)

                        if apc_result['max_apc_test_acc'] > max_apc_test_acc:
                            max_apc_test_acc = apc_result['max_apc_test_acc']
                        if apc_result['max_apc_test_f1'] > max_apc_test_f1:
                            max_apc_test_f1 = apc_result['max_apc_test_f1']
                        if ate_result > max_ate_test_f1:
                            max_ate_test_f1 = ate_result

                        current_apc_test_acc = apc_result['max_apc_test_acc']
                        current_apc_test_f1 = apc_result['max_apc_test_f1']
                        current_ate_test_f1 = round(ate_result, 2)

                        postfix += f'APC_ACC: {current_apc_test_acc}(max:{max_apc_test_acc}) ' \
                                   f'APC_f1: {current_apc_test_f1}(max:{max_apc_test_f1}) '

                        if args.use_bert_spc:
                            postfix = f'ATE_F1: {current_apc_test_f1}(max:{max_apc_test_f1})' \
                                      f' (Unreliable since `use_bert_spc` is "True".)'
                        else:
                            postfix += f'ATE_f1: {current_ate_test_f1}(max:{max_ate_test_f1})'
                        iterator.postfix = postfix
                        iterator.refresh()

        return save_path

    return train()


def convert_polarity(examples):
    for i in range(len(examples)):
        polarities = []
        for polarity in examples[i].polarity:
            if polarity == 2:
                polarities.append(1)
            else:
                polarities.append(polarity)
        examples[i].polarity = polarities
