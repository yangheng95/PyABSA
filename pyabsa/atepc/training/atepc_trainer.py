# -*- coding: utf-8 -*-
# file: train_atepc.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


import os
import random
import pickle
import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from seqeval.metrics import classification_report
from sklearn.metrics import f1_score
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
# from transformers import BertTokenizer
# from transformers.models.bert.modeling_bert import BertModel
from transformers import AutoTokenizer
from transformers import AutoModel

from ..dataset_utils.data_utils_for_training import ATEPCProcessor, convert_examples_to_features
from ..models.lcf_atepc import LCF_ATEPC
from ..models.rlcf_atepc import RLCF_ATEPC
from pyabsa.utils.logger import get_logger


def train4atepc(config):
    log_name = '{}_{}_{}'.format(config.model_name, config.lcf, config.SRD)
    logger = get_logger(os.getcwd(), log_name=log_name, log_type='training_tutorials')

    import warnings
    warnings.filterwarnings('ignore')

    opt = config

    if opt.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            opt.gradient_accumulation_steps))

    opt.batch_size = opt.batch_size // opt.gradient_accumulation_steps

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    if opt.model_path_to_save and not os.path.exists(opt.model_path_to_save):
        os.makedirs(opt.model_path_to_save)

    optimizers = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW
    }

    tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_bert_name, do_lower_case=True)
    bert_base_model = AutoModel.from_pretrained(opt.pretrained_bert_name)
    processor = ATEPCProcessor(tokenizer)
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    bert_base_model.config.num_labels = num_labels

    for arg in vars(opt):
        logger.info('>>> {0}: {1}'.format(arg, getattr(opt, arg)))

    train_examples = processor.get_train_examples(opt.dataset_file['train'], 'train')
    num_train_optimization_steps = int(
        len(train_examples) / opt.batch_size / opt.gradient_accumulation_steps) * opt.num_epoch
    train_features = convert_examples_to_features(train_examples, label_list, opt.max_seq_len, tokenizer, opt)
    all_spc_input_ids = torch.tensor([f.input_ids_spc for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
    all_polarities = torch.tensor([f.polarity for f in train_features], dtype=torch.long)
    lcf_cdm_vec = torch.tensor([f.lcf_cdm_vec for f in train_features], dtype=torch.float32)
    lcf_cdw_vec = torch.tensor([f.lcf_cdw_vec for f in train_features], dtype=torch.float32)

    train_data = TensorDataset(all_spc_input_ids, all_segment_ids, all_input_mask, all_label_ids,
                               all_polarities, all_valid_ids, all_lmask_ids, lcf_cdm_vec, lcf_cdw_vec)

    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=opt.batch_size)

    if 'test' in opt.dataset_file:
        eval_examples = processor.get_test_examples(opt.dataset_file['test'], 'test')
        eval_features = convert_examples_to_features(eval_examples, label_list, opt.max_seq_len, tokenizer, opt)
        all_spc_input_ids = torch.tensor([f.input_ids_spc for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_polarities = torch.tensor([f.polarity for f in eval_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        lcf_cdm_vec = torch.tensor([f.lcf_cdm_vec for f in eval_features], dtype=torch.float32)
        lcf_cdw_vec = torch.tensor([f.lcf_cdw_vec for f in eval_features], dtype=torch.float32)
        eval_data = TensorDataset(all_spc_input_ids, all_segment_ids, all_input_mask, all_label_ids, all_polarities,
                                  all_valid_ids, all_lmask_ids, lcf_cdm_vec, lcf_cdw_vec)
        # all_tokens = [f.tokens for f in eval_features]

        eval_sampler = RandomSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=opt.batch_size)

    def train():

        logger.info("***** Running training_tutorials *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", opt.batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        max_apc_test_acc = 0
        max_apc_test_f1 = 0
        max_ate_test_f1 = 0
        global_step = 0
        save_path = ''
        for epoch in range(int(opt.num_epoch)):
            nb_tr_examples, nb_tr_steps = 0, 0
            iterator = tqdm.tqdm(train_dataloader)
            for step, batch in enumerate(iterator):
                model.train()
                batch = tuple(t.to(opt.device) for t in batch)
                input_ids_spc, segment_ids, input_mask, label_ids, polarity, \
                valid_ids, l_mask, lcf_cdm_vec, lcf_cdw_vec = batch
                loss_ate, loss_apc = model(input_ids_spc,
                                           token_type_ids=segment_ids,
                                           attention_mask=input_mask,
                                           labels=label_ids,
                                           polarity=polarity,
                                           valid_ids=valid_ids,
                                           attention_mask_label=l_mask,
                                           lcf_cdm_vec=lcf_cdm_vec,
                                           lcf_cdw_vec=lcf_cdw_vec
                                           )
                loss = loss_ate + loss_apc
                loss.backward()
                nb_tr_examples += input_ids_spc.size(0)
                nb_tr_steps += 1
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                global_step += 1
                if 'test' in opt.dataset_file and global_step % opt.log_step == 0:
                    if epoch >= opt.evaluate_begin:
                        apc_result, ate_result = evaluate(
                            eval_ATE=not (opt.model_name == 'lcf_atepc' and opt.use_bert_spc))
                        # if save_path:
                        #     try:
                        #         shutil.rmtree(save_path)
                        #         # logger.info('Remove sub-optimal trained model:', save_path)
                        #     except:
                        #         logger.info('Can not remove sub-optimal trained model:', save_path)
                        if opt.model_path_to_save:
                            save_path = '{0}/{1}_{2}_apcacc_{3}_apcf1_{4}_atef1_{5}/'.format(
                                opt.model_path_to_save,
                                opt.model_name,
                                opt.lcf,
                                round(apc_result['max_apc_test_acc'], 2),
                                round(apc_result['max_apc_test_f1'], 2),
                                round(ate_result, 2)
                            )
                            if apc_result['max_apc_test_acc'] > max_apc_test_acc or \
                                    apc_result['max_apc_test_f1'] > max_apc_test_f1 or \
                                    ate_result > max_ate_test_f1:
                                _save_model(opt, model, save_path, mode=0)

                        if apc_result['max_apc_test_acc'] > max_apc_test_acc:
                            max_apc_test_acc = apc_result['max_apc_test_acc']
                        if apc_result['max_apc_test_f1'] > max_apc_test_f1:
                            max_apc_test_f1 = apc_result['max_apc_test_f1']
                        if ate_result > max_ate_test_f1:
                            max_ate_test_f1 = ate_result

                        current_apc_test_acc = apc_result['max_apc_test_acc']
                        current_apc_test_f1 = apc_result['max_apc_test_f1']
                        current_ate_test_f1 = round(ate_result, 2)

                        postfix = 'Epoch:{} | '.format(epoch)

                        postfix += 'loss_apc:{:.4f} | loss_ate:{:.4f} |'.format(loss_apc.item(), loss_ate.item())

                        postfix += f' APC_ACC: {current_apc_test_acc}(max:{max_apc_test_acc}) | ' \
                                   f' APC_F1: {current_apc_test_f1}(max:{max_apc_test_f1}) | '

                        if opt.model_name == 'lcf_atepc' and opt.use_bert_spc:
                            postfix += f'ATE_F1: N.A. for LCF-ATEPC under use_bert_spc=True)'
                        else:
                            postfix += f'ATE_F1: {current_ate_test_f1}(max:{max_ate_test_f1})'
                    else:
                        postfix = 'Epoch:{} | No evaluate until epoch:{}'.format(epoch, opt.evaluate_begin)

                    iterator.postfix = postfix
                    iterator.refresh()

        logger.info('------------------------------------Training Summary------------------------------------')
        logger.info('|Max APC Accuracy: {:.15f} Max APC F1: {:.15f} Max ATE F1: {}|'.format(max_apc_test_acc * 100,
                                                                                              max_apc_test_f1 * 100,
                                                                                              max_ate_test_f1)
                        )
        logger.info('------------------------------------Training Summary------------------------------------')
        # return the model paths of multiple training_tutorials in case of loading the best model after training_tutorials
        if save_path:
            return save_path
        else:
            # direct return model if do not evaluate
            if opt.model_path_to_save:
                save_path = '{0}/{1}_{2}/'.format(opt.model_path_to_save,
                                                  opt.model_name,
                                                  opt.lcf,
                                                  )
                _save_model(opt, model, save_path, mode=0)
            return model, opt

    def evaluate(eval_ATE=True, eval_APC=True):
        apc_result = {'max_apc_test_acc': 0, 'max_apc_test_f1': 0}
        ate_result = 0
        y_true = []
        y_pred = []
        n_test_correct, n_test_total = 0, 0
        test_apc_logits_all, test_polarities_all = None, None
        model.eval()
        label_map = {i: label for i, label in enumerate(label_list, 1)}

        for i, batch in enumerate(eval_dataloader):
            input_ids_spc, segment_ids, input_mask, label_ids, polarity, \
            valid_ids, l_mask, lcf_cdm_vec, lcf_cdw_vec = batch

            input_ids_spc = input_ids_spc.to(opt.device)
            input_mask = input_mask.to(opt.device)
            segment_ids = segment_ids.to(opt.device)
            valid_ids = valid_ids.to(opt.device)
            label_ids = label_ids.to(opt.device)
            polarity = polarity.to(opt.device)
            l_mask = l_mask.to(opt.device)
            lcf_cdm_vec = lcf_cdm_vec.to(opt.device)
            lcf_cdw_vec = lcf_cdw_vec.to(opt.device)

            with torch.no_grad():
                ate_logits, apc_logits = model(input_ids_spc,
                                               token_type_ids=segment_ids,
                                               attention_mask=input_mask,
                                               labels=None,
                                               polarity=polarity,
                                               valid_ids=valid_ids,
                                               attention_mask_label=l_mask,
                                               lcf_cdm_vec=lcf_cdm_vec,
                                               lcf_cdw_vec=lcf_cdw_vec
                                               )
            if eval_APC:
                n_test_correct += (torch.argmax(apc_logits, -1) == polarity).sum().item()
                n_test_total += len(polarity)

                if test_polarities_all is None:
                    test_polarities_all = polarity
                    test_apc_logits_all = apc_logits
                else:
                    test_polarities_all = torch.cat((test_polarities_all, polarity), dim=0)
                    test_apc_logits_all = torch.cat((test_apc_logits_all, apc_logits), dim=0)

            if eval_ATE:
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
                               labels=list(range(opt.polarities_dim)), average='macro')

            test_acc = round(test_acc * 100, 2)
            test_f1 = round(test_f1 * 100, 2)
            apc_result = {'max_apc_test_acc': test_acc, 'max_apc_test_f1': test_f1}

        if eval_ATE:
            report = classification_report(y_true, y_pred, digits=4)
            tmps = report.split()
            ate_result = round(float(tmps[7]) * 100, 2)
        return apc_result, ate_result

    # init the model behind the convert_examples_to_features function in case of updating polarities_dim
    model_classes = {
        'lcf_atepc': LCF_ATEPC,
        'rlcf_atepc': RLCF_ATEPC
    }
    model = model_classes[opt.model_name](bert_base_model, opt=opt)
    model.to(opt.device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': opt.l2reg},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': opt.l2reg}
    ]
    optimizer = optimizers[opt.optimizer](optimizer_grouped_parameters, lr=opt.learning_rate, weight_decay=opt.l2reg)

    def _save_model(args_to_save, model_to_save, save_path, mode=0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model_to_save.module if hasattr(model_to_save,
                                                        'module') else model_to_save  # Only save the model it-self

        if mode == 0:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # torch.save(model_to_save.cpu().state_dict(),
            #            save_path + args_to_save.model_name + '.state_dict')  # save the state dict
            torch.save(model_to_save.cpu(), save_path + args_to_save.model_name + '.model')  # save the whole model
            pickle.dump(args_to_save, open(save_path + args_to_save.model_name + '.config', 'wb'))
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

        # logger.info('trained model saved in: {}'.format(save_path))
        model_to_save.to(args_to_save.device)

    return train()
