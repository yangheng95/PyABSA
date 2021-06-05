# -*- coding: utf-8 -*-
# file: atepc_config.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


atepc_param_dict = {'model_name': "lca_atepc",
                    'optimizer': "adamw",
                    'learning_rate': 0.00002,
                    'pretrained_bert_name': "bert-base-uncased",
                    'use_dual_bert': False,
                    'use_bert_spc': True,
                    'max_seq_len': 80,
                    'SRD': 3,
                    'lcf': "cdw",
                    'window': "lr",
                    'dropout': 0,
                    'l2reg': 0.00001,
                    'num_epoch': 3,
                    'batch_size': 16,
                    'initializer': 'xavier_uniform_',
                    'seed': 996,
                    'embed_dim': 768,
                    'hidden_dim': 768,
                    'polarities_dim': 2,
                    'log_step': 3,
                    'gradient_accumulation_steps': 1,
                    'evaluate_begin': 2
                    }
