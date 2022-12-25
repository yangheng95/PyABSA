PyABSA Configurations
=====================

## Introduction
PyABSA provides a set of configurations to customize the behavior of PyABSA. 
You can customize the configurations by modifying the configuration object.
A configuration object is a Namespace object that contains a set of attributes.
If you are not sure if a configuration contains a certain attribute,
you can use the following code to check the attributes of a configuration object:

```python
config.get('attribute_name', None)
```

To set or get the value of an attribute, you can use the following code:

```python
config.attribute_name = value
print(config.attribute_name)
```
You can set any variable in the configuration object as long as it is a valid Python variable name.

## Default Configurations

```python3

from pyabsa import AspectTermExtraction as ATE
from pyabsa import AspectPolarityClassification as APC

# for example, these are the default values of the configuration, you can change them to your own values
# you can refer to the task-specific configuration in pyabsa.tasks.*.configuration.*config.py for more details
# if you are developing based on the pyabsa, you can set any value you want. e.g. config.my_parameter = 'my_parameter'

transformers_based_config = {'model': ATE.ATEPCModelList.LCF_ATEPC,
                             # model class, check available models in APCModelList, ATEPCModelList and TCModelList,
                             'optimizer': "adamw",  # Optimizer class and str are both acceptable (from pytorch)
                             'learning_rate': 0.00003,
                             # The default learning of transformers-based models generally ranges in [1e-5, 5e-5]
                             'pretrained_bert': "yangheng/deberta-v3-base-absa-v1.1",
                             # The pretrained_bert accepts model from the Huggingface Hub or local model, which use the AutoModel implementation
                             'cache_dataset': True,
                             # Don't cache the dataset in development, changing a param in the config probably triggers new caching process
                             'overwrite_cache': False,  # Overwrite the cache if exists
                             'use_amp': False,  # Use automatic mixed precision training
                             'glove_or_word2vec_path': None,
                             # The path of glove or word2vec file, if None, download the glove-840B embedding file from the Internet
                             'warmup_step': -1,  # Default to not use warmup_step, this is an experimental feature
                             'use_bert_spc': False,
                             # Use [CLS] + Context + [SEP] + aspect +[SEP] input format , which is helpful in ABSA
                             'show_metric': False,
                             # Display classification report during/after training, e.g., to see precision, recall, f1-score
                             'max_seq_len': 80,
                             # The max text  input length in modeling, longer texts will be truncated
                             'patience': 5,  # The patience tells trainer to stop in the `patience`  of epochs
                             'SRD': 3,
                             # This param is for local context focus mechanism, you don't need to change this param generally
                             'use_syntax_based_SRD': False,
                             # This parameter use syntax-based SRD in all models involving LCF mechanism
                             'lcf': "cdw",  # Type of LCF mechanism, accepts 'cdm' and 'cdw'
                             'window': "lr",  # This param only effects in LSA-models, refer to the paper of LSA
                             'dropout': 0.5,  # Refer to the original paper of dropout
                             'l2reg': 0.000001,
                             # This param is related to specific model, you need try some values to find the best setting
                             'num_epoch': 10,  # If you have enough, please set it to 30-40
                             'batch_size': 16,  # If you have enough, please set it to 32 or 64
                             'initializer': 'xavier_uniform_',  # No used in transformers-based models
                             'seed': 52,  # This param accepts a integer or a list/set of integers
                             'output_dim': 2,
                             # The output dimension of the model, 2 for binary classification, 3 for ternary classification
                             'log_step': 50,  # alias for evaluate_steps. Accepts -1 (means evaluate every epoch) or an integer
                             'gradient_accumulation_steps': 1,  # Unused
                             'dynamic_truncate': True,
                             # This param applies a aspect-centered truncation instead of head truncation
                             'srd_alignment': True,
                             # for srd_alignment, try to align the tree nodes of syntax (SpaCy) and tokenization (transformers)
                             'evaluate_begin': 0  # No evaluation until epoch 'evaluate_begin', aims at saving time
                             }

glove_based_config = {'model': APC.APCModelList.FAST_LSA_T_V2,
                      # model class, check available models in APCModelList, ATEPCModelList and TCModelList,
                      'optimizer': "",
                      'learning_rate': 0.00002,
                      'cache_dataset': True,
                      'warmup_step': -1,
                      'deep_ensemble': False,
                      'use_bert_spc': True,
                      'max_seq_len': 80,
                      'patience': 99999,
                      'SRD': 3,
                      'dlcf_a': 2,  # the a in dlcf_dca_bert
                      'dca_p': 1,  # the p in dlcf_dca_bert
                      'dca_layer': 3,  # the layer in dlcf_dca_bert
                      'use_syntax_based_SRD': False,
                      'sigma': 0.3,
                      'lcf': "cdw",
                      'lsa': False,
                      'window': "lr",
                      'eta': -1,
                      'eta_lr': 0.01,
                      'dropout': 0,
                      'l2reg': 0.000001,
                      'num_epoch': 10,
                      'batch_size': 16,
                      'initializer': 'xavier_uniform_',
                      'seed': 52,
                      'output_dim': 3,
                      'log_step': 10,
                      'dynamic_truncate': True,
                      'srd_alignment': True,  # for srd_alignment
                      'evaluate_begin': 0,
                      'similarity_threshold': 1,  # disable same text check for different examples
                      'cross_validate_fold': -1,
                      'use_amp': False,
                      'overwrite_cache': False,
                      'glove_or_word2vec_path': None,
                      # The path of glove or word2vec file, if None, download the glove-840B embedding file from the Internet
                      'show_metric': False,
                      # Display classification report during/after training, e.g., to see precision, recall, f1-score
                      }

```