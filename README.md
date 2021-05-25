# Aspect & Target Sentiment Classification Tool

> 本仓库提供易于使用的方面级情感分类的模型，只需简单几步即可分类方面级情感。

> 基于BERT / LCF机制的方面级情感分类模型库 (CPU & CUDA supported)

> PyTorch Implementations.

if you are looking for the original code in the papers, please go to the [master](https://github.com/yangheng95/LC-ABSA/tree/master) branch.

## Requirement

* Python 3.7 + (recommended)
* PyTorch >= 1.0
* transformers >= 4.4.2

## Introduction

This repository provides aspect/target sentiment classification APC models, especially those models based on the local
context focus mechanisms.

# Usages

Check the detailed usages in [examples](./examples) directory.

详细使用方式请见[examples](./examples)目录

# Quick Start

Install this repo by `pip install pyabsa`.

To use our models, you may need download `en_core_web_sm` by

`python -m spacy download en_core_web_sm`

0. Instant train and infer on the provided datasets:

```
from pyabsa import train, train_and_evaluate, load_trained_model
dataset_path = 'datasets/laptop14'
sent_classifier = train_and_evaluate(parameter_dict=None,
                                dataset_path=dataset_path,
                                model_path_to_save=None
                                )
text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent ,' \
       ' the [ASP]decor[ASP] cool and understated . !sent! 1 1'
sent_classifier.infer(text)

```

1. Train our models on your custom dataset:

```
from pyabsa import train, train_and_evaluate, load_trained_model
# see hyper-parameters in pyabsa/main/training_configs.py
param_dict = {'model_name': 'bert_base', 'batch_size': 16, 'device': 'cuda', 'num_epoch': 1}
# train_set_path = 'datasets/restaurant15'
train_set_path = 'sum_train.dat'
model_path_to_save = 'state_dict'

sent_classifier = train_apc(parameter_dict=param_dict,    # set param_dict=None to use default model
                            dataset_path=train_set_path,  # file or dir, datasets will be automatically detected
                            model_path_to_save=model_path_to_save,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=False,   # evaluate model while training if test set is available
                            auto_device=True  # Auto choose CUDA or CPU
                            )

# Or, use both train and test set to train a bset benchmarked model, train_config_path refer a base config file whose same name paramters will be replaced by those in param_dict.
# this fucntion need both train and test set

datasets_path = 'datasets/restaurant15'  # file or dir are accepted
sent_classifier = train_apc(parameter_dict=param_dict,    # set param_dict=None to use default model
                            dataset_path=datasets_path,   # train set and test set will be automatically detected
                            model_path_to_save=model_path_to_save,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,   # evaluate model while training if test set is available
                            auto_device=True  # Auto choose CUDA or CPU
                            )
```

We provide the pretrained models
on [Google Drive](https://drive.google.com/drive/folders/1yiMTucHKy2hAx945lgzhvb9QeHvJrStC?usp=sharing)
or [百度网盘（提取码：absa）](https://pan.baidu.com/s/1FSgaSP4ubGWy0BjBQdct5w) trained on a
large assembled ABSA [dataset](examples/sum_train.dat) based on BERT-BASE-UNCASED model,

1. BERT-BASE
2. BERT-SPC
3. LCF-BERT
4. LCFS-BERT
5. SLIDE_LCF_BERT
6. SLIDE_LCFS_BERT

download them if necessary, note that the provided models are best benchmarked. If you want train a best benchmarked
model, refer to the master branch.

2. Load the trained model:

Load a trained model will also load the training parameters, however the inference batch size will always be 1.

```
from pyabsa import load_trained_model

# The trained_model_path should be a dir containing the state_dict and config file
sent_classifier = load_trained_model(trained_model_path='state_dict')

```

3. Infer on an inference set:

```
# infer a formatted text, the reference sentiment begins with !sent! is optional
text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent , the [ASP]decor[ASP] cool and understated . !sent! 1 1'
# or text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent , the [ASP]decor[ASP] cool and understated .'

# The trained_model_path should be a dir containing the state_dict and config file
sent_classifier = load_trained_model(trained_model_path='../state_dict/slide_lcfs_bert_trained')

# The default device is CPU, do specify a valid device in case of successful working

# load the model to CPU
# sent_classifier.cpu()

# load the model to CUDA (0)
# sent_classifier.cuda()

# load the model to CPU or CUDA, like cpu, cuda:0, cuda:1, etc.
sent_classifier.to('cuda:0')

sent_classifier.infer(text)

# batch infer from prepared datasetm
test_set_path = './rest16_test_inferring.dat' 
results = sent_classifier.batch_infer(test_set_path, save_result=True)

```

4. Convert datasets for inference

```
from pyabsa import convert_dataset_for_inference
convert_dataset_for_inference('datasets/semeval16')
```

5. Get usage introductions and samples:

```
from pyabsa import print_usages, samples
print_usages()
samples = get_samples()
for sample in samples:
    sent_classifier.infer(sample)
```

How to set hyper-parameters:

```
param_dict = {'model_name': 'slide_lcfs_bert',  # optional: lcf_bert, lcfs_bert, bert_spc, bert_base
              'batch_size': 16,
              # you can use a set of random seeds in train_and_evaluate function to train multiple rounds
              'seed': {0, 1, 2},
              # 'seed': 996,  # or use one seed only
              'device': 'cuda',
              'num_epoch': 6,
              'optimizer': "adam",
              'learning_rate': 0.00002,
              'pretrained_bert_name': "bert-base-uncased",
              'use_dual_bert': False,
              'use_bert_spc': True,
              'max_seq_len': 80,
              'log_step': 3,  # evaluate per steps
              'SRD': 3,
              'eta': -1,
              'sigma': 0.3,
              'lcf': "cdw",
              'window': "lr",
              'dropout': 0,
              'l2reg': 0.00001,
              }
```

# Our LCF-based APC models

We hope this repository will help you and sincerely request bug reports and suggestions. If you like this repository you
can star or share this repository to your friends.

Codes for our paper(s):

- Yang H, Zeng
  B. [Enhancing Fine-grained Sentiment Classification Exploiting Local Context Embedding[J]](https://arxiv.org/abs/2010.00767)
  . arXiv preprint arXiv:2010.00767, 2020.

- Yang H, Zeng B, Yang J, et
  al. [A multi-task learning model for Chinese-oriented aspect polarity classification and aspect term extraction[J]](https://www.sciencedirect.com/science/article/abs/pii/S0925231220312534)
  . Neurocomputing, 419: 344-356.

- Zeng B, Yang H, Xu R, et
  al. [Lcf: A local context focus mechanism for aspect-based sentiment classification[J]](https://www.mdpi.com/2076-3417/9/16/3389)
  . Applied Sciences, 2019, 9(16): 3389.

Please try our best models `SLIDE-LCFS-BERT` and `SLIDE-LCF-BERT`.

- **[SLIDE-LCF-BERT](modules/models/slide_lcf_bert.py)**

- **[SLIDE-LCFS-BERT](modules/models/slide_lcf_bert.py)**

- **[LCA-BERT](modules/models/lca_bert.py)**

- **[LCF-BERT](modules/models/lcf_bert.py)**

Note that GloVe-based models have been removed.

## Other famous APC models

- **[LCFS-BERT](modules/models/lcf-bert.py)**

Phan M H, Ogunbona P O. [Modelling context and syntactical features for aspect-based sentiment
analysis[C]](https://www.aclweb.org/anthology/2020.acl-main.293/)//Proceedings of the 58th Annual Meeting of the
Association for Computational Linguistics. 2020: 3211-3220.

The following models are forked from [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch).

- **[BERT-BASE](modules/models/bert_base.py)**
- **[BERT-SPC](modules/models/bert_spc.py)**

# Citation

If this repository is helpful to you, please cite our papers:

    @article{yang2021multi,
        title={A multi-task learning model for chinese-oriented aspect polarity classification and aspect term extraction},
        author={Yang, Heng and Zeng, Biqing and Yang, JianHao and Song, Youwei and Xu, Ruyang},
        journal={Neurocomputing},
        volume={419},
        pages={344--356},
        year={2021},
        publisher={Elsevier}
    }

    @article{zeng2019lcf,
        title={LCF: A Local Context Focus Mechanism for Aspect-Based Sentiment Classification},
        author={Zeng, Biqing and Yang, Heng and Xu, Ruyang and Zhou, Wu and Han, Xuli},
        journal={Applied Sciences},
        volume={9},
        number={16},
        pages={3389},
        year={2019},
        publisher={Multidisciplinary Digital Publishing Institute}
    }

    @misc{yang2020enhancing,
        title={Enhancing Fine-grained Sentiment Classification Exploiting Local Context Embedding}, 
        author={Heng Yang and Biqing Zeng},
        year={2020},
        eprint={2010.00767},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }

## Acknowledgement

This work is based on the repositories of [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) and
the [transformers](https://github.com/huggingface/transformers). Thanks to the authors for their devotion and Thanks to
everyone who offered assistance. Feel free to help us optimize code or add new features!
欢迎提出疑问、意见和建议，或者帮助完善仓库，谢谢！

## To Do

1. Add more bert-based models
2. Add more APIs
3. Optimize codes and add comments


