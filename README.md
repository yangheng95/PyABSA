# Aspect & Target Sentiment Classification Tool (Based on Local Context Focus Mechanism)

> 方面级/目标级情感分类工具 (基于局部上下文专注机制的方面级情感分类模型库)

> PyTorch Implementations.

## Requirement

* Python 3.7 + (recommended)
* PyTorch >= 1.0

## Introduction

This repository provides a simple aspect/target sentiment classification methods based a variety of APC models,
especially the those based on the local context focus mechanisms.

# Quick Start

Install this repo by `pip install pyabsa`. 

To use our models, you may need download `en_core_web_sm` by using

`python -m spacy download en_core_web_sm`

1. Train our model your in your custom dataset:

```
from pyabsa import train, load_trained_model
param_dict = {'model_name':'lcf_bert', 'lcf':'cdw', 'batch_size': 16}

# public datasets can be found in the other branch
train_set_path = 'restaurant_train.raw'  
model_path_to_save = './'
infermodel = train(param_dict, train_set_path, model_path_to_save)

```
The trained models are available [here](https://pan.baidu.com/s/1u5q8EqahXexKi2-hw_CUYg) (access code: bert), download them if necessary, 
note that the provided models are best benchmarked. If you want train a best benchmarked model,
refer to the master branch.


2. Load the trained model:

```infermodel = load_trained_model(trained_model_path)```


3. Infer on inferring set:
```
# infer a formatted text, the reference sentiment begins with !sent! is optional
text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent , the [ASP]decor[ASP] cool and understated . !sent! 1 1'
# or text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent , the [ASP]decor[ASP] cool and understated .'

infermodel.infer(text)

# batch infer from prepared datasetm
test_set_path = './rest16_test_inferring.dat' 
infermodel.batch_infer(test_set_path, save_result=True)
```

4. Convert datasets for inferring

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
    infermodel.infer(sample)
```

How to set hyper-parameters:

```
param_dict = {'model_name':'lcf_bert', 'lcf':'cdw', 'batch_size': 16}

#  default hyper-parameters:
# model_name = "slide_lcfs_bert", # optional: slide_lcf_bert, lcf_bert, lcfs_bert, bert_spc, bert_base
# optimizer = "adam"
# learning_rate = 0.00002
# pretrained_bert_name = "bert-base-uncased"
# use_dual_bert = False
# use_bert_spc = True
# max_seq_len = 80
# SRD = 3
# lcf = "cdw"
# window = "lr"
# distance_aware_window = True
# dropout = 0.1
# l2reg = 0.00001
# batch_size = 16

# parameters only for training:
# num_epoch = 3
```

We hope this repository will help you and sincerely request bug reports and Suggestions. If you like this repository you
can star or share this repository to others.

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

### Our LCF-based APC models

Please try our best models `SLIDE-LCFS-BERT` and `SLIDE-LCF-BERT`.

- **[SLIDE-LCF-BERT](modules/models/slide_lcf_bert.py)**

- **[SLIDE-LCFS-BERT](modules/models/slide_lcf_bert.py)**

- **[LCA-BERT](modules/models/lca_bert.py)**

- **[LCF-BERT](modules/models/lcf_bert.py)**

Note that GloVe-based models have been removed.

### Other famous APC models

- **[LCFS-BERT](modules/models/lcf-bert.py)**

Phan M H, Ogunbona P O. [Modelling context and syntactical features for aspect-based sentiment
analysis[C]](https://www.aclweb.org/anthology/2020.acl-main.293/)//Proceedings of the 58th Annual Meeting of the
Association for Computational Linguistics. 2020: 3211-3220.

The following models are forked from [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch).

- **[BERT-BASE](modules/models/bert_base.py)**
- **[BERT-SPC](modules/models/bert_spc.py)**

## Contributions & Bug Reports.

This Repository is under development. There may be unknown problems in the code. Please do feel free to report any
problem, and PRs are welcome.

## Citation

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
the [pytorch-transformers](https://github.com/huggingface/transformers). Thanks to the authors for their devotion and
Thanks to everyone who offered assistance. Feel free to report any bug or discussing with us.

## To Do

1. Add more bert-based models
2. Add more APIs
3. Optimize codes and add comments


