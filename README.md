# Aspect Term Extraction & Sentiment Classification 
# 方面术语抽取及方面情感分类工具
![PyPI - Python Version](https://img.shields.io/badge/python-3.6-blue.svg) 
[![PyPI](https://img.shields.io/pypi/v/pyabsa)](https://pypi.org/project/pyabsa/)
![Repo Size](https://img.shields.io/github/repo-size/yangheng95/pyabsa)
[![PyPI_downloads](https://img.shields.io/pypi/dm/pyabsa)](https://pypi.org/project/pyabsa/)
![License](https://img.shields.io/pypi/l/pyabsa?logo=PyABSA)
![welcome](https://img.shields.io/badge/Contribution-Welcome-brightgreen)
[![Gitter](https://badges.gitter.im/PyABSA/community.svg)](https://gitter.im/PyABSA/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

> Build from LC-ABSA / LCF-ABSA / LCF-BERT and LCF-ATEPC.

> Fast & Enhanced implementation of Local Context Focus

> Easy to use toolkit of aspect term extraction and aspect-polarity classification.

> Provide tutorials of training and using of ATE and APC models.

> PyTorch Implementations (CPU & CUDA supported).

# Preface

This is an ASBA research-oriented code repository. I notice that some repos do not provide inference script,
and the codes may be redundant or hard to use, so I build PyABSA to make the training and inference easier.
PyABSA contains ATEPC and APC models now.
Except for providing SOTA models for both ATEPC and APC, some source codes in PyABSA are reusable. 
In another word, you can develop your model based PyABSA. 
e.g., using efficient local context focus implementation from PyASBA.
Please feel free to give me your interesting thoughts,
to help me build an easy-to-use toolkit in order to reduce the 
cost of building models and reproduction in ABSA task.

# Notice

If you are looking for the original codes of the LCF-related papers, please go to
the [LC-ABSA](https://github.com/yangheng95/LC-ABSA/tree/master)
or [LCF-ATEPC](https://github.com/yangheng95/LCF-ATEPC).


# Preliminaries

Please star this repository in order to keep notified of new features or tutorials in PyABSA.
Install the latest version using：

```
pip install -U pyabsa
```

To use our (APC) models, you may need download `en_core_web_sm` by

```
python -m spacy download en_core_web_sm
```
# Model Support

The pretrained ATEPC and APC models are available on 
[Google Drive](https://drive.google.com/drive/folders/1yiMTucHKy2hAx945lgzhvb9QeHvJrStC?usp=sharing)
or [百度网盘（提取码：absa）](https://pan.baidu.com/s/1FSgaSP4ubGWy0BjBQdct5w), 
download them if necessary.

## How to update available checkpoints from Google Drive

```
from pyabsa import update_checkpoints

checkpoint_map = update_checkpoints()
```

## ATEPC
1. [LCF-ATEPC](pyabsa/module/atepc/models/lcf_atepc.py) 
2. [LCF-ATEPC-LARGE](pyabsa/module/atepc/models/lcf_atepc_large.py) 
2. [FAST-LCF-ATEPC](pyabsa/module/atepc/models/fast_lcf_atepc.py) 
3. [LCFS-ATEPC](pyabsa/module/atepc/models/lcfs_atepc.py) 
4. [LCFS-ATEPC-LARGE](pyabsa/module/atepc/models/lcfs_atepc_large.py) 
5. [FAST-LCFS-ATEPC](pyabsa/module/atepc/models/fast_lcfs_atepc.py) 
6. [BERT-BASE](pyabsa/module/atepc/models/bert_base_atepc.py) 

## APC

1. [SLIDE-LCF-BERT *](pyabsa/module/apc/models/slide_lcf_bert.py) (Faster & Performs Better than LCF/LCFS-BERT)
2. [SLIDE-LCFS-BERT *](pyabsa/module/apc/models/slide_lcfs_bert.py) (Faster & Performs Better than LCF/LCFS-BERT)
3. [LCF-BERT](pyabsa/module/apc/models/lcf_bert.py) (Reimplemented & Enhanced)
4. [LCFS-BERT](pyabsa/module/apc/models/lcfs_bert.py) (Reimplemented & Enhanced)
5. [FAST-LCF-BERT](pyabsa/module/apc/models/fast_lcf_bert.py) (Faster with slightly performance loss)
6. [FAST_LCFS-BERT](pyabsa/module/apc/models/fast_lcfs_bert.py) (Faster with slightly performance loss)
7. [LCF-BERT-LARGE](pyabsa/module/apc/models/lcf_bert_large.py) (Dual BERT)
8. [LCFS-BERT-LARGE](pyabsa/module/apc/models/lcf_bert_large.py) (Dual BERT)
9. [BERT-BASE](pyabsa/module/apc/models/bert_base.py)
10. [BERT-SPC](pyabsa/module/apc/models/bert_spc.py)
11. [LCA-Net](pyabsa/module/apc/models/lca_bert.py)


'*' Copyrights Reserved, please wait the publishing of our paper to get introduction of them in detail. 

## Brief Performance Report

|      Models          | Laptop14 (acc) |  Rest14 (acc) | Rest15 (acc) | Rest16 (acc) |
| :------------------: | :------------: | :-----------: |:------------:|:------------:|
| SLIDE-LCFS-BERT (CDW)|    81.35       |        88.04  |    85.93     |   92.52      | 
| SLIDE-LCFS-BERT (CDM)|     82.13      |        87.5   |    85.37     |   92.36      |
| SLIDE-LCF-BERT (CDW) |      -         |       -       |    -         |    -         |
| SLIDE-LCF-BERT (CDM) |    -           |        -      |   -          |    -         |

The optimal performance obtained among three random seeds. Note that the with the update of this repo, 
the results could be updated. Help me to construct of
[leaderboard](examples/aspect_polarity_classification/leaderboard.md).

I notice the importance of the reproducibility of the experimental results, 
you can use the integrated benchmark function to reproduce the results easily.
```
from pyabsa.research.benchmark.apc_benchmark import run_slide_lcf_bert_cdw, run_slide_lcf_bert_cdm

from pyabsa.research.benchmark.atepc_benchmark import run_benchmark_for_atepc_models

run_slide_lcf_bert_cdw()
run_slide_lcf_bert_cdm()

```
# Aspect Term Extraction (ATE)

## Aspect Extraction & Sentiment Inference Output Format (方面抽取及情感分类结果示例如下):

```
Sentence with predicted labels:
关(O) 键(O) 的(O) 时(O) 候(O) 需(O) 要(O) 表(O) 现(O) 持(O) 续(O) 影(O) 像(O) 的(O) 短(B-ASP) 片(I-ASP) 功(I-ASP) 能(I-ASP) 还(O) 是(O) 很(O) 有(O) 用(O) 的(O)
{'aspect': '短 片 功 能', 'position': '14,15,16,17', 'sentiment': '1'}
Sentence with predicted labels:
相(O) 比(O) 较(O) 原(O) 系(O) 列(O) 锐(B-ASP) 度(I-ASP) 高(O) 了(O) 不(O) 少(O) 这(O) 一(O) 点(O) 好(O) 与(O) 不(O) 好(O) 大(O) 家(O) 有(O) 争(O) 议(O)
{'aspect': '锐 度', 'position': '6,7', 'sentiment': '0'}

Sentence with predicted labels:
It(O) was(O) pleasantly(O) uncrowded(O) ,(O) the(O) service(B-ASP) was(O) delightful(O) ,(O) the(O) garden(B-ASP) adorable(O) ,(O) the(O) food(B-ASP) -LRB-(O) from(O) appetizers(B-ASP) to(O) entrees(B-ASP) -RRB-(O) was(O) delectable(O) .(O)
{'aspect': 'service', 'position': '7', 'sentiment': 'Positive'}
{'aspect': 'garden', 'position': '12', 'sentiment': 'Positive'}
{'aspect': 'food', 'position': '16', 'sentiment': 'Positive'}
{'aspect': 'appetizers', 'position': '19', 'sentiment': 'Positive'}
{'aspect': 'entrees', 'position': '21', 'sentiment': 'Positive'}
Sentence with predicted labels:
```

Check the detailed usages in [ATE examples](examples/aspect_term_extraction) directory.

## Quick Start

### 1. Import necessary entries
```
from pyabsa import train_atepc, atepc_config_handler
from pyabsa import ABSADatasets
from pyabsa.models import ATEPCModelList
```

### 2. Choose a base param_dict
```
param_dict = atepc_config_handler.get_apc_param_dict_chinese()
```

### 3. Specify a APC model and alter some hyper-parameters (if necessary)
```
atepc_param_dict_chinese['model'] = ATEPCModelList.LCF_ATEPC
atepc_param_dict_chinese['log_step'] = 20
atepc_param_dict_chinese['evaluate_begin'] = 5
```
### 4. Configure runtime setting and running training
```
save_path = 'state_dict'
chinese_sets = ABSADatasets.Chinese
sent_classifier = train_apc(parameter_dict=param_dict,     # set param_dict=None to use default model
                            dataset_path=chinese_sets,     # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training_tutorials if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )

```
### 5. Aspect term extraction & sentiment inference
```
from pyabsa import load_aspect_extractor
from pyabsa import ATEPCTrainedModelManager

examples = ['相比较原系列锐度高了不少这一点好与不好大家有争议',
            '这款手机的大小真的很薄，但是颜色不太好看， 总体上我很满意啦。'
            ]
model_path = ATEPCTrainedModelManager.get_checkpoint(checkpoint_name='Chinese')

sentiment_map = {0: 'Bad', 1: 'Good', -999: ''}
aspect_extractor = load_aspect_extractor(trained_model_path=model_path,
                                         sentiment_map=sentiment_map,  # optional
                                         auto_device=False             # False means load model on CPU
                                         )

atepc_result = aspect_extractor.extract_aspect(examples=examples,    # list-support only, for now
                                               print_result=True,    # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )

```

# Aspect Polarity Classification (APC)

Check the detailed usages in [APC examples](examples/aspect_polarity_classification) directory.

## Aspect-Polarity Classification Output Format (方面极性分类输出示例如下):
```
love  selena gomez  !!!! she rock !!!!!!!!!!!!!!!! and she 's cool she 's my idol 
selena gomez --> Positive  Real: Positive (Correct)
thehils Heard great things about the  ipad  for speech/communication . Educational discounts are problem best bet . Maybe Thanksgiving ? 
ipad --> Neutral  Real: Neutral (Correct)
Jamie fox , Eddie Murphy , and  barack obama  because they all are exciting , cute , and inspirational to lots of people including me !!! 
barack obama --> Positive  Real: Neutral (Wrong)
```

## Quick Start
### 1. Import necessary entries
```
from pyabsa import train_apc, apc_config_handler
from pyabsa.models import APCModelList
from pyabsa import ABSADatasets
```
### 2. Choose a base param_dict
```
param_dict = apc_config_handler.get_atepc_param_dict_english()
```

### 3. Specify a APC model and alter some hyper-parameters (if necessary)
```
apc_param_dict_english['model'] = APCModelList.SLIDE_LCF_BERT
apc_param_dict_english['evaluate_begin'] = 2  # to reduce evaluation times and save resources 
apc_param_dict_english['similarity_threshold'] = 1
apc_param_dict_english['max_seq_len'] = 80
apc_param_dict_english['dropout'] = 0.5
apc_param_dict_english['log_step'] = 5
apc_param_dict_english['l2reg'] = 0.0001
apc_param_dict_english['dynamic_truncate'] = True
apc_param_dict_english['srd_alignment'] = True
```
check [parameter introduction](examples/param_dict_introduction.py) and learn how to set them

### 4. Configure runtime setting and running training
```
laptop14 = ABSADatasets.Laptop14  # Here I use the integrated dataset, you can use your dataset instead 
sent_classifier = train_apc(parameter_dict=apc_param_dict_english, # ignore this parameter will use defualt setting
                            dataset_path=laptop14,         # datasets will be recurrsively detected in this path
                            model_path_to_save=save_path,  # ignore this parameter to avoid saving model
                            auto_evaluate=True,            # evaluate model if testset is available
                            auto_device=True               # automatic choose CUDA if any, False means always use CPU
                            )
```
### 5. Sentiment inference
```
from pyabsa import load_sentiment_classifier
from pyabsa import ABSADatasets
from pyabsa.models import APCTrainedModelManager

# 如果有需要，使用以下方法自定义情感索引到情感标签的词典， 其中-999为必需的填充， e.g.,
sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive', -999: ''}

# Here I provided some pretrained models in case of having no resource to train a model,
# you can train a model and specify the model path to infer instead 
model_path = APCTrainedModelManager.get_checkpoint(checkpoint_name='English')

sent_classifier = load_sentiment_classifier(trained_model_path=model_path,
                                            auto_device=True,  # Use CUDA if available
                                            sentiment_map=sentiment_map  # define polarity2name map
                                            )

text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent , the [ASP]decor[ASP] cool and understated . !sent! 1 1'       
# Note reference sentiment like '!sent! 1 1' are not mandatory

sent_classifier.infer(text, print_result=True)

# batch inferring_tutorials returns the results, save the result if necessary using save_result=True
inference_sets = ABSADatasets.semeval
results = sent_classifier.batch_infer(target_file=inference_sets,
                                      print_result=True,
                                      save_result=True,
                                      ignore_error=True,  # some data are broken so ignore them
                                      )
```

### Searching optimal hyper-parameter in alternative parameter set. 
   You use this function to search optimal setting of some params, e.g., learning_rate.
```
from pyabsa.research.parameter_search.search_param_for_apc import apc_param_search

from pyabsa import ABSADatasets
from pyabsa.config.apc_config import apc_config_handler

apc_param_dict_english = apc_config_handler.get_apc_param_dict_english()
apc_param_dict_english['log_step'] = 10
apc_param_dict_english['evaluate_begin'] = 2

param_to_search = ['l2reg', [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]]
apc_param_search(parameter_dict=apc_param_dict_english,
                 dataset_path=ABSADatasets.Laptop14,
                 search_param=param_to_search,
                 auto_evaluate=True,
                 auto_device=True)
```



# [Datasets](https://github.com/yangheng95/ABSADatasets)

1. Twitter 
2. Laptop14
3. Restaurant14
4. Restaurant15
5. Restaurant16
6. Phone
7. Car
8. Camera
9. Notebook
10. Multilingual (The sum of above datasets.)

Basically, you don't have to download the datasets, as the datasets will be downloaded automatically. 


# Acknowledgement

This work build from LC-ABSA/LCF-ABSA and LCF-ATEPC, and other impressive works such as PyTorch-ABSA and LCFS-BERT. Feel free to help us optimize code or add new features!

欢迎提出疑问、意见和建议，或者帮助完善仓库，谢谢！

# To Do
1. Add more bert / glove based models
2. Add more APIs
3. Optimize codes and add comments

# Calling for Contribution
We hope you can help us to improve this work, e.g.,
provide new datasets. Or, if you **develop your model using this PyABSA**,
It is highly recommended to **release your model in PyABSA** by pull request, 
as open source project make your work much more valuable!
I will help you only if I have some free time.

The copyrights of contributed resources belong to the contributors, 
I hope you can help, thanks very much!

# Citation
If PyABSA is helpful, please star this repo and cite our paper:

- paper of LCF-ATEPC:
```
    @article{yang2021multi,
        title={A multi-task learning model for chinese-oriented aspect polarity classification and aspect term extraction},
        author={Yang, Heng and Zeng, Biqing and Yang, JianHao and Song, Youwei and Xu, Ruyang},
        journal={Neurocomputing},
        volume={419},
        pages={344--356},
        year={2021},
        publisher={Elsevier}
    }
```
- paper of LCF-BERT:
```
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
```
- paper of LCA-Net:
```    
    @misc{yang2020enhancing,
        title={Enhancing Fine-grained Sentiment Classification Exploiting Local Context Embedding}, 
        author={Heng Yang and Biqing Zeng},
        year={2020},
        eprint={2010.00767},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
```
