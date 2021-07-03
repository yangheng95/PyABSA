# PyABSA - Exploiting Fast LCF and BERT in Aspect-based Sentiment Analysis
# [English](README.md) | [中文](README_CN.md)

![PyPI - Python Version](https://img.shields.io/badge/python-3.6-blue.svg) 
[![PyPI](https://img.shields.io/pypi/v/pyabsa)](https://pypi.org/project/pyabsa/)
![Repo Size](https://img.shields.io/github/repo-size/yangheng95/pyabsa)
[![PyPI_downloads](https://img.shields.io/pypi/dm/pyabsa)](https://pypi.org/project/pyabsa/)
![License](https://img.shields.io/pypi/l/pyabsa?logo=PyABSA)
![welcome](https://img.shields.io/badge/Contribution-Welcome-brightgreen)
[![Gitter](https://badges.gitter.im/PyABSA/community.svg)](https://gitter.im/PyABSA/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->


> Fast & Low Memory requirement & Enhanced implementation of Local Context Focus.

> Build from LC-ABSA / LCF-ABSA / LCF-BERT and LCF-ATEPC.

> Provide tutorials of training and usages of ATE and APC models.

> PyTorch Implementations (CPU & CUDA supported).

# Preface

This is an ASBA research-oriented code repository. I notice that some Repos do not provide the inference script,
and the codes may be redundant or hard to use, so I build PyABSA to make the training and inference easier.
PyABSA contains ATEPC and APC models now.
Except for providing SOTA models for both ATEPC and APC, some source codes in PyABSA are reusable. 
In another word, you can develop your model based on PyABSA. 
e.g., using efficient local context focus implementation from PyASBA.
Please feel free to give me your interesting thoughts,
to help me build an easy-to-use toolkit to reduce the cost of building models and reproduction in ABSA tasks.

# Notice
The LCF is a simple and adoptive mechanism proposed for ABSA. 
Many models based on LCF has been proposed and achieved SOTA performance. 
Developing your models based on LCF will significantly improve your ABSA models.
If you are looking for the original theory of LCF, please redirect to [LCF-theory](https://github.com/yangheng95/PyABSA/tree/release/examples/local_context_focus). If you are looking for the original codes of the LCF-related papers, please redirect to [LC-ABSA / LCF-ABSA](https://github.com/yangheng95/LC-ABSA/tree/master)
or [LCF-ATEPC](https://github.com/XuMayi/LCF-ATEPC).


# Preliminaries

**Please star this repository in order to keep notified of new features or tutorials in PyABSA.**
To use PyABSA, install the latest version from pip or source code:

```
pip install -U pyabsa
```

Then clone our example and try our [tutorials](examples) and have fun! 
```
git clone https://github.com/yangheng95/PyABSA --depth=1
cd PyABSA/examples/aspect_polarity_classification
python sentiment_inference_chinese.py
```

# Model Support

Except for the following models, we provide a template model involving LCF vec, 
you can develop your model based on the [LCF-APC](pyabsa/tasks/apc/models/lcf_template_apc.py) model template 
or [LCF-ATEPC](pyabsa/tasks/atepc/models/lcf_template_atepc.py) model template.

## ATEPC
1. [LCF-ATEPC](pyabsa/tasks/atepc/models/lcf_atepc.py) 
2. [LCF-ATEPC-LARGE](pyabsa/tasks/atepc/models/lcf_atepc_large.py) 
2. [FAST-LCF-ATEPC](pyabsa/tasks/atepc/models/fast_lcf_atepc.py) 
3. [LCFS-ATEPC](pyabsa/tasks/atepc/models/lcfs_atepc.py) 
4. [LCFS-ATEPC-LARGE](pyabsa/tasks/atepc/models/lcfs_atepc_large.py) 
5. [FAST-LCFS-ATEPC](pyabsa/tasks/atepc/models/fast_lcfs_atepc.py) 
6. [BERT-BASE](pyabsa/tasks/atepc/models/bert_base_atepc.py) 

## APC

1. [SLIDE-LCF-BERT *](pyabsa/tasks/apc/models/slide_lcf_bert.py) (Faster & Performs Better than LCF/LCFS-BERT)
2. [SLIDE-LCFS-BERT *](pyabsa/tasks/apc/models/slide_lcfs_bert.py) (Faster & Performs Better than LCF/LCFS-BERT)
3. [LCF-BERT](pyabsa/tasks/apc/models/lcf_bert.py) (Reimplemented & Enhanced)
4. [LCFS-BERT](pyabsa/tasks/apc/models/lcfs_bert.py) (Reimplemented & Enhanced)
5. [FAST-LCF-BERT](pyabsa/tasks/apc/models/fast_lcf_bert.py) (Faster with slightly performance loss)
6. [FAST_LCFS-BERT](pyabsa/tasks/apc/models/fast_lcfs_bert.py) (Faster with slightly performance loss)
7. [LCF-BERT-LARGE](pyabsa/tasks/apc/models/lcf_bert_large.py) (Dual BERT)
8. [LCFS-BERT-LARGE](pyabsa/tasks/apc/models/lcf_bert_large.py) (Dual BERT)
9. [BERT-BASE](pyabsa/tasks/apc/models/bert_base.py)
10. [BERT-SPC](pyabsa/tasks/apc/models/bert_spc.py)
11. [LCA-Net](pyabsa/tasks/apc/models/lca_bert.py)

* Copyrights Reserved, please wait for the publishing of our paper to get the introduction of them in detail. 

## Brief Performance Report

|      Models          | Laptop14 (acc) |  Rest14 (acc) | Rest15 (acc) | Rest16 (acc) |
| :------------------: | :------------: | :-----------: |:------------:|:------------:|
| SLIDE-LCFS-BERT (CDW)|    81.66       |        86.68  |    85.19     |   92.36      | 
| SLIDE-LCFS-BERT (CDM)|     81.35      |        88.21   |    85.19     |   92.20      |
| SLIDE-LCF-BERT (CDW) |      81.66         |        87.59      |      84.81      |    92.03         |
| SLIDE-LCF-BERT (CDM) |    80.25          |        86.86      |   85.74          |    91.71         |

The optimal performance result among three random seeds. Note that with the update of this repo, 
the results could be updated. We are working on the construction of
**[leaderboard](examples/aspect_polarity_classification/leaderboard.md)**, 
you can help us by reporting performance of other models.


## How to get available checkpoints from Google Drive
PyABSA will check the latest available checkpoints before and load the latest checkpoint from Google Drive. 
To view available checkpoints, you can use the following code and load the checkpoint by name:
```
from pyabsa import update_checkpoints

checkpoint_map = update_checkpoints()
```

## How to share checkpoints (e.g., checkpoints trained on your custom dataset) with community

For resource limitation, we do not provide diversities of checkpoints, 
we hope you can share your checkpoints with those who have not enough resource to train their model.

1. Upload your zipped checkpoint to Google Drive **in a shared folder**.

2. Register the checkpoint in the [checkpoint_map](examples/checkpoint_map.json), 
   then make a pull request. We will update the checkpoints index as soon as we can, Thanks for your help!

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
from pyabsa import ATEPCModelList
```

### 2. Choose a base param_dict
```
param_dict = atepc_config_handler.get_apc_param_dict_chinese()
```

### 3. Specify an ATEPC model and alter some hyper-parameters (if necessary)
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
from pyabsa import APCModelList
from pyabsa import ABSADatasets
```
### 2. Choose a base param_dict
```
param_dict = apc_config_handler.get_atepc_param_dict_english()
```

### 3. Specify an APC model and alter some hyper-parameters (if necessary)
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
check [parameter introduction](examples/common_usages/param_dict_introduction.py) and learn how to set them

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

# Here I provided some pre-trained models in case of having no resource to train a model,
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

### Searching optimal hyper-parameter in the alternative parameter set. 
   You use this function to search the optimal setting of some params, e.g., learning_rate.
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
10. Multilingual (The sum of the above datasets.)

Basically, you don't have to download the datasets, as the datasets will be downloaded automatically. 


# Acknowledgement

This work build from LC-ABSA/LCF-ABSA and LCF-ATEPC, and other impressive works such as PyTorch-ABSA and LCFS-BERT. Feel free to help us optimize code or add new features!

欢迎提出疑问、意见和建议，或者帮助完善仓库，谢谢！

# To Do
1. Add more BERT / glove based models
2. Add more APIs
3. Optimize codes and add comments

# Calling for Contribution
We hope you can help us to improve this work, e.g.,
provide new datasets. Or, if you **develop your model using this PyABSA**,
It is highly recommended to **release your model in PyABSA** by pull request, 
as open-source projects make your work much more valuable!
We will help you to do this, only if we have some free time.

The copyrights of contributed resources belong to the contributors, 
we hope you can help, thanks very much!

# License 
MIT
## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/XuMayi"><img src="https://avatars.githubusercontent.com/u/50400855?v=4?s=100" width="100px;" alt=""/><br /><sub><b>XuMayi</b></sub></a><br /><a href="https://github.com/yangheng95/PyABSA/commits?author=XuMayi" title="Code">💻</a></td>
    <td align="center"><a href="https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=zh-CN"><img src="https://avatars.githubusercontent.com/u/51735130?v=4?s=100" width="100px;" alt=""/><br /><sub><b>YangHeng</b></sub></a><br /><a href="#projectManagement-yangheng95" title="Project Management">📆</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
