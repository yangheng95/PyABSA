# PyABSA - Open & Efficient for Framework for Aspect-based Sentiment Analysis

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

**PyABSA is an active project and under development, if you have the interest to integrate your models or datasets into
PyABSA, please do feel free to contact us for necessary support. Any message would receive response at least in a day.**

# Preface

This is an ASBA research-oriented code repository. I notice that some Repos do not provide the inference script, and the
codes may be redundant or hard to use, so I build PyABSA to make the training and inference easier. PyABSA contains
ATEPC and APC models now. Except for providing SOTA models for both ATEPC and APC, some source codes in PyABSA are
reusable. In another word, you can develop your model based on PyABSA. e.g., using efficient local context focus
implementation from PyASBA. Please feel free to give me your interesting thoughts, to help me build an easy-to-use
toolkit to reduce the cost of building models and reproduction in ABSA tasks.

**If PyABSA helps you, please star PyABSA in order to keep it developing and find latest released features or
tutorials.**

# Notice

The LCF is a simple and adoptive mechanism proposed for ABSA. Many models based on LCF has been proposed and achieved
SOTA performance. Developing your models based on LCF will significantly improve your ABSA models. If you are looking
for the original proposal of local context focus, please redirect to the introduction of
[LCF](https://github.com/yangheng95/PyABSA/tree/release/examples/local_context_focus). If you are looking for the
original codes of the LCF-related papers, please redirect
to [LC-ABSA / LCF-ABSA](https://github.com/yangheng95/LC-ABSA/tree/LC-ABSA)
or [LCF-ATEPC](https://github.com/XuMayi/LCF-ATEPC).

# Preliminaries

To use PyABSA, install the latest version from pip or source code:

```
pip install -U pyabsa
```

Then clone our [tutorials](examples) and have fun!

```
git clone https://github.com/yangheng95/PyABSA --depth=1

cd PyABSA/examples/aspect_polarity_classification

python sentiment_inference_chinese.py
```

# Model Support

Except for the following models, we provide a template model involving LCF vec, you can develop your model based on
the [LCF-APC](pyabsa/core/apc/models/lcf_template_apc.py) model template
or [LCF-ATEPC](pyabsa/core/atepc/models/lcf_template_atepc.py) model template.

## ATEPC

1. [LCF-ATEPC](pyabsa/core/atepc/models/lcf_atepc.py)
2. [LCF-ATEPC-LARGE](pyabsa/core/atepc/models/lcf_atepc_large.py) (Dual BERT)
2. [FAST-LCF-ATEPC](pyabsa/core/atepc/models/fast_lcf_atepc.py)
3. [LCFS-ATEPC](pyabsa/core/atepc/models/lcfs_atepc.py)
4. [LCFS-ATEPC-LARGE](pyabsa/core/atepc/models/lcfs_atepc_large.py) (Dual BERT)
5. [FAST-LCFS-ATEPC](pyabsa/core/atepc/models/fast_lcfs_atepc.py)
6. [BERT-BASE](pyabsa/core/atepc/models/bert_base_atepc.py)

## APC

1. [SLIDE-LCF-BERT *](pyabsa/core/apc/models/slide_lcf_bert.py) (Faster & Performs Better than LCF/LCFS-BERT)
2. [SLIDE-LCFS-BERT *](pyabsa/core/apc/models/slide_lcfs_bert.py) (Faster & Performs Better than LCF/LCFS-BERT)
3. [LCF-BERT](pyabsa/core/apc/models/lcf_bert.py) (Reimplemented & Enhanced)
4. [LCFS-BERT](pyabsa/core/apc/models/lcfs_bert.py) (Reimplemented & Enhanced)
5. [FAST-LCF-BERT](pyabsa/core/apc/models/fast_lcf_bert.py) (Faster with slightly performance loss)
6. [FAST_LCFS-BERT](pyabsa/core/apc/models/fast_lcfs_bert.py) (Faster with slightly performance loss)
7. [LCF-DUAL-BERT](pyabsa/core/apc/models/lcf_dual_bert.py) (Dual BERT)
8. [LCFS-DUAL-BERT](pyabsa/core/apc/models/lcfs_dual_bert.py) (Dual BERT)
9. [BERT-BASE](pyabsa/core/apc/models/bert_base.py)
10. [BERT-SPC](pyabsa/core/apc/models/bert_spc.py)
11. [LCA-Net](pyabsa/core/apc/models/lca_bert.py)
12. [DLCF-DCA-BERT](pyabsa/core/apc/models/dlcf_dca_bert.py)

'*' Copyrights Reserved, please wait for the publishing of our paper to get the introduction of them in detail.

### GloVe-based APC models

13. [AOA](pyabsa/core/apc/__glove__/models/aoa.py) (AOA_BERT)
14. [ASGCN](pyabsa/core/apc/__glove__/models/asgcn.py) (ASGCN_BERT)
15. [ATAE-LSTM](pyabsa/core/apc/__glove__/models/atae_lstm.py) (ATAE_LSTM_BERT)
16. [Cabasc](pyabsa/core/apc/__glove__/models/cabasc.py) (Cabasc_BERT)
17. [IAN](pyabsa/core/apc/__glove__/models/ian.py) (IAN_BERT)
18. [LSTM](pyabsa/core/apc/__glove__/models/lstm.py) (LSTM_BERT)
19. [MemNet](pyabsa/core/apc/__glove__/models/memnet.py) (MemNet_BERT)
20. [MGAN](pyabsa/core/apc/__glove__/models/mgan.py) (MGAN_BERT)
21. [RAM](pyabsa/core/apc/__glove__/models/ram.py) (RAM_BERT)
22. [TD-LSTM](pyabsa/core/apc/__glove__/models/td_lstm.py) (TD_LSTM_BERT)
23. [TD-LSTM](pyabsa/core/apc/__glove__/models/tc_lstm.py) (TC_LSTM_BERT)
24. [TNet_LF](pyabsa/core/apc/__glove__/models/tnet_lf.py) (TNet_LF_BERT)

### You can use $glove model name$_BERT to use the BERT version

e.g.,

```
from pyabsa.functional import BERTBaselineAPCModelList
ASGCN_BERT = BERTBaselineAPCModelList.ASGCN_BERT
```

## Brief Performance Report

|      Models          | Laptop14 (acc) |  Rest14 (acc) | Rest15 (acc) | Rest16 (acc) |
| :------------------: | :------------: | :-----------: |:------------:|:------------:|
| SLIDE-LCFS-BERT (CDW)|      81.66     |      86.68    |     85.19    |    92.36     | 
| SLIDE-LCFS-BERT (CDM)|      81.35     |      88.21    |     85.19    |    92.20     |
| SLIDE-LCF-BERT (CDW) |      81.66     |      87.59    |     84.81    |    92.03     |
| SLIDE-LCF-BERT (CDM) |      80.25     |      86.86    |     85.74    |    91.71     |

The optimal performance result among three random seeds. Note that with the update of this repo, the results could be
updated. We are working on the construction of
**[leaderboard](examples/aspect_polarity_classification/leaderboard.md)**, you can help us by reporting performance of
other models.

## How to get available checkpoints from Google Drive

PyABSA will check the latest available checkpoints before and load the latest checkpoint from Google Drive. To view
available checkpoints, you can use the following code and load the checkpoint by name:

```
from pyabsa import available_checkpoints

checkpoint_map = available_checkpoints()
```

If you can not access to Google Drive, you can download our checkpoints and load the unzipped checkpoint manually.
如果您无法访问谷歌Drive，您可以下载我们预训练的模型，并手动解压缩并加载模型。 模型下载[地址](https://pan.baidu.com/s/1oKkO7RJ6Ob9vY6flnJk3Sg) 提取码：ABSA

## How to share checkpoints (e.g., checkpoints trained on your custom dataset) with community

For resource limitation, we do not provide diversities of checkpoints, we hope you can share your checkpoints with those
who have not enough resource to train their model.

1. Upload your zipped checkpoint to Google Drive **in a shared folder**.

2. Register the checkpoint in the [checkpoint_map](examples/checkpoint_map.json), then make a pull request. We will
   update the checkpoints index as soon as we can, Thanks for your help!

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
from pyabsa.functional import ATEPCModelList
from pyabsa.functional import Trainer, ATEPCTrainer
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import ATEPCConfigManager
```

### 2. Choose a base param_dict

```
config = ATEPCConfigManager.get_atepc_config_english()
```

### 3. Specify an ATEPC model and alter some hyper-parameters (if necessary)

```
atepc_config_english = ATEPCConfigManager.get_atepc_config_english()
atepc_config_english.num_epoch = 10
atepc_config_english.evaluate_begin = 4
atepc_config_english.lot_step = 100
atepc_config_english.model = ATEPCModelList.LCF_ATEPC
```

### 4. Configure runtime setting and running training

```
laptop14 = ABSADatasetList.Laptop14

aspect_extractor = ATEPCTrainer(config=atepc_config_english,  # set config=None to use default model
                                dataset=laptop14,  # file or dir, dataset_utils(s) will be automatically detected
                                save_checkpoint=True,  # set model_path_to_save=None to avoid save model
                                auto_device=True  # Auto choose CUDA or CPU
                                )
```

### 5. Aspect term extraction & sentiment inference

```
from pyabsa import ATEPCCheckpointManager

examples = ['相比较原系列锐度高了不少这一点好与不好大家有争议',
            '这款手机的大小真的很薄，但是颜色不太好看， 总体上我很满意啦。'
            ]
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='chinese',
                                                               auto_device=True  # False means load model on CPU
                                                               )

inference_source = pyabsa.ABSADatasetList.SemEval
atepc_result = aspect_extractor.extract_aspect(inference_source=inference_source,  #
                                               save_result=True,
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )
```

# [Datasets](https://github.com/yangheng95/ABSADatasetList)

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
11. TShirt
12. Television

Basically, you don't have to download the datasets, as the datasets will be downloaded automatically.

# Acknowledgement

This work build from LC-ABSA/LCF-ABSA and LCF-ATEPC, and other impressive works such as PyTorch-ABSA and LCFS-BERT.

欢迎提出疑问、意见和建议，或者帮助完善仓库，谢谢！

# To Do

1. Add more BERT models
2. Add more APIs
3. Optimize codes and add comments

# Calling for Contribution

We hope you can help us to improve this work, e.g., provide new datasets. Or, if you **develop your model using this
PyABSA**, It is highly recommended to **release your model in PyABSA** by pull request, as open-source projects make
your work much more valuable!
We will help you to do this, only if we have some free time.

The copyrights of contributed resources belong to the contributors, we hope you can help, thanks very much!

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

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification.
Contributions of any kind welcome!
