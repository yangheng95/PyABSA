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
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->


> 快捷的 & 低内存需求的 & 局部上下文焦机制的增强实现。

> 构建于 LC-ABSA / LCF-ABSA / LCF-BERT 以及 LCF-ATEPC。

> 提供ATE和APC模型的训练和使用教程。

> PyTorch 实现(CPU & CUDA 支持).

# 前言

这是一个面向ABSA研究的代码库。我注意到有些repo不提供推理脚本，
并且代码存在冗余以及难以使用的问题，为了使模型训练和推理更容易，我们构建了PyABSA。
PyABSA现在包含ATEPC和APC模型。
除了为ATEPC和APC提供SOTA模型之外，PyABSA中的一些源代码是可以作为您构建自己的模型的基础。
您可以基于PyABSA快捷且高效地开发您的模型。
例如，从PyABSA中使用高效的局部上下文聚焦机制作为您开发的基础。
如果您有有趣的想法以及任何问题，欢迎随时告知我们，
欢迎帮助我们构建一个易于使用的工具包，以降低在ABSA任务中构建和复制模型的成本。

# 公告
LCF是为ABSA提出的一个简单高效且易于使用的机制。
许多基于LCF的模型已经被提出并实现了SOTA性能。
基于LCF开发模型将显著改进ABSA模型。
如果您在寻找LCF的原始理论，请查看 [LCF-theory](https://github.com/yangheng95/PyABSA/tree/release/examples/local_context_focus). 如果您正在寻找与LCF相关的论文的原始代码，请跳转到 [LC-ABSA / LCF-ABSA](https://github.com/yangheng95/LC-ABSA/tree/master)
或者[LCF-ATEPC](https://github.com/XuMayi/LCF-ATEPC).


# 试用

**如果您认为此库对您有帮助，您可以给这个库打上一个小星星，以便随时接受PyABSA中的新特性或教程的通知。**
如果您需要使用PyABSA，请从pip或源代码安装最新版本:

```
pip install -U pyabsa
```

然后复制我们的示例并尝试我们的教程(示例)，并获得乐趣!
```
git clone https://github.com/yangheng95/PyABSA --depth=1
cd PyABSA/examples/aspect_polarity_classification
python sentiment_inference_chinese.py
```

# 模型支持

除了下面的模型之外，我们提供了一个包含LCF vec的模板模型，
您可以基于 [LCF-APC](pyabsa/tasks/apc/models/lcf_template_apc.py) 模型模板 
或 [LCF-ATEPC](pyabsa/tasks/atepc/models/lcf_template_atepc.py) 模型模版来开始你的创新.

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

* 版权所有，请等待我们的论文发表以获取详细的介绍。

## 模型性能表现

|      Models          | Laptop14 (acc) |  Rest14 (acc) | Rest15 (acc) | Rest16 (acc) |
| :------------------: | :------------: | :-----------: |:------------:|:------------:|
| SLIDE-LCFS-BERT (CDW)|    81.66       |        86.68  |    85.19     |   92.36      | 
| SLIDE-LCFS-BERT (CDM)|     81.35      |        88.21   |    85.19     |   92.20      |
| SLIDE-LCF-BERT (CDW) |      81.66         |        87.59      |      84.81      |    92.03         |
| SLIDE-LCF-BERT (CDM) |    80.25          |        86.86      |   85.74          |    91.71         |

上述结果取自三种随机种子的最佳性能结果。 我们会在版本更新的过程中及时更新上述性能表现。我们正在构建一个面向APC的
**[leaderboard](examples/aspect_polarity_classification/leaderboard.md)**, 
您可以通过告知我们其他模型的性能来帮助我们完善这个排行榜.


## 如何从谷歌驱动器获得可用的checkpoints
PyABSA将检查最新可用checkpoints，并从谷歌驱动器加载最新checkpoints。
要查看可用checkpoints，可以使用以下代码并按名称加载checkpoints:
```
from pyabsa import update_checkpoints

checkpoint_map = update_checkpoints()
```

## 如何分享你的checkpoints (例如在您构建的数据集上训练的checkpoints)到社区之中

由于资源限制，我们无法提供充足的checkpoints，
我们希望您能与那些没有足够资源来训练他们的模型的人分享您的checkpoints。

1. 上传你的压缩的checkpoint 到 Google Drive **至 shared folder**.

2. 在[checkpoint_map](examples/checkpoint_map.json)中注册您的checkpoints, 
  然后提交pull request。我们会尽快更新checkpoint索引，谢谢您的帮助!

# 方面词抽取 (ATE)

## 方面抽取及情感分类结果示例如下:

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

在 [ATE examples](examples/aspect_term_extraction) 目录中查看详细的用法.

## 快速启动

### 1. 加载必要的包
```
from pyabsa import train_atepc, atepc_config_handler
from pyabsa import ABSADatasets
from pyabsa import ATEPCModelList
```

### 2. 选择一个基本的param_dict
```
param_dict = atepc_config_handler.get_apc_param_dict_chinese()
```

### 3. 指定一个ATEPC模型并在您需要的情况下更改一些超参数
```
atepc_param_dict_chinese['model'] = ATEPCModelList.LCF_ATEPC
atepc_param_dict_chinese['log_step'] = 20
atepc_param_dict_chinese['evaluate_begin'] = 5
```
### 4. 配置运行时的设置和运行训练
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
### 5. 方面词抽取 & 情感推断
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

# 方面级情感分类 (APC)

在 [APC examples](examples/aspect_polarity_classification) 目录中查看详细的用法.

## 方面极性分类输出示例如下:
```
love  selena gomez  !!!! she rock !!!!!!!!!!!!!!!! and she 's cool she 's my idol 
selena gomez --> Positive  Real: Positive (Correct)
thehils Heard great things about the  ipad  for speech/communication . Educational discounts are problem best bet . Maybe Thanksgiving ? 
ipad --> Neutral  Real: Neutral (Correct)
Jamie fox , Eddie Murphy , and  barack obama  because they all are exciting , cute , and inspirational to lots of people including me !!! 
barack obama --> Positive  Real: Neutral (Wrong)
```

## 快速启动
### 1. 加载必要的包
```
from pyabsa import train_apc, apc_config_handler
from pyabsa import APCModelList
from pyabsa import ABSADatasets
```
### 2. 选择一个基本的param_dict
```
param_dict = apc_config_handler.get_atepc_param_dict_english()
```

### 3. 指定一个APC模型并在您需要的情况下更改一些超参数
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
查看 [parameter introduction](examples/common_usages/param_dict_introduction.py) 并学习如何设置它们

### 4.  配置运行时的设置和运行训练
```
laptop14 = ABSADatasets.Laptop14  # Here I use the integrated dataset, you can use your dataset instead 
sent_classifier = train_apc(parameter_dict=apc_param_dict_english, # ignore this parameter will use defualt setting
                            dataset_path=laptop14,         # datasets will be recurrsively detected in this path
                            model_path_to_save=save_path,  # ignore this parameter to avoid saving model
                            auto_evaluate=True,            # evaluate model if testset is available
                            auto_device=True               # automatic choose CUDA if any, False means always use CPU
                            )
```
### 5.  情感推断
```
from pyabsa import load_sentiment_classifier
from pyabsa import ABSADatasets
from pyabsa.models import APCTrainedModelManager

# 如果有需要，使用以下方法自定义情感索引到情感标签的词典， 其中-999为必需的填充， e.g.,
sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive', -999: ''}

# 在没有资源来训练模型的情况下，我在这里提供了一些预先训练的模型，
# 您可以训练模型并指定模型路径来进行推断 
model_path = APCTrainedModelManager.get_checkpoint(checkpoint_name='English')

sent_classifier = load_sentiment_classifier(trained_model_path=model_path,
                                            auto_device=True,  # Use CUDA if available
                                            sentiment_map=sentiment_map  # define polarity2name map
                                            )

text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent , the [ASP]decor[ASP] cool and understated . !sent! 1 1'       
# Note reference sentiment like '!sent! 1 1' are not mandatory

sent_classifier.infer(text, print_result=True)

# 批处理inferring_tutorials返回结果, 如果需要保存结果，请设置 save_result=True
inference_sets = ABSADatasets.semeval
results = sent_classifier.batch_infer(target_file=inference_sets,
                                      print_result=True,
                                      save_result=True,
                                      ignore_error=True,  # some data are broken so ignore them
                                      )
```

### 在备选参数集中寻找最优超参数。
   您可以使用此函数搜索一些参数的最佳设置，例如learning_rate。
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



# [数据集](https://github.com/yangheng95/ABSADatasets)

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

您不需要手动下载数据集，PyABSA会自动下载数据集。


# 致谢

这项工作是在LC-ABSA、LCF-ABSA、LCF-ATEPC，以及其他优秀的代码库，如PyTorch-ABSA和LCFS-BERT的基础上完成构建的。欢迎随时帮助我们优化代码或添加新功能!

欢迎提出疑问、意见和建议，或者帮助完善仓库，谢谢！

# 未来计划
1. 添加 BERT / glove 相关基础语言模型
2. 增加更多的APIs
3. 优化代码并添加更多注释

# 期待您的贡献
我们期待您能帮助我们改进这项工作，例如:
提供新的数据集。或者使用**PyABSA开发你的模型**，
我们非常欢迎您**在PyABSA中通过pull request开源您的模型**，
开源项目会让您的工作更有价值!
只要我们有空闲时间，我们非常乐意协助您完成您的开源工作。

供稿资源的著作权属于供稿人，
希望能得到您的帮助，非常感谢!

# 许可证
MIT
## 贡献者 ✨

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
