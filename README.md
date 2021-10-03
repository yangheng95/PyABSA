# PyABSA - Open & Efficient for Framework for Aspect-based Sentiment Analysis

![PyPI - Python Version](https://img.shields.io/badge/python-3.6-blue.svg)
[![PyPI](https://img.shields.io/pypi/v/pyabsa)](https://pypi.org/project/pyabsa/)
![Repo Size](https://img.shields.io/github/repo-size/yangheng95/pyabsa)
[![PyPI_downloads](https://img.shields.io/pypi/dm/pyabsa)](https://pypi.org/project/pyabsa/)
![License](https://img.shields.io/pypi/l/pyabsa?logo=PyABSA)
[![Gitter](https://badges.gitter.im/PyABSA/community.svg)](https://gitter.im/PyABSA/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

> Fast & Low Memory requirement & Enhanced implementation of Local Context Focus.
>
> Build from LC-ABSA / LCF-ABSA / LCF-BERT and LCF-ATEPC.
>
> Provide tutorials of training and usages of ATE and APC models.
>
> PyTorch Implementations (CPU & CUDA supported).

## Tips

- PyABSA use the [FindFile](https://github.com/yangheng95/findfile) to find the target file which means you can specify a dataset/checkpoint by keywords
instead of using absolute path. e.g.,

```bash
dataset = 'laptop' # instead of './SemEval/LAPTOP'. keyword case doesn't matter
checkpoint = 'lcfs' # any checkpoint whose absolute path contains lcfs
```

- PyABSA use the [AutoCUDA](https://github.com/yangheng95/autocuda) to support automatic cuda assignment, but you can still set a preferred device.
```python3
auto_device=True  # to auto assign a cuda device for training / inference
auto_device=False  # to use cpu
auto_device='cuda:1'  # to specify a preferred device
auto_device='cpu'  # to specify a preferred device
```

- PyABSA support auto label fixing which means you can set the labels to any token (except -999), e.g., sentiment labels = {-9. 2, negative, positive}
- Check and make sure the version and datasets of checkpoint are compatible to your current PyABSA. The version information of PyABSA is also available in the output while loading checkpoints training args.
- You can train a model using multiple datasets with same sentiment labels, and you can even contribute and define a combination of datasets [here](https://github.com/yangheng95/PyABSA/blob/6defdb10b7fded79e2c989d9ddd95e6b06bebbbf/pyabsa/functional/dataset/dataset_manager.py#L32)!
- Other features are available to be found

# Instruction

If you are willing to support PyABSA project, please star this repository as your contribution. 

- [Installation](#installation)
- [Package Overview](#package-overview)
- [Quick-Start](#quick-start)
    - [Aspect Term Extraction and Polarity Classification (ATEPC)](#install-via-pip)
    - [Aspect Polarity Classification (APC)](#install-via-source)
- [Model Support](#model-support)
- [Dataset Support](https://github.com/yangheng95/ABSADatasets)
- [Make Contributions](#contribution)
- [All Examples](examples)
- [Notice for LCF-BERT & LCF-ATEPC](#notice)

# Installation

Please do not install the version without corresponding release note to avoid installing a test version.

## install via pip

To use PyABSA, install the latest version from pip or source code:

```bash
pip install -U pyabsa
```

## install via source

```bash
git clone https://github.com/yangheng95/PyABSA --depth=1
cd PyABSA 
python setup.py install
```

# Package Overview

<table>
<tr>
    <td><b> pyabsa </b></td>
    <td> package root (including all interfaces) </td>
</tr>
<tr>
    <td><b> pyabsa.functional </b></td>
    <td> recommend interface entry</td>
</tr>
<tr>
    <td><b> pyabsa.functional.checkpoint </b></td>
    <td> checkpoint manager entry, inference model entry</td>
</tr>
<tr>
    <td><b> pyabsa.functional.dataset </b></td>
    <td> datasets entry </td>
</tr>
<tr>
    <td><b> pyabsa.functional.config </b></td>
    <td> predefined config manager </td>
</tr>
<tr>
    <td><b> pyabsa.functional.trainer </b></td>
    <td> training module, every trainer return a inference model </td>
</tr>

</table>

# Quick Start

## Aspect Polarity Classification (APC)

### 1. Import necessary entries

```python3
from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList

# Get model list for Bert-based APC models
from pyabsa.functional import APCModelList

# Get model list for Bert-based APC baseline models
# from pyabsa.functional import BERTBaselineAPCModelList 

# Get model list for GloVe-based APC baseline models
# from pyabsa.functional import GloVeAPCModelList
```

### 2. Choose a base param config

```python3
# Choose a Bert-based APC models param_dict
apc_config_english = APCConfigManager.get_apc_config_english()

# Choose a Bert-based APC baseline models param_dict
# apc_config_english = APCConfigManager.get_apc_config_bert_baseline()

# Choose a GloVe-based APC baseline models param_dict
# apc_config_english = APCConfigManager.get_apc_config_glove()
```

### 3. Specify an APC model and alter some hyper-parameters (if necessary)

```python3
# Specify a Bert-based APC model
apc_config_english.model = APCModelList.SLIDE_LCFS_BERT

# Specify a Bert-based APC baseline model
# apc_config_english.model = BERTBaselineAPCModelList.ASGCN_BERT

# Specify a GloVe-based APC baseline model
# apc_config_english.model = GloVeAPCModelList.ASGCN

apc_config_english.similarity_threshold = 1
apc_config_english.max_seq_len = 80
apc_config_english.dropout = 0.5
apc_config_english.log_step = 5
apc_config_english.num_epoch = 10
apc_config_english.evaluate_begin = 4
apc_config_english.l2reg = 0.0005
apc_config_english.seed = {1, 2, 3}
apc_config_english.cross_validate_fold = -1
```

### 4. Configure runtime setting and running training

```python3
dataset_path = ABSADatasetList.SemEval #or set your local dataset
sent_classifier = Trainer(config=apc_config_english,
                          dataset=dataset_path,  # train set and test set will be automatically detected
                          checkpoint_save_mode=1,  # = None to avoid save model
                          auto_device=True  # automatic choose CUDA or CPU
                          )
```

### 5. Sentiment inference

```python3
# batch inferring_tutorials returns the results, save the result if necessary using save_result=True
inference_dataset = ABSADatasetList.SemEval # or set your local dataset
results = sent_classifier.batch_infer(target_file=inference_dataset,
                                      print_result=True,
                                      save_result=True,
                                      ignore_error=True,
                                      )
```

### 6. Sentiment inference output format (情感分类结果示例如下)

```
 I do not like too much  Windows 8  .  
Windows 8 --> Negative  Real: Negative (Correct)
Took a long time trying to decide between one with  retina display  and one without .  
retina display --> Neutral  Real: Neutral (Correct)
 It 's so nice that the  battery  last so long and that this machine has the snow lion !  
battery --> Positive  Real: Positive (Correct)
 It 's so nice that the battery last so long and that this machine has the  snow lion  !  
snow lion --> Positive  Real: Positive (Correct)
```

Check the detailed usages in [APC examples](examples/aspect_polarity_classification) directory.

## Aspect Term Extraction and Polarity Classification (ATEPC)

### 1. Import necessary entries

```python3
from pyabsa.functional import ATEPCModelList
from pyabsa.functional import Trainer, ATEPCTrainer
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import ATEPCConfigManager
```

### 2. Choose a base param config

```python3
config = ATEPCConfigManager.get_atepc_config_english()
```

### 3. Specify an ATEPC model and alter some hyper-parameters (if necessary)

```python3
atepc_config_english = ATEPCConfigManager.get_atepc_config_english()
atepc_config_english.num_epoch = 10
atepc_config_english.evaluate_begin = 4
atepc_config_english.log_step = 100
atepc_config_english.model = ATEPCModelList.LCF_ATEPC
```

### 4. Configure runtime setting and running training

```python3
laptop14 = ABSADatasetList.Laptop14

aspect_extractor = ATEPCTrainer(config=atepc_config_english, 
                                dataset=laptop14
                                )
```

### 5. Aspect term extraction & sentiment inference

```python3
from pyabsa import ATEPCCheckpointManager

examples = ['相比较原系列锐度高了不少这一点好与不好大家有争议',
            '这款手机的大小真的很薄，但是颜色不太好看， 总体上我很满意啦。'
            ]
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='chinese',
                                                               auto_device=True  # False means load model on CPU
                                                               )

inference_source = pyabsa.ABSADatasetList.SemEval
atepc_result = aspect_extractor.extract_aspect(inference_source=inference_source, 
                                               save_result=True,
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )
```

### 6. Aspect term extraction & sentiment inference output format (方面抽取及情感分类结果示例如下):

```bash
关(O) 键(O) 的(O) 时(O) 候(O) 需(O) 要(O) 表(O) 现(O) 持(O) 续(O) 影(O) 像(O) 的(O) 短(B-ASP) 片(I-ASP) 功(I-ASP) 能(I-ASP) 还(O) 是(O) 很(O) 有(O) 用(O) 的(O)
{'aspect': '短 片 功 能', 'position': '14,15,16,17', 'sentiment': '1'}
相(O) 比(O) 较(O) 原(O) 系(O) 列(O) 锐(B-ASP) 度(I-ASP) 高(O) 了(O) 不(O) 少(O) 这(O) 一(O) 点(O) 好(O) 与(O) 不(O) 好(O) 大(O) 家(O) 有(O) 争(O) 议(O)
{'aspect': '锐 度', 'position': '6,7', 'sentiment': '0'}

It(O) was(O) pleasantly(O) uncrowded(O) ,(O) the(O) service(B-ASP) was(O) delightful(O) ,(O) the(O) garden(B-ASP) adorable(O) ,(O) the(O) food(B-ASP) -LRB-(O) from(O) appetizers(B-ASP) to(O) entrees(B-ASP) -RRB-(O) was(O) delectable(O) .(O)
{'aspect': 'service', 'position': '7', 'sentiment': 'Positive'}
{'aspect': 'garden', 'position': '12', 'sentiment': 'Positive'}
{'aspect': 'food', 'position': '16', 'sentiment': 'Positive'}
{'aspect': 'appetizers', 'position': '19', 'sentiment': 'Positive'}
{'aspect': 'entrees', 'position': '21', 'sentiment': 'Positive'}
```

Check the detailed usages in [ATE examples](examples/aspect_term_extraction) directory.

# Checkpoint

## How to get available checkpoints from Google Drive

PyABSA will check the latest available checkpoints before and load the latest checkpoint from Google Drive. To view
available checkpoints, you can use the following code and load the checkpoint by name:

```python3
from pyabsa import available_checkpoints

checkpoint_map = available_checkpoints()
```

If you can not access to Google Drive, you can download our checkpoints and load the unzipped checkpoint manually.
如果您无法访问谷歌Drive，您可以下载我们预训练的模型，并手动解压缩并加载模型。 模型下载[地址](https://pan.baidu.com/s/1oKkO7RJ6Ob9vY6flnJk3Sg) 提取码：ABSA

## [How to share checkpoints (e.g., checkpoints trained on your custom dataset) with community](examples/documents/share-checkpoint.md)

## How to use checkpoints

### 1. Sentiment inference

#### 1.1 Import necessary entries

```python3
import os
from pyabsa import APCCheckpointManager, ABSADatasetList
os.environ['PYTHONIOENCODING'] = 'UTF8'
```

#### 1.2 Assume the sent_classifier and checkpoint

```python3
sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive', -999: ''}

sent_classifier = APCCheckpointManager.get_sentiment_classifier(checkpoint='dlcf-dca-bert1', #or set your local checkpoint
                                                                auto_device='cuda',  # Use CUDA if available
                                                                sentiment_map=sentiment_map
                                                                )
```

#### 1.3 Configure inferring setting

```python3
# batch inferring_tutorials returns the results, save the result if necessary using save_result=True
inference_datasets = ABSADatasetList.Laptop14 # or set your local dataset
results = sent_classifier.batch_infer(target_file=inference_datasets,
                                      print_result=True,
                                      save_result=True,
                                      ignore_error=True,
                                      )
```

### 2. Aspect term extraction & sentiment inference

#### 2.1 Import necessary entries

```python3
import os
from pyabsa import ABSADatasetList
from pyabsa import ATEPCCheckpointManager
os.environ['PYTHONIOENCODING'] = 'UTF8'
```

#### 2.2 Assume the sent_classifier and checkpoint

```python3
sentiment_map = {0: 'Negative', 1: "Neutral", 2: 'Positive', -999: ''}

aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='Laptop14', # or your local checkpoint
                                                               auto_device=True  # False means load model on CPU
                                                               )
```

#### 2.3 Configure extraction and inferring setting

```python3
# inference_dataset = ABSADatasetList.SemEval # or set your local dataset
atepc_result = aspect_extractor.extract_aspect(inference_source=inference_dataset,
                                               save_result=True,
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )
```

### 3. Train based on checkpoint

#### 3.1 Import necessary entries

```python3
from pyabsa.functional import APCCheckpointManager
from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList
```

#### 3.2 Choose a base param_dict

```python3
apc_config_english = APCConfigManager.get_apc_config_english()
```

#### 3.3 Specify an APC model and alter some hyper-parameters (if necessary)

```python3
apc_config_english.model = APCModelList.SLIDE_LCF_BERT
apc_config_english.evaluate_begin = 2
apc_config_english.similarity_threshold = 1
apc_config_english.max_seq_len = 80
apc_config_english.dropout = 0.5
apc_config_english.log_step = 5
apc_config_english.l2reg = 0.0001
apc_config_english.dynamic_truncate = True
apc_config_english.srd_alignment = True
```

#### 3.4 Configure checkpoint

```python3
# Ensure the corresponding checkpoint of trained model
checkpoint_path = APCCheckpointManager.get_checkpoint('slide-lcf-bert')
```

#### 3.5 Configure runtime setting and running training

```python3
dataset_path = ABSADatasetList.SemEval #or set your local dataset
sent_classifier = Trainer(config=apc_config_english,
                          dataset=dataset_path,
                          from_checkpoint=checkpoint_path,
                          checkpoint_save_mode=1,
                          auto_device=True
                          )
```

# Datasets

More datasets are available at [ABSADatasets](https://github.com/yangheng95/ABSADatasets).

1. Twitter
2. Laptop14
3. Restaurant14
4. Restaurant15
5. Restaurant16
6. Phone
7. Car
8. Camera
9. Notebook
10. MAMS
11. TShirt
12. Television
13. Multilingual (The sum of all datasets.)

Basically, you don't have to download the datasets, as the datasets will be downloaded automatically.

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

### Bert-based APC models

1. [SLIDE-LCF-BERT](pyabsa/core/apc/models/slide_lcf_bert.py) (Faster & Performs Better than LCF/LCFS-BERT)
2. [SLIDE-LCFS-BERT](pyabsa/core/apc/models/slide_lcfs_bert.py) (Faster & Performs Better than LCF/LCFS-BERT)
3. [LCF-BERT](pyabsa/core/apc/models/lcf_bert.py) (Reimplemented & Enhanced)
4. [LCFS-BERT](pyabsa/core/apc/models/lcfs_bert.py) (Reimplemented & Enhanced)
5. [FAST-LCF-BERT](pyabsa/core/apc/models/fast_lcf_bert.py) (Faster with slightly performance loss)
6. [FAST_LCFS-BERT](pyabsa/core/apc/models/fast_lcfs_bert.py) (Faster with slightly performance loss)
7. [LCF-DUAL-BERT](pyabsa/core/apc/models/lcf_dual_bert.py) (Dual BERT)
8. [LCFS-DUAL-BERT](pyabsa/core/apc/models/lcfs_dual_bert.py) (Dual BERT)
9. [BERT-BASE](pyabsa/core/apc/models/bert_base.py)
10. [BERT-SPC](pyabsa/core/apc/models/bert_spc.py)
11. [LCA-Net](pyabsa/core/apc/models/lca_bert.py)
12. [DLCF-DCA-BERT *](pyabsa/core/apc/models/dlcf_dca_bert.py)

### Bert-based APC baseline models

1. [AOA_BERT](pyabsa/core/apc/classic/__bert__/models/aoa.py)
2. [ASGCN_BERT](pyabsa/core/apc/classic/__bert__/models/asgcn.py)
3. [ATAE_LSTM_BERT](pyabsa/core/apc/classic/__bert__/models/atae_lstm.py)
4. [Cabasc_BERT](pyabsa/core/apc/classic/__bert__/models/cabasc.py)
5. [IAN_BERT](pyabsa/core/apc/classic/__bert__/models/ian.py)
6. [LSTM_BERT](pyabsa/core/apc/classic/__bert__/models/lstm.py)
7. [MemNet_BERT](pyabsa/core/apc/classic/__bert__/models/memnet.py)
8. [MGAN_BERT](pyabsa/core/apc/classic/__bert__/models/mgan.py)
9. [RAM_BERT](pyabsa/core/apc/classic/__bert__/models/ram.py)
10. [TD_LSTM_BERT](pyabsa/core/apc/classic/__bert__/models/td_lstm.py)
11. [TC_LSTM_BERT](pyabsa/core/apc/classic/__bert__/models/tc_lstm.py)
12. [TNet_LF_BERT](pyabsa/core/apc/classic/__bert__/models/tnet_lf.py)

### GloVe-based APC baseline models

1. [AOA](pyabsa/core/apc/classic/__glove__/models/aoa.py)
2. [ASGCN](pyabsa/core/apc/classic/__glove__/models/asgcn.py)
3. [ATAE-LSTM](pyabsa/core/apc/classic/__glove__/models/atae_lstm.py)
4. [Cabasc](pyabsa/core/apc/classic/__glove__/models/cabasc.py)
5. [IAN](pyabsa/core/apc/classic/__glove__/models/ian.py)
6. [LSTM](pyabsa/core/apc/classic/__glove__/models/lstm.py)
7. [MemNet](pyabsa/core/apc/classic/__glove__/models/memnet.py)
8. [MGAN](pyabsa/core/apc/classic/__glove__/models/mgan.py)
9. [RAM](pyabsa/core/apc/classic/__glove__/models/ram.py)
10. [TD-LSTM](pyabsa/core/apc/classic/__glove__/models/td_lstm.py)
11. [TD-LSTM](pyabsa/core/apc/classic/__glove__/models/tc_lstm.py)
12. [TNet_LF](pyabsa/core/apc/classic/__glove__/models/tnet_lf.py)

# Contribution
We expect that you can help us improve this project, and your contributions are welcome. 
You can make a contribution in many ways, including:

- Share your custom dataset in PyABSA and [ABSADatasets](https://github.com/yangheng95/ABSADatasets)
- Integrates your models in PyABSA. (You can share your models whether it is or not based on PyABSA. if you are interested, we will help you)
- Raise a bug report while you use PyABSA or review the code (PyABSA is a individual project driven by enthusiasm so your help is needed)
- Give us some advice about feature design/refactor (You can advise to improve some feature)
- Correct/Rewrite some error-messages or code comment (The comments are not written by native english speaker, you can help us improve documents)
- Create an example script in a particular situation (Such as specify a SpaCy model, pretrainedbert type, some hyperparameters)
- Star this repository to keep it active

# Notice

The LCF is a simple and adoptive mechanism proposed for ABSA. Many models based on LCF has been proposed and achieved
SOTA performance. Developing your models based on LCF will significantly improve your ABSA models. If you are looking
for the original proposal of local context focus, please redirect to the introduction of
[LCF](https://github.com/yangheng95/PyABSA/tree/release/examples/local_context_focus). If you are looking for the
original codes of the LCF-related papers, please redirect
to [LC-ABSA / LCF-ABSA](https://github.com/yangheng95/LC-ABSA/tree/LC-ABSA)
or [LCF-ATEPC](https://github.com/XuMayi/LCF-ATEPC).

## Acknowledgement

This work build from LC-ABSA/LCF-ABSA and LCF-ATEPC, and other impressive works such as PyTorch-ABSA and LCFS-BERT.

## License

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
    <td align="center"><a href="https://github.com/brightgems"><img src="https://avatars.githubusercontent.com/u/8269060?v=4?s=100" width="100px;" alt=""/><br /><sub><b>brtgpy</b></sub></a><br /><a href="#data-brightgems" title="Data">🔣</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification.
Contributions of any kind welcome!
