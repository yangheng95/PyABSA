# PyABSA - Open Framework for Aspect-based Sentiment Analysis

![PyPI - Python Version](https://img.shields.io/badge/python-3.6-blue.svg)
[![PyPI](https://img.shields.io/pypi/v/pyabsa)](https://pypi.org/project/pyabsa/)
[![PyPI_downloads](https://img.shields.io/pypi/dm/pyabsa)](https://pypi.org/project/pyabsa/)
![License](https://img.shields.io/pypi/l/pyabsa?logo=PyABSA)

[![total views](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_views.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)
[![total views per week](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_views_per_week.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)
[![total clones](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_clones.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)
[![total clones per week](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_clones_per_week.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
> Aspect Term Extraction (ATE) & Aspect Polarity Classification (APC)
> 
> Fast & Low Memory requirement & Enhanced implementation of Local Context Focus
>
> Build from LC-ABSA / LCF-ABSA / LCF-BERT and LCF-ATEPC.
> 
> PyTorch Implementations (CPU & CUDA supported). 

If you are willing to support PyABSA project, please star this repository as your contribution.
 
## 1. Package Overview

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

## 2. Read the Important Tips

### 2.1 Use your custom dataset
PyABSA use the [FindFile](https://github.com/yangheng95/findfile) to find the target file which means you can specify
  a dataset/checkpoint by keywords instead of using absolute path. e.g.,

- First, refer to [ABSADatasets](https://github.com/yangheng95/ABSADatasets) to prepare your dataset into acceptable format.
- You can PR to contribute your dataset and use it like `ABDADatasets.your_dataset`, or use it by dataset absolute / relative path, or dataset dir name
```bash
dataset = './laptop' # relative path
dataset = 'ABSOLUTE_PATH/laptop/' # absolute path
dataset = 'laptop' # dataset directory name, keyword case doesn't matter
dataset = 'lapto' # search any directory whose path contains the 'lapto' or 'aptop'

checkpoint = 'lcfs' # checkpoint assignment is similar to above methods
```

### 2.2 Auto select the free cuda for training & inference
PyABSA use the [AutoCUDA](https://github.com/yangheng95/autocuda) to support automatic cuda assignment, but you can
  still set a preferred device.

```python3
auto_device=True  # to auto assign a cuda device for training / inference
auto_device=False  # to use cpu
auto_device='cuda:1'  # to specify a preferred device
auto_device='cpu'  # to specify a preferred device
```

### 2.3 Flexible labels than others 
PyABSA encourages you to use string labels instead of numbers. e.g., sentiment labels = {negative, positive, unknown}
- What labels you labeled in the dataset, what labels will be output in inference
- The version information of PyABSA is also available in the output while loading checkpoints training args.
- You can train a model using multiple datasets with same sentiment labels, and you can even contribute and define a
  combination of datasets [here](./pyabsa/functional/dataset/dataset_manager.py#L32)!


### 2.3 Get/Set config options
The default spaCy english model is en_core_web_sm, if you didn't install it, PyABSA will download/install it automatically.

If you would like to change english model (or other pre-defined options), you can get/set as following:

```python3
from pyabsa.functional.config.apc_config_manager import APCConfigManager
from pyabsa.functional.config.atepc_config_manager import ATEPCConfigManager
from pyabsa.functional.config.classification_config_manager import ClassificationConfigManager

#Set
APCConfigManager.set_apc_config_english({'spacy_model': 'en_core_web_lg'})
ATEPCConfigManager.set_atepc_config_english({'spacy_model': 'en_core_web_lg'})
ClassificationConfigManager.set_classification_config_english({'spacy_model': 'en_core_web_lg'})

#Get
APCConfigManager.get_apc_config_english()
ATEPCConfigManager.get_atepc_config_english()
ClassificationConfigManager.get_classification_config_english()

#Manually Set spaCy nlp Language object
from pyabsa.core.apc.dataset_utils.apc_utils import configure_spacy_model
nlp = configure_spacy_model(APCConfigManager.get_apc_config_english())
```

## 3. Quick Start
- Create a new python environment and install pyabsa
- ind a target demo script ([ATEPC](https://github.com/yangheng95/PyABSA/tree/release/demos/aspect_term_extraction), [APC](https://github.com/yangheng95/PyABSA/tree/release/demos/aspect_polarity_classification), [Text Classification](https://github.com/yangheng95/PyABSA/tree/release/demos/text_classification)) to prepare your work
- Format your dataset referring to [ABSADatasets](https://github.com/yangheng95/ABSADatasets) or use public dataset in ABSADatasets
- Init your config to specify Model, Dataset, hyper-parameters
- Training your model and get checkpoints
- Share your checkpoint and dataset

## 4. Installation

Please do not install the version without corresponding release note to avoid installing a test version.

### 4.1 install via pip

To use PyABSA, install the latest version from pip or source code:

```bash
pip install -U pyabsa
```

### 4.2 install via source

```bash
git clone https://github.com/yangheng95/PyABSA --depth=1
cd PyABSA 
python setup.py install
```
## 5. Learning to Use Checkpoint

### 5.1 How to get available checkpoints from Google Drive

PyABSA will check the latest available checkpoints before and load the latest checkpoint from Google Drive. To view
available checkpoints, you can use the following code and load the checkpoint by name:

```python3
from pyabsa import available_checkpoints
checkpoint_map = available_checkpoints()  # show available checkpoints of PyABSA of current version 
```
If you can not access to Google Drive, you can download our checkpoints and load the unzipped checkpoint manually.
如果您无法访问谷歌Drive，您可以从[此处 (提取码：ABSA)](https://pan.baidu.com/s/1oKkO7RJ6Ob9vY6flnJk3Sg) 下载我们预训练的模型，并手动解压缩并加载模型。

## 5.2 How to use our pretrained checkpoints on your dataset
- [Aspect terms extraction & polarity classification](https://github.com/yangheng95/PyABSA/blob/release/demos/aspect_term_extraction/extract_aspects.py)
- [Aspect polarity classification](https://github.com/yangheng95/PyABSA/blob/release/demos/aspect_polarity_classification/sentiment_inference.py)

## 5.3 [How to share checkpoints (e.g., checkpoints trained on your custom dataset) with community](demos/documents/share-checkpoint.md)

## 6. Datasets

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
13. MOOC
14. Shampoo
15. Multilingual (The sum of all datasets.)

You don't have to download the datasets, as the datasets will be downloaded automatically.

## 7. Model Support

Except for the following models, we provide a template model involving LCF vec, you can develop your model based on
the [LCF-APC](pyabsa/core/apc/models/lcf_template_apc.py) model template
or [LCF-ATEPC](pyabsa/core/atepc/models/lcf_template_atepc.py) model template.

### 7.1 ATEPC

1. [LCF-ATEPC](pyabsa/core/atepc/models/lcf_atepc.py)
2. [LCF-ATEPC-LARGE](pyabsa/core/atepc/models/lcf_atepc_large.py) (Dual BERT)
2. [FAST-LCF-ATEPC](pyabsa/core/atepc/models/fast_lcf_atepc.py)
3. [LCFS-ATEPC](pyabsa/core/atepc/models/lcfs_atepc.py)
4. [LCFS-ATEPC-LARGE](pyabsa/core/atepc/models/lcfs_atepc_large.py) (Dual BERT)
5. [FAST-LCFS-ATEPC](pyabsa/core/atepc/models/fast_lcfs_atepc.py)
6. [BERT-BASE](pyabsa/core/atepc/models/bert_base_atepc.py)

### 7.2 APC

#### Bert-based APC models

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

#### Bert-based APC baseline models

1. [AOA_BERT](pyabsa/core/apc/classic/__bert__/models/aoa_bert.py)
2. [ASGCN_BERT](pyabsa/core/apc/classic/__bert__/models/asgcn_bert.py)
3. [ATAE_LSTM_BERT](pyabsa/core/apc/classic/__bert__/models/atae_lstm_bert.py)
4. [Cabasc_BERT](pyabsa/core/apc/classic/__bert__/models/cabasc_bert.py)
5. [IAN_BERT](pyabsa/core/apc/classic/__bert__/models/ian_bert.py)
6. [LSTM_BERT](pyabsa/core/apc/classic/__bert__/models/lstm_bert.py)
7. [MemNet_BERT](pyabsa/core/apc/classic/__bert__/models/memnet_bert.py)
8. [MGAN_BERT](pyabsa/core/apc/classic/__bert__/models/mgan_bert.py)
9. [RAM_BERT](pyabsa/core/apc/classic/__bert__/models/ram_bert.py)
10. [TD_LSTM_BERT](pyabsa/core/apc/classic/__bert__/models/td_lstm_bert.py)
11. [TC_LSTM_BERT](pyabsa/core/apc/classic/__bert__/models/tc_lstm_bert.py)
12. [TNet_LF_BERT](pyabsa/core/apc/classic/__bert__/models/tnet_lf_bert.py)

#### GloVe-based APC baseline models

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

## Contribution

We expect that you can help us improve this project, and your contributions are welcome. You can make a contribution in
many ways, including:

- Share your custom dataset in PyABSA and [ABSADatasets](https://github.com/yangheng95/ABSADatasets)
- Integrates your models in PyABSA. (You can share your models whether it is or not based on PyABSA. if you are
  interested, we will help you)
- Raise a bug report while you use PyABSA or review the code (PyABSA is a individual project driven by enthusiasm so
  your help is needed)
- Give us some advice about feature design/refactor (You can advise to improve some feature)
- Correct/Rewrite some error-messages or code comment (The comments are not written by native english speaker, you can
  help us improve documents)
- Create an example script in a particular situation (Such as specify a SpaCy model, pretrainedbert type, some
  hyperparameters)
- Star this repository to keep it active

## Notice

The LCF is a simple and adoptive mechanism proposed for ABSA. Many models based on LCF has been proposed and achieved
SOTA performance. Developing your models based on LCF will significantly improve your ABSA models. If you are looking
for the original proposal of local context focus, please redirect to the introduction of
[LCF](https://github.com/yangheng95/PyABSA/tree/release/demos/documents). If you are looking for the original codes of
the LCF-related papers, please redirect to [LC-ABSA / LCF-ABSA](https://github.com/yangheng95/LC-ABSA/tree/LC-ABSA)
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
    <td align="center"><a href="https://github.com/FrancisDacian"><img src="https://avatars.githubusercontent.com/u/24215706?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ryan</b></sub></a><br /><a href="https://github.com/yangheng95/PyABSA/commits?author=FrancisDacian" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/lpfy"><img src="https://avatars.githubusercontent.com/u/4684417?v=4?s=100" width="100px;" alt=""/><br /><sub><b>lpfy</b></sub></a><br /><a href="https://github.com/yangheng95/PyABSA/commits?author=lpfy" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification.
Contributions of any kind welcome!
