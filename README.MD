# PyABSA - Open Framework for Aspect-based Sentiment Analysis

![PyPI - Python Version](https://img.shields.io/badge/python-3.6-blue.svg)
[![PyPI](https://img.shields.io/pypi/v/pyabsa)](https://pypi.org/project/pyabsa/)
[![PyPI_downloads](https://img.shields.io/pypi/dm/pyabsa)](https://pypi.org/project/pyabsa/)
![License](https://img.shields.io/pypi/l/pyabsa?logo=PyABSA)

[![total views](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_views.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)
[![total views per week](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_views_per_week.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)
[![total clones](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_clones.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)
[![total clones per week](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_clones_per_week.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/back-to-reality-leveraging-pattern-driven/aspect-based-sentiment-analysis-on-semeval)](https://paperswithcode.com/sota/aspect-based-sentiment-analysis-on-semeval?p=back-to-reality-leveraging-pattern-driven)

**Hi, there!** Please star this repo if it helps you! Each Star helps PyABSA go further, many thanks.

# | [Overview](./README.MD) | [HuggingfaceHub](readme/huggingface_readme.md) | [ABDADatasets](readme/dataset_readme.md) | [ABSA Models](readme/model_readme.md) | [Colab Tutorials](readme/tutorial_readme.md) |

## Try our demos on Huggingface Space
- [Aspect-based sentiment classification (Multilingual)](https://huggingface.co/spaces/yangheng/PyABSA-APC) （English, Chinese, etc.）
- [Aspect term extraction & sentiment classification](https://huggingface.co/spaces/yangheng/PyABSA-ATEPC) (English, Chinese, Arabic, Dutch, French, Russian, Spanish, Turkish, etc.)
- [方面术语提取和情感分类](https://huggingface.co/spaces/yangheng/PyABSA-ATEPC-Chinese) （中文, etc.）

### About Performance
<details>
<summary>
Aspect-term extraction
</summary>

|             Models             | Laptop14 (APC-Acc) | Laptop14 (ATE-F1) | Restaurant14 (APC-Acc) | Restaurant14 (ATE-F1) |
|:------------------------------:|:------------------:|:-----------------:|:----------------------:|:---------------------:|
|     FAST-LCF-ATEPC (BERT)      |       80.72        |       85.75       |         86.40          |         89.29         |
|    FAST-LCF-ATEPC (DeBERTa)    |       83.65        |       89.03       |         90.52          |         91.77         |
| FAST-LCF-ATEPC (DeBERTa-Large) |       85.06        |       87.76       |          89.8          |         92.5          |

</details>

<details>
<summary>
Aspect-based sentiment analysis
</summary>

|          Models           | Laptop14 (Acc) | Restaurant14 (acc) |          Models           | Laptop14 (Acc) | Restaurant14 (acc) |
|:-------------------------:|:--------------:|:------------------:|:-------------------------:|:--------------:|:------------------:|
|    FAST-LSA-T(DeBERTa)    |     84.80      |       89.91        | FAST-LSA-T(DeBERTa-Large) |     86.31      |       90.86        | 
|    FAST-LSA-S(DeBERTa)    |     84.17      |       89.64        | FAST-LSA-S(DeBERTa-Large) |     86.21      |       89.38        | 
|    FAST-LSA-P(DeBERTa)    |     84.33      |       89.91        | FAST-LSA-P(DeBERTa-Large) |     86.00      |       90.33        | 
</details>


## Package Overview

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

## Installation

### install via pip

To use PyABSA, install the latest version from pip or source code:

```bash
pip install -U pyabsa
```

### install via source

```bash
git clone https://github.com/yangheng95/PyABSA --depth=1
cd PyABSA 
python setup.py install
```

## Examples

1. Train a model of aspect term extraction

```python3
from pyabsa.functional import ATEPCModelList
from pyabsa.functional import Trainer, ATEPCTrainer
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import ATEPCConfigManager

atepc_config = ATEPCConfigManager.get_atepc_config_english()

atepc_config.pretrained_bert = 'microsoft/deberta-v3-base'
atepc_config.model = ATEPCModelList.FAST_LCF_ATEPC
dataset_path = ABSADatasetList.Restaurant14
# or your local dataset: dataset_path = 'your local dataset path'

aspect_extractor = ATEPCTrainer(config=atepc_config,
                                dataset=dataset_path,
                                from_checkpoint='',  # set checkpoint to train on the checkpoint.
                                checkpoint_save_mode=1,
                                auto_device=True
                                ).load_trained_model()


```

2. Inference Example of aspect term extraction

```python3
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import ATEPCCheckpointManager

examples = ['But the staff was so nice to us .',
            'But the staff was so horrible to us .',
            r'Not only was the food outstanding , but the little ` perks \' were great .',
            'It took half an hour to get our check , which was perfect since we could sit , have drinks and talk !',
            'It was pleasantly uncrowded , the service was delightful , the garden adorable , '
            'the food -LRB- from appetizers to entrees -RRB- was delectable .',
            'How pretentious and inappropriate for MJ Grill to claim that it provides power lunch and dinners !'
            ]

inference_source = ABSADatasetList.Restaurant14
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='multilingual2')
atepc_result = aspect_extractor.extract_aspect(inference_source=inference_source,
                                               save_result=True,
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )

```

3. Get available checkpoints from Google Drive

PyABSA will check the latest available checkpoints before and load the latest checkpoint from Google Drive. To view
available checkpoints, you can use the following code and load the checkpoint by name:

```python3
from pyabsa import available_checkpoints

# The results of available_checkpoints() depend on the PyABSA version
checkpoint_map = available_checkpoints()  # show available checkpoints of PyABSA of current version 
```

If you can not access to Google Drive, you can download our checkpoints and load the unzipped checkpoint manually.

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
- Create an example script in a particular situation (Such as specify a SpaCy model, pretrained-bert type, some
  hyperparameters)
- Star this repository to keep it active

## Notice

If you are looking for the original proposal of local context focus, please redirect to the introduction of
[LCF](https://github.com/yangheng95/PyABSA/tree/release/demos/documents). If you are looking for the original codes of
the LCF-related papers, please redirect to [LC-ABSA / LCF-ABSA](https://github.com/yangheng95/LC-ABSA/tree/LC-ABSA)
or [LCF-ATEPC](https://github.com/XuMayi/LCF-ATEPC).

## Acknowledgement

This work is built from LC-ABSA/LCF-ABSA and LCF-ATEPC, and other impressive works such as PyTorch-ABSA and LCFS-BERT.
