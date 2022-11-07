# PyABSA - Open Framework for Aspect-based Sentiment Analysis

![PyPI - Python Version](https://img.shields.io/badge/python-3.6-blue.svg)
[![PyPI](https://img.shields.io/pypi/v/pyabsa)](https://pypi.org/project/pyabsa/)
[![Downloads](https://pepy.tech/badge/pyabsa)](https://pepy.tech/project/pyabsa)
[![Downloads](https://pepy.tech/badge/pyabsa/month)](https://pepy.tech/project/pyabsa)
![License](https://img.shields.io/pypi/l/pyabsa?logo=PyABSA)

[![total views](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_views.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)
[![total views per week](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_views_per_week.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)
[![total clones](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_clones.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)
[![total clones per week](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_clones_per_week.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/back-to-reality-leveraging-pattern-driven/aspect-based-sentiment-analysis-on-semeval)](https://paperswithcode.com/sota/aspect-based-sentiment-analysis-on-semeval?p=back-to-reality-leveraging-pattern-driven)

# Note: If you need to find the v1.x version, please visit [release](https://github.com/yangheng95/PyABSA/tree/release)

**Hi, there!** Please star this repo if it helps you! Each Star helps PyABSA go further, many thanks.

PyABSA is a free and open-source tool for everyone, but please do not forget to attach the (informal or formal) author
information and project address in your works, products and publications, etc.

# | [Overview](./README.MD) | [Paper](https://arxiv.org/abs/2208.01368) | [HuggingfaceHub](readme/huggingface_readme.md) | [ABDADatasets](readme/dataset_readme.md) | [ABSA Models](readme/model_readme.md) | [Colab Tutorials](readme/tutorial_readme.md) | [Troubleshooting(常见问题解决)](https://github.com/yangheng95/PyABSA/issues/189)

## Try our demos on Huggingface Space

- [(Gradio) Aspect term extraction & sentiment classification](https://huggingface.co/spaces/Gradio-Blocks/Multilingual-Aspect-Based-Sentiment-Analysis) (
  English,
  Chinese, Arabic, Dutch, French, Russian, Spanish, Turkish, etc.)
- [(Prototype) Aspect term extraction & sentiment classification](https://huggingface.co/spaces/yangheng/PyABSA-ATEPC) (
  English,
  Chinese, Arabic, Dutch, French, Russian, Spanish, Turkish, etc.)
- [方面术语提取和情感分类](https://huggingface.co/spaces/yangheng/PyABSA-ATEPC-Chinese) （中文, etc.）
- [Aspect-based sentiment classification (Multilingual)](https://huggingface.co/spaces/yangheng/PyABSA-APC) （English,
  Chinese, etc.）

## Notice

This repository is based on our papers for ABSA research. Here are the papers that you can cite or refer to for your
implementations:

<details>
<summary>
Aspect sentiment polarity classification models
</summary>

1. [Back to Reality: Leveraging Pattern-driven Modeling to Enable Affordable Sentiment Dependency Learning](https://arxiv.org/abs/2110.08604) (
   e.g., Fast-LSA, 2020)
2. [Learning for target-dependent sentiment based on local context-aware embedding](https://link.springer.com/article/10.1007/s11227-021-04047-1) (
   e.g., LCA-Net, 2020)
3. [LCF: A Local Context Focus Mechanism for Aspect-Based Sentiment Classification](https://www.mdpi.com/2076-3417/9/16/3389) (
   e.g., LCF-BERT, 2019)

</details>

<details>
<summary>
Aspect sentiment polarity classification & Aspect term extraction models
</summary>

1. [A multi-task learning model for Chinese-oriented aspect polarity classification and aspect term extraction](https://www.sciencedirect.com/science/article/pii/S0925231220312534)] (
   e.g., Fast-LCF-ATEPC, 2020)
2. [(Arxiv) A multi-task learning model for Chinese-oriented aspect polarity classification and aspect term extraction](https://arxiv.org/pdf/1912.07976.pdf)

</details>

If you are looking for the original proposal of local context focus, here are some introduction at
[here](https://github.com/yangheng95/PyABSA/tree/release/demos/documents).

## Package Overview

<table>
<tr>
    <td><b> pyabsa </b></td>
    <td> package root (including all interfaces) </td>
</tr>
<tr>
    <td><b> pyabsa.framework </b></td>
    <td> recommend interface entry</td>
</tr>
<tr>
    <td><b> pyabsa.framework.checkpoint </b></td>
    <td> checkpoint manager entry, inference model entry</td>
</tr>
<tr>
    <td><b> pyabsa.framework.dataset </b></td>
    <td> datasets entry </td>
</tr>
<tr>
    <td><b> pyabsa.framework.config </b></td>
    <td> predefined config manager </td>
</tr>
<tr>
    <td><b> pyabsa.framework.trainer </b></td>
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
import random

from pyabsa import AspectTermExtraction as ATEPC

config = ATEPC.ATEPCConfigManager.get_atepc_config_english()
config.model = ATEPC.ATEPCModelList.FAST_LCF_ATEPC
config.evaluate_begin = 0
config.max_seq_len = 512
config.pretrained_bert = 'yangheng/deberta-v3-base-absa'
config.l2reg = 1e-8
config.seed = random.randint(1, 100)
config.use_bert_spc = True
config.use_amp = False
config.cache_dataset = False

chinese_sets = ATEPC.ATEPCDatasetList.Multilingual

aspect_extractor = ATEPC.ATEPCTrainer(config=config,
                                      dataset=chinese_sets,
                                      checkpoint_save_mode=1,
                                      auto_device=True
                                      ).load_trained_model()

atepc_examples = ['But the staff was so nice to us .',
                  'But the staff was so horrible to us .',
                  r'Not only was the food outstanding , but the little ` perks \' were great .',
                  'It took half an hour to get our check , which was perfect since we could sit , have drinks and talk !',
                  'It was pleasantly uncrowded , the service was delightful , the garden adorable , '
                  'the food -LRB- from appetizers to entrees -RRB- was delectable .',
                  'How pretentious and inappropriate for MJ Grill to claim that it provides power lunch and dinners !'
                  ]
aspect_extractor.batch_predict(inference_source=atepc_examples,  #
                               save_result=True,
                               print_result=True,  # print the result
                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                               )



```

2. Inference Example of aspect term extraction

```python3
from pyabsa import AspectTermExtraction as ATEPC

# checkpoint_map = available_checkpoints(from_local=False)


aspect_extractor = ATEPC.AspectExtractor('multilingual',
                                         auto_device=True,  # False means load model on CPU
                                         cal_perplexity=True,
                                         )

inference_source = ATEPC.ATEPCDatasetList.SemEval
atepc_result = aspect_extractor.batch_predict(inference_source=inference_source,  #
                                              save_result=True,
                                              print_result=True,  # print the result
                                              pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                              )

print(atepc_result)


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

This repository is developed and maintained by HENG YANG ([yangheng95@GitHub](https://github.com/yangheng95)),
with great contribution from community researchers.
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
  hyper-parameters)
- Star this repository to keep it active

## Citation

```bibtex
@misc{YangL2022,
  title = {PyABSA: Open Framework for Aspect-based Sentiment Analysis},
  author = {Yang, Heng and Li, Ke},
  doi = {10.48550/ARXIV.2208.01368},
  url = {https://arxiv.org/abs/2208.01368},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## License

PyABSA is released under MIT licence, please cite this repo (or papers) or attach the author information in your work
(repository, blog, product, etc.)
