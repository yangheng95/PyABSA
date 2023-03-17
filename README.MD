# [PyABSA - Open Framework for Aspect-based Sentiment Analysis](https://arxiv.org/pdf/2208.01368)

![PyPI - Python Version](https://img.shields.io/badge/python-3.6-blue.svg)
[![PyPI](https://img.shields.io/pypi/v/pyabsa)](https://pypi.org/project/pyabsa/)
[![Downloads](https://pepy.tech/badge/pyabsa)](https://pepy.tech/project/pyabsa)
[![Downloads](https://pepy.tech/badge/pyabsa/month)](https://pepy.tech/project/pyabsa)
![License](https://img.shields.io/pypi/l/pyabsa?logo=PyABSA)
[![Documentation Status](https://readthedocs.org/projects/pyabsa/badge/?version=v2)](https://pyabsa.readthedocs.io/en/v2/?badge=v2)


[![total views](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_views.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)
[![total views per week](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_views_per_week.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)
[![total clones](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_clones.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)
[![total clones per week](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_clones_per_week.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/back-to-reality-leveraging-pattern-driven/aspect-based-sentiment-analysis-on-semeval)](https://paperswithcode.com/sota/aspect-based-sentiment-analysis-on-semeval?p=back-to-reality-leveraging-pattern-driven)

**Hi, there!** Please star this repo if it helps you! Each Star helps PyABSA go further, many thanks. PyABSA is a free
and open-source tool for everyone, but please do not forget to attach the (informal or formal) author
information and project address in your works, products and publications, etc.

## Try our demos on Huggingface Space

Apart from the [paper](https://arxiv.org/pdf/2208.01368), there are two new features in PyABSA: Aspect sentiment triplet extraction and Aspect quadruple extraction.
We have deployed the demos on Huggingface Space, you can try them online.
- **[Aspect sentiment quadruple extraction](https://huggingface.co/spaces/yangheng/PyABSA) (English) New feature**
- **[Aspect sentiment triplet extraction](https://huggingface.co/spaces/yangheng/PyABSA) (English) New feature**
- [(Gradio) Aspect term extraction & sentiment classification](https://huggingface.co/spaces/Gradio-Blocks/Multilingual-Aspect-Based-Sentiment-Analysis) (
  English, Chinese, Arabic, Dutch, French, Russian, Spanish, Turkish, etc.)
- [(Prototype) Aspect term extraction & sentiment classification](https://huggingface.co/spaces/yangheng/PyABSA-ATEPC) (
  English,
  Chinese, Arabic, Dutch, French, Russian, Spanish, Turkish, etc.)
- [方面术语提取和情感分类](https://huggingface.co/spaces/yangheng/PyABSA-ATEPC-Chinese) （中文, etc.）
- [Aspect-based sentiment classification (Multilingual)](https://huggingface.co/spaces/yangheng/PyABSA-APC) （English,
  Chinese, etc.）

## Usage Examples
We have prepared many examples for different tasks. Please refer to [Examples](./examples-v2) for more usage examples.

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

## Quick-Start

### Extract aspect terms and classify sentiments

```python3
from pyabsa import AspectTermExtraction as ATEPC, available_checkpoints

# you can view all available checkpoints by calling available_checkpoints()
checkpoint_map = available_checkpoints()

aspect_extractor = ATEPC.AspectExtractor('multilingual',
                                         auto_device=True,  # False means load model on CPU
                                         cal_perplexity=True,
                                         )

# instance inference
aspect_extractor.predict(['I love this movie, it is so great!'],
                         save_result=True,
                         print_result=True,  # print the result
                         ignore_error=True,  # ignore the error when the model cannot predict the input
                         )

inference_source = ATEPC.ATEPCDatasetList.Restaurant16
atepc_result = aspect_extractor.batch_predict(target_file=inference_source,  #
                                              save_result=True,
                                              print_result=True,  # print the result
                                              pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                              )

print(atepc_result)


```
### Aspect-based sentiment analysis

```python3
from pyabsa import AspectPolarityClassification as APC, available_checkpoints

# you can view all available checkpoints by calling available_checkpoints()
checkpoint_map = available_checkpoints(show_ckpts=True)

classifier = APC.SentimentClassifier('multilingual',
                                     auto_device=True,  # False means load model on CPU
                                     cal_perplexity=True,
                                     )

# instance inference
classifier.predict(['I love this movie, it is so great!'],
                   save_result=True,
                   print_result=True,  # print the result
                   ignore_error=True,  # ignore the error when the model cannot predict the input
                   )

inference_source = APC.APCDatasetList.Laptop14
apc_result = classifier.batch_predict(target_file=inference_source,  #
                                      save_result=True,
                                      print_result=True,  # print the result
                                      pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                      )

print(apc_result)

```
## Dataset Annotation and Model Training
please refer to the documentation: [PyABSA Documentation](https://pyabsa.readthedocs.io/en/v2/).
If you have any questions about the docs, please feel free to raise an issue. Also, you can join to improve the docs.

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

## Citation
```bibtex
@article{YangL22,
  author    = {Heng Yang and
               Ke Li},
  title     = {PyABSA: Open Framework for Aspect-based Sentiment Analysis},
  journal   = {CoRR},
  volume    = {abs/2208.01368},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2208.01368},
  doi       = {10.48550/arXiv.2208.01368},
  eprinttype = {arXiv},
  eprint    = {2208.01368},
  timestamp = {Tue, 08 Nov 2022 21:46:32 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2208-01368.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
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



## License

PyABSA is released under MIT licence, please cite this repo (or papers) or attach the author information in your work
(repository, blog, product, etc.)
