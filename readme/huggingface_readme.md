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

# | [Overview](../README.MD) | [HuggingfaceHub](huggingface_readme.md) | [ABDADatasets](dataset_readme.md) | [ABSA Models](model_readme.md) | [Colab Tutorials](tutorial_readme.md) |

## Try our demos on Huggingface Space

- [Aspect-based sentiment classification (Multilingual)](https://huggingface.co/spaces/yangheng/PyABSA-APC) （English,
  Chinese, etc.）
- [Aspect term extraction & sentiment classification](https://huggingface.co/spaces/yangheng/PyABSA-ATEPC) (English,
  Chinese, Arabic, Dutch, French, Russian, Spanish, Turkish, etc.)
- [方面术语提取和情感分类](https://huggingface.co/spaces/yangheng/PyABSA-ATEPC-Chinese) （中文, etc.）

## Try our demos on Huggingface Space via API

- [Aspect-based sentiment classification (Multilingual)](https://huggingface.co/spaces/yangheng/PyABSA-APC)

```python3
import requests
r = requests.post(url='https://hf.space/embed/yangheng/PyABSA-APC/+/api/predict/',
                  json={"data": ["I have had my [ASP]computer[ASP] for 2 weeks already and it [ASP]works[ASP] perfectly . !sent!  Positive, Positive"]})
r.json()
```

- [Aspect term extraction & sentiment classification (Multilingual)](https://huggingface.co/spaces/yangheng/PyABSA-ATEPC)

```python3
import requests
r = requests.post(
    url='https://hf.space/embed/yangheng/PyABSA-ATEPC/+/api/predict/',
    json={"data": ['The wine list is incredible and extensive and diverse , '
                   'the food is all incredible and the staff was all very nice ,'
                   'good at their jobs and cultured .']})
r.json() 
```

- [方面术语提取和情感分类（中文）](https://huggingface.co/spaces/yangheng/PyABSA-ATEPC-Chinese)

```python3
import requests
r = requests.post(url='https://hf.space/embed/yangheng/PyABSA-ATEPC-Chinese/+/api/predict/',
                  json={"data": ["这款手机真的很薄，但是颜色不太好看，总体上我很满意啦。"]})
r.json()
```

# Develop & Research based on PyABSA

## Use Our Model via Transformers Model Hub

**If you do not need the best models of APC or ATEPC, you can easily try our pretrained model to save your time!**

To facilitate ABSA research and application, we train our fast-lcf-bert model based on
the [https://huggingface.co/yangheng/deberta-v3-base-absa-v1.1](https://huggingface.co/https://huggingface.co/yangheng/deberta-v3-base-absa-v1.1) with all the english datasets provided
by [ABSADatasets](https://github.com/yangheng95/ABSADatasets), the model is available
at [yangheng/deberta-v3-base-absa-v1.1](https://huggingface.co/yangheng/deberta-v3-base-absa-v1.1). You can use **
yangheng/deberta-v3-base-absa**
to **easily** improve your model if your model is based on the `transformers`. e.g.:

### Use Our Pretrained model to Classify Sentiments

The `yangheng/deberta-v3-base-absa-v1.1` and `yangheng/deberta-v3-large-absa-v1.1` are fine-tuned on the english
datasets (30k+ examples) from
[ABSADatasets](https://github.com/yangheng95/ABSADatasets), and have the output layer to be used in the
sentiment-analysis pipeline in huggingface hub.

```python3
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
# model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-large-absa-v1.1")

inputs = tokenizer("[CLS] when tables opened up, the manager sat another party before us. [SEP] manager [SEP]", return_tensors="pt")
outputs = model(**inputs)
```

### Use Our Pretrained model as a Backbone Model

The `yangheng/deberta-v3-base-absa` and `yangheng/deberta-v3-large-absa` are fine-tuned on the english datasets (
including the augmentation data, 180k+ examples) from
[ABSADatasets](https://github.com/yangheng95/ABSADatasets), and have no output layer. They are more effective when being
used as backbone model compared to `v1.1`

```python3
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa")
model = AutoModel.from_pretrained("yangheng/deberta-v3-base-absa")
# model = AutoModel.from_pretrained("yangheng/deberta-v3-large-absa")

inputs = tokenizer("good product especially video and audio quality fantastic.", return_tensors="pt")
outputs = model(**inputs)
```
