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

# | [Overview](../README.MD) | [HuggingfaceHub](huggingface_readme.md) | [Colab Tutorials](tutorial_readme.md) |

## To augment your datasets, please refer to [BoostTextAugmentation](https://github.com/yangheng95/BoostTextAugmentation)

## Auto-annotate your datasets via PyABSA!

There is an experimental feature which allows you to auto-build APC dataset and ATEPC datasets, see the usage here:

```python3 
from pyabsa import make_ABSA_dataset 

# refer to the comments in this function for detailed usage
make_ABSA_dataset(dataset_name_or_path='review', checkpoint='english')
```

## Public and Community-shared ABSADatasets

More datasets are available at [ABSADatasets](https://github.com/yangheng95/ABSADatasets).

- MAMS https://github.com/siat-nlp/MAMS-for-ABSA
- SemEval 2014: https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools
- SemEval 2015: https://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools
- Chinese: https://www.sciencedirect.com/science/article/abs/pii/S0950705118300972?via%3Dihub
- Shampoo: [brightgems@GitHub](https://github.com/brightgems/ABSADatasets)
- MOOC: [jmc-123@GitHub](https://github.com/jmc-123/ABSADatasets) with GPL License
- Twitter: https://dl.acm.org/doi/10.5555/2832415.2832437
- Television & TShirt: https://github.com/rajdeep345/ABSA-Reproducibility
- Yelp: [WeiLi9811@GitHub](https://github.com/WeiLi9811)
- SemEval2016Task5: [YaxinCui@GitHub](https://github.com/YaxinCui/ABSADataset)
    - Arabic Hotel Reviews
    - Dutch Restaurant Reviews
    - English Restaurant Reviews
    - French Restaurant Reviews
    - Russian Restaurant Reviews
    - Spanish Restaurant Reviews
    - Turkish Restaurant Reviews
      You don't have to download the datasets, as the datasets will be downloaded automatically.

## Annotate Your Own Dataset

The repo [ABSADatasets/DPT](https://github.com/yangheng95/ABSADatasets/tree/v1.2/DPT) provides an open-source dataset
annotating tool, you can easily annotate your dataset before using PyABSA.

### Important: Rename your dataset filename before use it in PyABSA

Although the integrated datasets have no ids, it is recommended to assign an id for your dataset. While merge your
datasets into ABSADatasets, please keep the id remained.

- APC dataset name should be {id}.{dataset name}, and the dataset files should be named in {dataset
  name}.{type}.dat.atepc e.g.,

```tree
datasets
├── 101.restaurant
│    ├── restaurant.train.dat  # train_dataset
│    ├── restaurant.test.dat  # test_dataset
│    └── restaurant.valid.dat  # valid_dataset, dev set are not recognized in PyASBA, please rename dev-set to valid-set
└── others
```

- ATEPC dataset files should be {id}.{dataset name}.{type}.dat.atepc, e.g.,

```tree
datasets
├── 101.restaurant
│    ├── restaurant.train.dat.atepc  # train_dataset
│    ├── restaurant.test.dat.atepc  # test_dataset
│    └── restaurant.valid.dat.atepc  # valid_dataset, dev set are not recognized in PyASBA, please rename dev-set to valid-set
└── others
```

## Fit on Your Existing Dataset

- First, refer to [ABSADatasets](https://github.com/yangheng95/ABSADatasets) to prepare your dataset into acceptable
  format.
- You can PR to contribute your dataset and use it like `ABDADatasets.your_dataset` (All the datasets are for research
  only, shall not danger your data copyright)

## Use Human-readable Labels in Your Dataset

PyABSA encourages you to use string labels instead of numbers. e.g., sentiment labels = {negative, positive, Neutral,
unknown}

- What labels you use in the dataset, what labels will be output in inference
- You can train a model using multiple datasets with same sentiment labels, and you can even contribute and define a
  combination of datasets [here](../pyabsa/functional/dataset/dataset_manager.py)!
- The version information of PyABSA is also available in the output while loading checkpoints training args.
