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


## Public and Community-shared ABSADatasets

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
15. Yelp

You don't have to download the datasets, as the datasets will be downloaded automatically.

## Annotate Your Own Dataset

The repo [ABSADatasets/DPT](https://github.com/yangheng95/ABSADatasets/tree/v1.2/DPT) provides an open-source dataset
annotating tool, you can easily annotate your dataset before using PyABSA.

### Important: Rename your dataset filename before use it in PyABSA

Although the integrated datasets have no ids, it is recommended to assign an id for your dataset. 
While merge your datasets into ABSADatasets, please keep the id remained. 

- APC dataset name should be {id}.{dataset name}, and the dataset files should be named in {dataset name}.{type}.dat.atepc e.g., 
```tree
datasets
├── 101.restaurant
│    ├── restaurant.train.dat  # train_dataset
│    ├── restaurant.test.dat  # test_dataset
│    └── restaurant.valid.dat  # valid_dataset, dev set are not recognized in PyASBA, please rename dev-set to valid-set
└── others
```

- ATEPC dataset files should be {id}.{dataset name}.{type}.dat.atepc,
e.g., 
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
