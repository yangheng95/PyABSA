ABSA datasets for [PyABSA](https://github.com/yangheng95/PyABSA)
--------------------------------------------------------------

[![total views](https://raw.githubusercontent.com/yangheng95/ABSADatasets/traffic/total_views.svg)](https://github.com/yangheng95/ABSADatasets/tree/traffic#-total-traffic-data-badge)
[![total views per week](https://raw.githubusercontent.com/yangheng95/ABSADatasets/traffic/total_views_per_week.svg)](https://github.com/yangheng95/ABSADatasets/tree/traffic#-total-traffic-data-badge)
[![total clones](https://raw.githubusercontent.com/yangheng95/ABSADatasets/traffic/total_clones.svg)](https://github.com/yangheng95/ABSADatasets/tree/traffic#-total-traffic-data-badge)
[![total clones per week](https://raw.githubusercontent.com/yangheng95/ABSADatasets/traffic/total_clones_per_week.svg)](https://github.com/yangheng95/ABSADatasets/tree/traffic#-total-traffic-data-badge)

## To augment your datasets, please refer to [BoostTextAugmentation](https://github.com/yangheng95/BoostTextAugmentation)

## Auto-annoate your datasets via PyABSA!

There is an experimental feature which allows you to auto-build APC dataset and ATEPC datasets, see the usage here:

```python3 
from pyabsa import make_ABSA_dataset 
# refer to the comments in this function for detailed usage
make_ABSA_dataset(dataset_name_or_path='integrated_datasets/review', checkpoint='english')
```

## Contribute (prepare) your dataset to ABSADatasets to use it in PyABSA

We hope you can share your custom dataset or an available public dataset. If you are willing to, please follow the
instruction to process your data and open a PR.

### Format your dataset to use it in PyABSA

- Format your APC dataset according to our dataset format. (**Recommended. Once you finished this step, we can help you
  to finish other steps**)
  ![image](https://user-images.githubusercontent.com/51735130/136219441-c3e9b4e2-6e31-4d32-b6c3-a66788b179f6.png)

- Generate the inference dataset for APC / ATEPC task (**Optional**. The example is
  available [here](https://github.com/yangheng95/PyABSA/blob/release/demos/aspect_polarity_classification/generate_inference_set.py))
- Convert the APC dataset to ATEPC dataset, and move the transformed ATEPC datasets from apc_dataset to corresponding
  atepc_datasets. (**Optional**. The example is
  available [here](https://github.com/yangheng95/PyABSA/blob/release/demos/aspect_term_extraction/convert_apc_set_to_atepc_set.py) )
- Register your dataset in PyABSA. (**Optional**.
  Register [here](https://github.com/yangheng95/PyABSA/blob/302da1e4b2292cdbc5b9c712862e623c427132b8/pyabsa/functional/dataset/dataset_manager.py#L37))

### Important: Rename your dataset filename before use it in PyABSA

It is recommended to assign an id for your dataset, which will avoid some potential problem (e.g., dataset mis-loading)
while PyABSA detects the dataset.
Merging your datasets into ABSADatasets, please keep the id remained.

- For a custom APC dataset, its name should be **{id}.{dataset name}.{type}.dat.apc**,

```tree
datasets
├── apc_datasets
│     ├── 101.restaurant
│     │    ├── restaurant.train.dat.apc  # train_dataset
│     │    ├── restaurant.test.dat.apc  # test_dataset
│     │    └── restaurant.valid.dat.apc  # valid_dataset, dev set are not recognized in PyASBA, please rename dev-set to valid-set
│     └── others
├── atepc_datasets
```

- ATEPC dataset files should be **{id}.{dataset name}.{type}.dat.atepc**,
  e.g.,

```tree
datasets
├── 101.restaurant
│    ├── restaurant.train.dat.atepc  # train_dataset
│    ├── restaurant.test.dat.atepc  # test_dataset
│    └── restaurant.valid.dat.atepc  # valid_dataset, dev set are not recognized in PyASBA, please rename dev-set to valid-set
└── others
```

I prepare a demo custom APC/ATEPC dataset which is based on third-party annotated Yelp dataset. Iif you got problem in
dataset renaming, please put your data into the prepared dataset files.
Check [datasets/apc_datasets/100.CustomDataset](https://github.com/yangheng95/ABSADatasets/tree/v2.0/datasets/apc_datasets/100.CustomDataset)
and [datasets/atepc_datasets/100.CustomDataset](https://github.com/yangheng95/ABSADatasets/tree/v2.0/datasets/atepc_datasets/100.CustomDataset) to view or rewrite the custom
dataset.

Then, use the {id}.{dataset name} to locate your dataset, e.g., `dataset = '101.restaurant'`


### Need to Annotate Your Own Dataset?

- A Stand-alone browser based tool to help process data for the training
  set. [here](https://github.com/yangheng95/ABSADatasets/tree/v1.2/DPT)
  ![image1](https://user-images.githubusercontent.com/4684417/139701633-d77a009b-1a12-4ef2-9663-37d2d36e1af1.JPG)

  Once data saved, 3 files will be created:

    1. a CSV file training set for classic sentiment analysis
    2. a TXT file training set for PyABSA
    3. a JSON file for saving unfinished work

## Notice

All datasets provided are for research only, we do not hold any Copyright of any datasets. These datasets follow their
original licenses (if any).

## Datasets source:

MAMS https://github.com/siat-nlp/MAMS-for-ABSA

SemEval 2014: https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools

SemEval 2015: https://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools

SemEval 2016: https://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools

Chinese: https://www.sciencedirect.com/science/article/abs/pii/S0950705118300972?via%3Dihub

Shampoo: [brightgems@GitHub](https://github.com/brightgems/ABSADatasets)

MOOC: [jmc-123@GitHub](https://github.com/jmc-123/ABSADatasets) with GPL License

Twitter: https://dl.acm.org/doi/10.5555/2832415.2832437

Television & TShirt: https://github.com/rajdeep345/ABSA-Reproducibility

Yelp: [WeiLi9811@GitHub](https://github.com/WeiLi9811)

SemEval2016Task5: [YaxinCui@GitHub](https://github.com/YaxinCui/ABSADataset)

- Arabic Hotel Reviews
- Dutch Restaurant Reviews
- English Restaurant Reviews
- French Restaurant Reviews
- Russian Restaurant Reviews
- Spanish Restaurant Reviews
- Turkish Restaurant Reviews

English-MOOC github: [aparnavalli@GitHub](https://github.com/aparnavalli)
