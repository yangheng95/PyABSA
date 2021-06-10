# Aspect Term Extraction & Sentiment Classification 
# 方面术语抽取及方面情感分类工具
![PyPI - Python Version](https://img.shields.io/badge/python-3.6-blue.svg) 
[![PyPI](https://img.shields.io/pypi/v/pyabsa)](https://pypi.org/project/pyabsa/)
![Repo Size](https://img.shields.io/github/repo-size/yangheng95/pyabsa)
[![PyPI_downloads](https://img.shields.io/pypi/dm/pyabsa)](https://pypi.org/project/pyabsa/)
![License](https://img.shields.io/pypi/l/pyabsa?logo=PyABSA)
![welcome](https://img.shields.io/badge/Contribution-Welcome-brightgreen)
[![Gitter](https://badges.gitter.im/PyABSA/community.svg)](https://gitter.im/PyABSA/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

> Build from LC-ABSA / LCF-ABSA / LCF-BERT and LCF-ATEPC.

> Easy to use interfaces of aspect term extraction and aspect-polarity classification.

> Provide the tutorials of using ATE and APC interfaces.

> PyTorch Implementations (CPU & CUDA supported).
> 
# Introduction

This is an ASBA research-oriented code repository. I notice that some repos do not provide inference scripts and the codes may be redundant or hard to reproduce, so I build the framework to make the training and inference process easier. Except for providing SOTA model for both ATE and APC subtasks,  some functions and methods are reusable. In another word, you can develop your model using integrated functions, e.g., local context weight vector building. Please feel free to give me your interesting thoughts, to help me build an easy-to-use toolkit in order to reduce the cost of building models and reproduction.

# Notice

If you are looking for the original codes of the LCF-related papers, please go to
the [LC-ABSA](https://github.com/yangheng95/LC-ABSA/tree/master)
or [LCF-ATEPC](https://github.com/yangheng95/LCF-ATEPC).


# Preliminaries

Please star this repository in case of keeping find new examples and providing your advice.

Install the latest version using,

```
pip install pyabsa
```

please always update if there is a newer version. 
To use our (APC) models, you may need download `en_core_web_sm` by

```
python -m spacy download en_core_web_sm
```
# Model Support
We provide the pretrained ATEPC and APC models
on [Google Drive](https://drive.google.com/drive/folders/1yiMTucHKy2hAx945lgzhvb9QeHvJrStC?usp=sharing)
or [百度网盘（提取码：absa）](https://pan.baidu.com/s/1FSgaSP4ubGWy0BjBQdct5w):

## ATEPC
1. [LCF-ATEPC](pyabsa/module/atepc/models/lcf_atepc.py) 
2. [BERT-BASE](pyabsa/module/atepc/models/bert_base.py) 

## APC
1. [BERT-BASE](pyabsa/module/apc/models/bert_base.py)
2. [BERT-SPC](pyabsa/module/apc/models/bert_spc.py)
3. [LCF-BERT](pyabsa/module/apc/models/lcf_bert.py)
4. [LCFS-BERT](pyabsa/module/apc/models/lcf_bert.py)
5. [SLIDE-LCF-BERT](pyabsa/module/apc/models/slide_lcf_bert.py)
6. [SLIDE-LCFS-BERT](pyabsa/module/apc/models/slide_lcf_bert.py)
7. [LCA-Net](pyabsa/module/apc/models/lca_bert.py)

download them if necessary, note that most of the provided models are trained on the assembled train set without evaluation on test set. 


# Aspect Term Extraction (ATE)

## Aspect Extraction Output Format (方面术语抽取结果示例如下):

```
Sentence with predicted labels:
尤(O) 其(O) 是(O) 照(O) 的(O) 大(O) 尺(O) 寸(O) 照(O) 片(O) 时(O) 效(B-ASP) 果(I-ASP) 也(O) 是(O) 非(O) 常(O) 不(O) 错(O) 的(O)
{'aspect': '效 果', 'position': '11,12', 'sentiment': '1'}
Sentence with predicted labels:
照(O) 大(O) 尺(O) 寸(O) 的(O) 照(O) 片(O) 的(O) 时(O) 候(O) 手(O) 机(O) 反(B-ASP) 映(I-ASP) 速(I-ASP) 度(I-ASP) 太(O) 慢(O)
{'aspect': '反 映 速 度', 'position': '12,13,14,15', 'sentiment': '0'}
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
How(O) pretentious(O) and(O) inappropriate(O) for(O) MJ(O) Grill(O) to(O) claim(O) that(O) it(O) provides(O) power(O) lunch(B-ASP) and(O) dinners(B-ASP) !(O)
{'aspect': 'lunch', 'position': '14', 'sentiment': 'Negative'}
{'aspect': 'dinners', 'position': '16', 'sentiment': 'Negative'}
```

Check the detailed usages in [ATE examples](examples/aspect_term_extraction) directory.

## Quick Start

1. Convert APC datasets to ATEPC datasets

If you got apc datasets with the same format as provided 
   [apc datasets](examples/aspect_polarity_classification/apc_datasets),
you can convert them to atepc datasets:

```
from pyabsa import convert_apc_set_to_atepc
convert_apc_set_to_atepc_set(r'apc_usages/datasets/restaurant16')
```

2. Training for ATEPC

```
from pyabsa import train_atepc, get_atepc_param_dict_english

from pyabsa.dataset import restaurant15

save_path = 'state_dict'

atepc_param_dict_english = get_atepc_param_dict_english()
aspect_extractor = train_atepc(parameter_dict=atepc_param_dict_english,      # set param_dict=None to use default model
                               dataset_path=restaurant15,    # file or dir, dataset(s) will be automatically detected
                               model_path_to_save=save_path,   # set model_path_to_save=None to avoid save model
                               auto_evaluate=True,             # evaluate model while training_tutorials if test set is available
                               auto_device=True                # Auto choose CUDA or CPU
                               )
```

3. Extract aspect terms
```
from pyabsa import load_aspect_extractor

examples = ['But the staff was so nice to us .',
            'But the staff was so horrible to us .',
            r'Not only was the food outstanding , but the little ` perks \' were great .',
            'It took half an hour to get our check , which was perfect since we could sit , have drinks and talk !'
            ]
            
# Download the provided pre-training models from Google Drive
model_path = 'state_dict/lcf_atepc_cdw_rest14_without_spc'

aspect_extractor = load_aspect_extractor(trained_model_path=model_path,
                                         auto_device=True)

atepc_result = aspect_extractor.extract_aspect(examples=examples,   # list-support only, for now
                                               print_result=True,   # print the result
                                               pred_sentiment=True  # Predict the sentiment of extracted aspect terms
                                               )
```

4. Training on Multiple datasets
```
from pyabsa import train_apc, get_apc_param_dict_english

from pyabsa.dataset import semeval

# You can place multiple atepc_datasets file in one dir to easily train using some atepc_datasets

save_path = 'state_dict'
sent_classifier = train_apc(parameter_dict=get_apc_param_dict_english(),           # set param_dict=None to use default model
                            dataset_path=semeval,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training_tutorials if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )
```

# Aspect Polarity Classification (APC)

Check the detailed usages in [APC examples](examples/aspect_polarity_classification) directory.

## Aspect-Polarity Classification Output Format (方面级性分类输出示例如下):
```
love  selena gomez  !!!! she rock !!!!!!!!!!!!!!!! and she 's cool she 's my idol 
selena gomez --> Positive  Real: Positive (Correct)
thehils Heard great things about the  ipad  for speech/communication . Educational discounts are problem best bet . Maybe Thanksgiving ? 
ipad --> Neutral  Real: Neutral (Correct)
Jamie fox , Eddie Murphy , and  barack obama  because they all are exciting , cute , and inspirational to lots of people including me !!! 
barack obama --> Positive  Real: Neutral (Wrong)
```

## Quick Start

0. Run benchmark for our sota apc models!

```
from pyabsa.research.apc.apc_benchmark import run_benchmark_for_apc_models

run_benchmark_for_apc_models()

```

1. Train our models on your custom dataset or public datasets, 
   in any train function can use the dataset name to load the dataset from network:

```
from pyabsa import train_apc, get_atepc_param_dict_base

from pyabsa.dataset import laptop14

save_path = 'state_dict'

apc_param_dict_base = get_atepc_param_dict_base()

# datasets_path = '../apc_datasets/SemEval/laptop14'  # automatic detect all datasets files in this path
sent_classifier = train_apc(parameter_dict=apc_param_dict_base,  # set param_dict=None will use the apc_param_dict as well
                            dataset_path=laptop14,          # train set and test set will be automatically detected
                            model_path_to_save=save_path,        # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,                  # evaluate model while training_tutorials if test set is available
                            auto_device=True                     # automatic choose CUDA or CPU
                            )

```

2. Load the trained model:

Load a trained model will also load the hyper-parameters used in training.

```
from pyabsa import load_sentiment_classifier

# The trained_model_path should be a dir containing the state_dict and config file
state_dict_path = 'state_dict/slide_lcfs_bert_trained'
sent_classifier = load_sentiment_classifier(trained_model_path=state_dict_path)


# 如果有需要，使用以下方法自定义情感索引到情感标签的词典， 其中-999为必需的填充， e.g.,
sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive', -999: ''}
sent_classifier.set_sentiment_map(sentiment_map)
```

3. Sentiment Prediction on an inference set:

```
from pyabsa import load_sentiment_classifier

from pyabsa.dataset import semeval

# Assume the sent_classifier is loaded or obtained using train function

model_path = '../state_dict/slide_lcfs_bert_cdw'   # please always check update on Google Drive before using
sent_classifier = load_sentiment_classifier(trained_model_path=model_path,
                                            auto_device=True,  # Use CUDA if available
                                            sentiment_map=sentiment_map
                                            )

text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent ,' \
       ' the [ASP]decor[ASP] cool and understated . !sent! 1 1'
sent_classifier.infer(text, print_result=True)

# batch inferring_tutorials returns the results, save the result if necessary using save_result=True
results = sent_classifier.batch_infer(target_file=semeval,
                                      print_result=True,
                                      save_result=True,
                                      ignore_error=True,
                                      )

```

4. Convert datasets for inference

```
from pyabsa import generate_inferrence_set_for_apc

from pyabsa.dataset import apc_datasets

# This function coverts a ABSA dataset to inference set, try to convert every dataset found in the dir
# please do check the output file!
generate_inferrence_set_for_apc(apc_datasets)

```
# [Datasets](https://github.com/yangheng95/ABSADatasets)

1. Twitter 
2. Laptop14
3. Restaurant14
4. Restaurant15
5. Restaurant16
6. Phone
7. Car
8. Camera
9. Notebook
10. Multilingual (The sum of above datasets.)
   


# Acknowledgement

This work build from LC-ABSA/LCF-ABSA and LCF-ATEPC, and other impressive works such as PyTorch-ABSA and LCFS-BERT. Feel free to help us optimize code or add new features!

欢迎提出疑问、意见和建议，或者帮助完善仓库，谢谢！

# To Do
1. Add more bert-based models
2. Add more APIs
3. Optimize codes and add comments

# Calling for New Datasets and Models
We hope you can help us to improve this work, e.g., provide new dataset and model implementations.
the copyrights of contributed resources belong to the contributors, thanks for your help.

# Citation
If this repository is helpful, please cite our paper:

```
    @article{yang2021multi,
        title={A multi-task learning model for chinese-oriented aspect polarity classification and aspect term extraction},
        author={Yang, Heng and Zeng, Biqing and Yang, JianHao and Song, Youwei and Xu, Ruyang},
        journal={Neurocomputing},
        volume={419},
        pages={344--356},
        year={2021},
        publisher={Elsevier}
    }
```

```
    @article{zeng2019lcf,
        title={LCF: A Local Context Focus Mechanism for Aspect-Based Sentiment Classification},
        author={Zeng, Biqing and Yang, Heng and Xu, Ruyang and Zhou, Wu and Han, Xuli},
        journal={Applied Sciences},
        volume={9},
        number={16},
        pages={3389},
        year={2019},
        publisher={Multidisciplinary Digital Publishing Institute}
    }
```
    
```    
    @misc{yang2020enhancing,
        title={Enhancing Fine-grained Sentiment Classification Exploiting Local Context Embedding}, 
        author={Heng Yang and Biqing Zeng},
        year={2020},
        eprint={2010.00767},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
```