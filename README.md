# Aspect Term Extraction & Sentiment Classification 
# 方面术语抽取及方面情感分类工具
![PyPI - Python Version](https://img.shields.io/badge/python-3.6-blue.svg) 
[![PyPI](https://img.shields.io/pypi/v/pyabsa)](https://pypi.org/project/pyabsa/)
[![PyPI_downloads](https://img.shields.io/pypi/dm/pyabsa)](https://pypi.org/project/pyabsa/)
> Build from LC-ABSA/LCF-ABSA and LCF-ATEPC.

> Easy to use interfaces of aspect term extraction and aspect sentiment classification.

> Provide the tutorials of using ATE and APC interfaces.

> PyTorch Implementations (CPU & CUDA supported).

# Notice

if you are looking for the original codes of the LCF-related papers, please go to
the [LC-ABSA](https://github.com/yangheng95/LC-ABSA/tree/master)
or [LCF-ATEPC](https://github.com/yangheng95/LCF-ATEPC).

# Preliminaries

Please star this repository in case of keeping find new examples and providing your advice.

Install the latest version using `pip install pyabsa`, always update if there is a newer version. 
To use our (APC) models, you may need download `en_core_web_sm` by

```
python -m spacy download en_core_web_sm
```
# Model Support
We provide the pretrained ATEPC and APC models
on [Google Drive](https://drive.google.com/drive/folders/1yiMTucHKy2hAx945lgzhvb9QeHvJrStC?usp=sharing)
or [百度网盘（提取码：absa）](https://pan.baidu.com/s/1FSgaSP4ubGWy0BjBQdct5w):

## ATEPC
1. [LCF-ATEPC](pyabsa/atepc/models/lcf_atepc.py) 

## APC
1. [BERT-BASE](pyabsa/apc/models/bert_base.py)
2. [BERT-SPC](pyabsa/apc/models/bert_spc.py)
3. [LCF-BERT](pyabsa/apc/models/lcf_bert.py)
4. [LCFS-BERT](pyabsa/apc/models/lcf_bert.py)
5. [SLIDE-LCF-BERT](pyabsa/apc/models/slide_lcf_bert.py)
6. [SLIDE-LCFS-BERT](pyabsa/apc/models/slide_lcf_bert.py)
7. [LCA-Net](pyabsa/apc/models/lca_bert.py)

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
   [apc datasets](examples/aspect_polarity_classification/datasets),
you can convert them to atepc datasets:

```
from pyabsa import convert_apc_set_to_atepc
convert_apc_set_to_atepc(r'../apc_usages/datasets/restaurant16')
```

2. Training for ATEPC

```
from pyabsa import train_atepc

# see hyper-parameters in pyabsa/main/training_configs.py
param_dict = {'model_name': 'rlcf_atepc',   #  {lcf_atepc, rlcf_atepc}
              'batch_size': 16,
              'seed': 1,
              'device': 'cuda',
              'num_epoch': 5,
              'optimizer': "adamw",
              'learning_rate': 0.00002,
              'pretrained_bert_name': "bert-base-uncased",
              'use_dual_bert': False,
              'use_bert_spc': False,
              'max_seq_len': 80,
              'log_step': 30,
              'SRD': 3,
              'lcf': "cdw",
              'dropout': 0,
              'l2reg': 0.00001,
              'polarities_dim': 3
              }

# Mind that polarities_dim = 2 for Chinese datasets, and the 'train_atepc' function only evaluates in last few epochs

train_set_path = 'atepc_datasets/restaurant14'
save_path = '../atepc_usages/state_dict'
aspect_extractor = train_atepc(parameter_dict=param_dict,      # set param_dict=None to use default model
                               dataset_path=train_set_path,    # file or dir, dataset(s) will be automatically detected
                               model_path_to_save=save_path,   # set model_path_to_save=None to avoid save model
                               auto_evaluate=True,             # evaluate model while training if test set is available
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
from pyabsa import train_apc

# You can place multiple datasets file in one dir to easily train using some datasets

# for example, training on the SemEval datasets, you can organize the dir as follow

# ATEPC同样支持多数据集集成训练，但请不要将极性标签（种类，长度）不同的数据集融合训练！
# --datasets
# ----laptop14
# ----restaurant14
# ----restaurant15
# ----restaurant16

# or
# --datasets
# ----SemEval2014
# ------laptop14
# ------restaurant14
# ----SemEval2015
# ------restaurant15
# ----SemEval2016
# ------restaurant16

save_path = 'state_dict'
datasets_path = 'datasets/SemEval'  # file or dir are accepted
sent_classifier = train_apc(parameter_dict=None,           # set param_dict=None to use default model
                            dataset_path=datasets_path,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )
```

# Aspect Polarity Classification (APC)

Check the detailed usages in [APC examples](examples/aspect_polarity_classification) directory.

## Quick Start

0. Instant train and infer on the provided datasets:

```
from pyabsa import train, train_and_evaluate, load_sentiment_classifier
dataset_path = 'datasets/laptop14'
sent_classifier = train_and_evaluate(parameter_dict=None,
                                     dataset_path=dataset_path,
                                     model_path_to_save=None
                                     )
text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent ,' \
       ' the [ASP]decor[ASP] cool and understated . !sent! 1 1'
sent_classifier.infer(text)
```

1. Train our models on your custom dataset:

```
from pyabsa import train, train_and_evaluate, load_sentiment_classifier
# see hyper-parameters in pyabsa/main/training_configs.py
param_dict = {'model_name': 'bert_base', 'batch_size': 16, 'device': 'cuda', 'num_epoch': 6}
# train_set_path = 'datasets/restaurant15'
train_set_path = 'example_files/sum_train.dat'  # replace the path of your custom dataset(s) here
model_path_to_save = 'state_dict'

sent_classifier = train_apc(parameter_dict=param_dict,    # set param_dict=None to use default model
                            dataset_path=train_set_path,  # file or dir, datasets will be automatically detected
                            model_path_to_save=model_path_to_save,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=False,          # evaluate model while training if test set is available
                            auto_device=True              # Auto choose CUDA or CPU
                            )

# Or, if you have the test set, this function also could evaluate model while training
datasets_path = 'datasets/restaurant15'                    # Refer to the path where the the train and test sets is placed
sent_classifier = train_apc(parameter_dict=param_dict,     # set param_dict=None to use default model
                            dataset_path=train_set_path,   # file or dir, dataset(s) will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=False,           # evaluate model while training if test set is available
                            auto_device=True               # Auto choose CUDA or CPU
                            )
```

2. Load the trained model:

Load a trained model will also load the hyper-parameters used in training.

```
from pyabsa import load_sentiment_classifier

# The trained_model_path should be a dir containing the state_dict and config file
state_dict_path = 'state_dict/slide_lcfs_bert_trained'
sent_classifier = load_sentiment_classifier(trained_model_path=state_dict_path)
```

3. Sentiment Prediction on an inference set:

```
# Infer a formatted text, the reference sentiment begins with !sent! is optional

text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent , the [ASP]decor[ASP] cool and understated . !sent! 1 1'
# or text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent , the [ASP]decor[ASP] cool and understated .'

sent_classifier = load_sentiment_classifier(trained_model_path='../state_dict/slide_lcfs_bert_trained')

# The default loading device is CPU, you can alter the loading device

# load the model to CPU
# sent_classifier.cpu()

# load the model to CUDA (0)
# sent_classifier.cuda()

# load the model to CPU or CUDA, like cpu, cuda:0, cuda:1, etc.
sent_classifier.to('cuda:0')

sent_classifier.infer(text)

# batch inference from on a inference dataset
test_set_path = 'example_files/rest16_test_inferring.dat' 
results = sent_classifier.batch_infer(test_set_path, print_result=True, save_result=True)
```

4. Convert datasets for inference

```
from pyabsa import generate_inferring_set_for_apc

# This function coverts a ABSA dataset to inference set, try to convert every dataset found in the dir
generate_inferring_set_for_apc('datasets/restaurant14')
```


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