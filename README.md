# LCF-based Aspect Polarity Classification (基于局部上下文专注机制的方面级情感分类模型库)

> Training & Inferring & Reproducing SOTA models of ABSA

> Aspect-based Sentiment Analysis (GloVe / BERT).

> Chinese Aspect-based Sentiment Analysis (中文方面级情感分类)

> PyTorch Implementations.

We hope this repository will help you and sincerely request bug reports and Suggestions.
If you like this repository you can star or share this repository to others. 


Codes for our paper(s): 

- Yang H, Zeng B. [Enhancing Fine-grained Sentiment Classification Exploiting Local Context Embedding[J]](https://arxiv.org/abs/2010.00767). arXiv preprint arXiv:2010.00767, 2020.

- Yang H, Zeng B, Yang J, et al. [A multi-task learning model for Chinese-oriented aspect polarity classification and aspect term extraction[J]](https://www.sciencedirect.com/science/article/abs/pii/S0925231220312534). Neurocomputing, 419: 344-356.

- Zeng B, Yang H, Xu R, et al. [Lcf: A local context focus mechanism for aspect-based sentiment classification[J]](https://www.mdpi.com/2076-3417/9/16/3389).  Applied Sciences, 2019, 9(16): 3389.


## Requirement
* Python 3.7 + (recommended)
* PyTorch >= 1.0
* [transformers](https://github.com/huggingface/transformers) 
```pip install transformers or conda install transformers```
* Try to set ```batch_size=8``` or ```max_seq_len=40``` while out-of-memory error occurs.


## Before Training
* Download the [GloVe](https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors) if you want to use the GloVe-based models.
* Download the [domain-adapted BERT](https://github.com/deepopinion/domain-adapted-atsc) if you want to use state-of-the-art bert-based models.
* Set `use_bert_spc=True` to employ BERT-SPC input format and improve model performance.
* Set `use_dual_bert=True` to use dual BERTs for modeling local context and global context, respectively.
Bert-based models need more computational resources, e.g. system memory.

## Model Introduction 
This repository provides a variety of APC models, 
especially the those based on the local context focus mechanisms, including:
 
### Our LCF-based APC models

- **[LCA-LSTM](modules/models/lca_lstm.py)** 

- **[LCA-GloVe](modules/models/lca_glove.py)**

- **[LCA-BERT](modules/models/lca_bert.py)**

- **[LCF-GloVe](modules/models/lcf_glove.py)**

- **[LCF-BERT](modules/models/lcf_bert.py)**


### Other famous APC models

- **[LCFS-BERT](modules/models/lcf-bert.py)** 

Phan M H, Ogunbona P O. [Modelling context and syntactical features for aspect-based sentiment analysis[C]](https://www.aclweb.org/anthology/2020.acl-main.293/)//Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020: 3211-3220.

  The following models are forked from [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch).

- **[AEN-BERT](modules/models/aen.py)** | **[BERT-BASE](modules/models/bert_base.py)** | **[BERT-SPC](modules/models/bert_spc.py)**

Song Y, Wang J, Jiang T, et al. [Attentional encoder network for targeted sentiment classification](https://arxiv.org/pdf/1902.09314.pdf)[J]. arXiv preprint arXiv:1902.09314, 2019.

- **[MGAN](modules/models/mgan.py)** 

Fan F, Feng Y, Zhao D. [Multi-grained attention network for aspect-level sentiment classification](https://www.aclweb.org/anthology/D18-1380.pdf)[C]//Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 2018: 3433-3442.

- **[AOA](modules/models/aoa.py)** 

Huang B, Ou Y, Carley K M. [Aspect level sentiment classification with attention-over-attention neural networks](https://arxiv.org/pdf/1804.06536.pdf)[C]//International Conference on Social Computing, Behavioral-Cultural Modeling and Prediction and Behavior Representation in Modeling and Simulation. Springer, Cham, 2018: 197-206.

- **[TNet ](modules/models/tnet_lf.py)** 

Li X, Bing L, Lam W, et al. [ Transformation Networks for Target-Oriented Sentiment Classification](https://www.aclweb.org/anthology/P18-1087.pdf)[C]//Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018: 946-956.

- **[Cabasc](modules/models/cabasc.py)** 

Liu Q, Zhang H, Zeng Y, et al. [Content attention model for aspect based sentiment analysis](https://dl.acm.org/doi/abs/10.1145/3178876.3186001)[C]//Proceedings of the 2018 World Wide Web Conference. 2018: 1023-1032.

- **[RAM](modules/models/ram.py)** 

Chen P, Sun Z, Bing L, et al. [Recurrent attention network on memory for aspect sentiment analysis](https://www.aclweb.org/anthology/D17-1047.pdf)[C]//Proceedings of the 2017 conference on empirical methods in natural language processing. 2017: 452-461.

- **[MemNet](modules/models/memnet.py)** 

Tang D, Qin B, Liu T. [Aspect Level Sentiment Classification with Deep Memory Network](https://www.aclweb.org/anthology/D16-1021.pdf)[C]//Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing. 2016: 214-224.

- **[IAN](modules/models/ian.py)**
 
 Ma D, Li S, Zhang X, et al. [Interactive attention networks for aspect-level sentiment classification](https://dl.acm.org/doi/abs/10.5555/3171837.3171854)[C]//Proceedings of the 26th International Joint Conference on Artificial Intelligence. 2017: 4068-4074.

- **[ATAE-LSTM](modules/models/atae_lstm.py)** 

Wang Y, Huang M, Zhu X, et al. [Attention-based LSTM for aspect-level sentiment classification](https://www.aclweb.org/anthology/D16-1058.pdf)[C]//Proceedings of the 2016 conference on empirical methods in natural language processing. 2016: 606-615.

- **[TD-LSTM](modules/models/td_lstm.py)** 

Tang D, Qin B, Feng X, et al. [Effective LSTMs for Target-Dependent Sentiment Classification](https://www.aclweb.org/anthology/C16-1311.pdf)[C]//Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers. 2016: 3298-3307.

- **[LSTM](modules/models/lstm.py)** 



## Datasets

* ACL Twitter dataset
* Chinese Review Datasets 
* Multilingual dataset (combining of above datasets)
* SemEval-2014 
* SemEval-2015 (From [ASGAN](https://github.com/GeneZC/ASGCN)) 
* SemEval-2016 (From [ASGAN](https://github.com/GeneZC/ASGCN)) 


## Extra Hyperparameters

We list the valid parameters for each model for reference.

|    Models   | srd | lcf | LCA | lcp |sigma(σ)|   use_bert_spc  |
|:-----------:|:---:|:---:|:---:|:---:| :----: | :-------------: |
|  BERT-BASE  |  X  |  X  |  X  |  X  |    X   |        X        | 
|  BERT-SPC   |  X  |  X  |  X  |  X  |    X   |        X        | 
|  LCF-GloVe  |  √  |  √  |  X  |  X  |    X   |        X        |   
|  LCF-BERT   |  √  |  √  |  X  |  X  |    X   |        √        |  
|  LCA-LSTM   |  √  |  √  |  √  |  X  |    √   |        X        |  
|  LCA-Glove  |  √  |  √  |  √  |  √  |    √   |        X        | 
|  LCA-BERT   |  √  |  √  |  √  |  √  |    √   |        √        |  


Although datasets and models and be combined in most scenarios, some combinations are not recommended. Such Chinese dataset and BERT-base-uncased (English), Chinese and LCFS-BERT.
## Performance of BERT-based Models

The state-of-the-art benchmarks of the ABSA task can be found at [NLP-progress](https://nlpprogress.com) (See Section of SemEval-2014 subtask4)
"D", "S" and "A" denote dual-BERT, single-BERT and adapted-BERT, respectively. "N/A" means waiting to test.

|      Models          | Laptop14 (acc) |  Restaurant14 (acc) | Twitter(acc) | Memory Usage |
| :------------------: | :------------: | :-----------------: |:------------:|:------------:|
|  LCF-BERT-CDM (D+A)  |      82.92     |        89.11        |    77.89     |    < 8 G     | 
|  LCF-BERT-CDW (D+A)  |      82.76     |        89.38        |    77.17     |    < 8 G     |
|  LCF-BERT-CDM (S+A)  |      80.72     |        89.22        |    75.72     |    < 5.5 G   |
|  LCF-BERT-CDW (S+A)  |      80.88     |        88.57        |    75.58     |    < 5.5 G   |
|   LCF-BERT-CDM (S)   |      80.56     |        85.45        |    75.29     |    < 5.5 G   |
|   LCF-BERT-CDW (S)   |      80.25     |        85.54        |    76.59     |    < 5.5 G   |
|   LCA-BERT (S+A)     |      82.45     |        88.93        |    77.46     |    < 5.5 G   |
|    LCA-BERT (S)      |      81.66     |        86.07        |    76.59     |    < 5.5 G   |
|      AEN-BERT        |      79.93     |        83.12        |    74.71     |    < 6 G     |

We provides a training [log](training_logs/train.log.dat) of LCF-BERT based on [domain-adapted BERT](https://arxiv.org/pdf/1908.11860.pdf) to guide reproductions.
Try to set other random seeds to explore different results.
Learn to train the domain adapted BERT pretrained models from [domain-adapted-atsc](https://github.com/deepopinion/domain-adapted-atsc), and place the pre-trained models in bert_pretrained_models. 

## [Training](./training.py)

Training single model with cmd:

```
python train_apc.py
```
or running multiple experiments using config file 
```
python train_apc.py --config experiments_apc.json
```

## [Inferring](./batch_inferring/README.md)

Now, we release the universal batch inferring of aspect polarity for all listed APC models! 
Check [here](./batch_inferring/README.md) see the instructions of batch inferring

## Acknowledgement

This work is based on the repositories of [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) and the [pytorch-transformers](https://github.com/huggingface/transformers). Thanks to the authors for their devotion and Thanks to everyone who offered assistance.
Feel free to report any bug or discussing with us. 

## Contributions & Bug Reports.

This Repository is under development. There may be unknown problems in the code. We hope to get your help to make it easier to use and stable.

## Citation
If this repository is helpful to you, please cite our papers:

    @article{yang2021multi,
        title={A multi-task learning model for chinese-oriented aspect polarity classification and aspect term extraction},
        author={Yang, Heng and Zeng, Biqing and Yang, JianHao and Song, Youwei and Xu, Ruyang},
        journal={Neurocomputing},
        volume={419},
        pages={344--356},
        year={2021},
        publisher={Elsevier}
    }
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
    @misc{yang2020enhancing,
        title={Enhancing Fine-grained Sentiment Classification Exploiting Local Context Embedding}, 
        author={Heng Yang and Biqing Zeng},
        year={2020},
        eprint={2010.00767},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }

## Related Repositories


[ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch)

[domain-adapted-atsc](https://github.com/deepopinion/domain-adapted-atsc)

[LCFS-BERT](https://github.com/HieuPhan33/LCFS-BERT)
