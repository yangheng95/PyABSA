# Local Context-based ABSA


# [APC](README.md) | [ATEPC](README_atepc.md)

> PyTorch Implementations.

> PyTorch-transformers.

> Aspect-based Sentiment Analysis (GloVe/BERT).

>Chinese Aspect-based Sentiment Analysis
>> 中文方面级情感分析（中文ABSA）

We hope this repository will help you and sincerely request bug reports and Suggestions.
If you like this repository you can star or share this repository to others. 

### Tips
* Download the [GloVe](https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors) before use the GloVe-based models.
* Download the [domain-adapted BERT](https://github.com/deepopinion/domain-adapted-atsc) if you want to improve model performance.
* Set `use_bert_spc=True` to employ BERT-SPC input format and improve model performance.
* Set `use_dual_bert=True` to use independent BERT for local context and global context
and improve performance with consuming more resources, e.g. system memory.


Codes for our paper(s): 
- [LCF: A Local Context Focus Mechanism for Aspect-Based Sentiment Classification](https://www.mdpi.com/2076-3417/9/16/3389). 

- [Enhancing Fine-grained Sentiment Classification Exploiting Local Context Embedding](https://arxiv.org/abs/2010.00767).

## Requirement
* Python 3.7 + (recommended)
* PyTorch >= 1.0
* [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) 
```pip install pytorch-transformers==1.2.0```
* Try to set ```batch_size=8``` or ```max_seq_len=40``` while out-of-memory error occurs.

## Model Introduction 
This repository provides a variety of ABSA models, 
especially the those based on the local context focus mechanisms, including:
 
### LC-based ABSA models
- **[BERT-BASE](models/lc_apc/bert_base.py)** 

- **[BERT-SPC](models/lc_apc/bert_spc.py)**

- **[LCF-GloVe](models/lc_apc/lcf_glove.py)**

- **[LCF-BERT](models/lc_apc/lcf_bert.py)** (Set 'lcfs = True' to use [LCFS-BERT](https://www.aclweb.org/anthology/2020.acl-main.293))

- **[LCA-LSTM](models/lc_apc/lca_lstm.py)** 

- **[LCA-GloVe](models/lc_apc/lca_glove.py)**

- **[LCA-BERT](models/lc_apc/lca_bert.py)**

- **[LCF-ATEPC](models/lc_atepc/lcf_atepc.py)**

- HLCF-GloVe (pending release)
- HLCF-BERT (pending release)
- Developing Models

### General ABSA models
  The following models are forked from [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch).

- **[AEN-BERT](models/apc/aen.py)** 

Song Y, Wang J, Jiang T, et al. [Attentional encoder network for targeted sentiment classification](https://arxiv.org/pdf/1902.09314.pdf)[J]. arXiv preprint arXiv:1902.09314, 2019.

- **[MGAN](models/apc/mgan.py)** 

Fan F, Feng Y, Zhao D. [Multi-grained attention network for aspect-level sentiment classification](https://www.aclweb.org/anthology/D18-1380.pdf)[C]//Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 2018: 3433-3442.

- **[AOA](models/apc/aoa.py)** 

Huang B, Ou Y, Carley K M. [Aspect level sentiment classification with attention-over-attention neural networks](https://arxiv.org/pdf/1804.06536.pdf)[C]//International Conference on Social Computing, Behavioral-Cultural Modeling and Prediction and Behavior Representation in Modeling and Simulation. Springer, Cham, 2018: 197-206.

- **[TNet ](models/apc/tnet_lf.py)** 

Li X, Bing L, Lam W, et al. [ Transformation Networks for Target-Oriented Sentiment Classification](https://www.aclweb.org/anthology/P18-1087.pdf)[C]//Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018: 946-956.

- **[Cabasc](models/apc/cabasc.py)** 

Liu Q, Zhang H, Zeng Y, et al. [Content attention model for aspect based sentiment analysis](https://dl.acm.org/doi/abs/10.1145/3178876.3186001)[C]//Proceedings of the 2018 World Wide Web Conference. 2018: 1023-1032.

- **[RAM](models/apc/ram.py)** 

Chen P, Sun Z, Bing L, et al. [Recurrent attention network on memory for aspect sentiment analysis](https://www.aclweb.org/anthology/D17-1047.pdf)[C]//Proceedings of the 2017 conference on empirical methods in natural language processing. 2017: 452-461.

- **[MemNet](models/apc/memnet.py)** 

Tang D, Qin B, Liu T. [Aspect Level Sentiment Classification with Deep Memory Network](https://www.aclweb.org/anthology/D16-1021.pdf)[C]//Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing. 2016: 214-224.

- **[IAN](models/apc/ian.py)**
 
 Ma D, Li S, Zhang X, et al. [Interactive attention networks for aspect-level sentiment classification](https://dl.acm.org/doi/abs/10.5555/3171837.3171854)[C]//Proceedings of the 26th International Joint Conference on Artificial Intelligence. 2017: 4068-4074.

- **[ATAE-LSTM](models/apc/atae_lstm.py)** 

Wang Y, Huang M, Zhu X, et al. [Attention-based LSTM for aspect-level sentiment classification](https://www.aclweb.org/anthology/D16-1058.pdf)[C]//Proceedings of the 2016 conference on empirical methods in natural language processing. 2016: 606-615.

- **[TD-LSTM](models/apc/td_lstm.py)** 

Tang D, Qin B, Feng X, et al. [Effective LSTMs for Target-Dependent Sentiment Classification](https://www.aclweb.org/anthology/C16-1311.pdf)[C]//Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers. 2016: 3298-3307.

- **[LSTM](models/apc/lstm.py)** 


### Extra Parameters

We list the valid parameters for each model for reference.

|    Models   | srd | lcf | LCA | lcp |sigma(σ)|   use_bert_spc  | hlcf |  
|:-----------:|:---:|:---:|:---:|:---:| :----: | :-------------: |:----:|
|  BERT-BASE  |  X  |  X  |  X  |  X  |    X   |        X        |   X  |
|  BERT-SPC   |  X  |  X  |  X  |  X  |    X   |        X        |   X  |
|  LCF-GloVe  |  √  |  √  |  X  |  X  |    X   |        X        |   X  |
|  LCF-BERT   |  √  |  √  |  X  |  X  |    X   |        √        |   X  |
|  LCA-LSTM   |  √  |  √  |  √  |  X  |    √   |        X        |   X  |
|  LCA-Glove  |  √  |  √  |  √  |  √  |    √   |        X        |   X  |
|  LCA-BERT   |  √  |  √  |  √  |  √  |    √   |        √        |   X  |
|  LCF-ATEPC  |  √  |  √  |  X  |  X  |    X   |   √ (for APC)   |   X  |
|  HLCF-GloVe |  √  |  √  |  X  |  X  |    X   |        X        |   √  |
|  HLCF-BERT  |  √  |  √  |  X  |  X  |    X   |        √        |   √  |


## Datasets

* SemEval-2014 task4(Restaurant and Laptop datasets) 
* ACL Twitter dataset
* Chinese Review Datasets (Temporarily Untested)

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
|      HLCF-BERT       |       N/A      |         N/A         |    N/A       |    N/A       |
|      AEN-BERT        |      79.93     |        83.12        |    74.71     |    < 6 G     |

We provides a training [log](logs/train.log.dat) of LCF-BERT based on [domain-adapted BERT](https://arxiv.org/pdf/1908.11860.pdf) to guide reproductions.
Try to set other random seeds to explore different results.
Learn to train the domain adapted BERT pretrained models from [domain-adapted-atsc](https://github.com/deepopinion/domain-adapted-atsc), and place the pre-trained models in bert_pretrained_models. 

## Training

Train the model with cmd:

```
python train_apc.py
```
or 
```
python train_apc.py --config experiments_apc.json
```

## Inferring

Now, we release the universal batch inferring of aspect polarity for all listed ABSA models! 
(Although it is not automatic enough and unstable but may help you)
Put the saved state-dicts, inferring model-config, and the inferring dataset in the [infer_dataset](inferring_dataset) directory.
Then, run the batch inferring script:
```
python apc_infer.py --infer_dataset infer_dataset
```

## Acknowledgement

This work is based on the repositories of [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) and the [pytorch-transformers](https://github.com/huggingface/transformers). Thanks to the authors for their devotion and Thanks to everyone who offered assistance.
Feel free to report any bug or discussing with us. 

## Contributions & Bug reports are welcomed.

This Repository is under development. There may be unknown problems in the code. We hope to get your help to make it easier to use and stable.

## Citation
If this repository is helpful to you, please cite our papers:

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
