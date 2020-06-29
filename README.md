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
* Download the [domain-adapted BERT](https://github.com/deepopinion/domain-adapted-atsc) if you wang improve model performance.
* Set `use_bert_spc=True` to employ BERT-SPC input format and improve model performance.
* Set `use_dual_bert=True` to use independent BERT for local context and gloabl context
and improve performance with consuming more resources, e.g. system memory.


Codes for our paper(s): 
- [LCF: A Local Context Focus Mechanism for Aspect-Based Sentiment Classification](https://www.mdpi.com/2076-3417/9/16/3389). 
- Unreleased Paper(s)

## Requirement
* Python 3.7 (recommended)
* PyTorch >= 1.0
* [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) 
```pip install pytorch-transformers==1.2.0```
* To unleash the performance of LCF-BERT models, a GPU equipped with a large memory (>=11GB) is recommended. 
* Try to set ```batch_size=8``` or ```max_seq_len=40``` while out-of-memory error occurs.

## Model Introduction 
This repository provides a variety of ABSA models, 
especially the those based on the local context focus mechanisms, including:
 
### LC-based ABSA models
- [BERT-BASE](models/lc_absa/bert_base.py)
- [BERT-SPC](models/lc_absa/bert_spc.py)
- [LCF-GloVe](models/lc_absa/lcf_glove.py)
- [LCF-BERT](models/lc_absa/lcf_bert.py)  
- [LCE-LSTM](models/lc_absa/LCE_lstm.py) 
- [LCE-GloVe](models/lc_absa/LCE_glove.py) 
- [LCE-BERT](models/lc_absa/LCE_bert.py)
- [LCF-ATEPC](models/lc_absa/lcf_atepc.py)
- HLCF-GloVe (pending release)
- HLCF-BERT (pending release)
- Developing Models

### General ABSA models
  The following models are forked from [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch).
- [AEN-BERT](models/absa/aen.py)
- [MGAN](models/absa/mgan.py)
- [AOA](models/absa/aoa.py)  
- [TNet ](models/absa/tnet_lf.py) 
- [Cabasc](models/absa/cabasc.py) 
- [RAM](models/absa/ram.py)
- [MemNet](models/absa/memnet.py)
- [IAN](models/absa/ian.py) 
- [ATAE-LSTM](models/absa/atae_lstm.py)
- [TD-LSTM](models/absa/td_lstm.py)
- [LSTM](models/absa/lstm.py)


### Extra Parameters

We list the valid parameters for each model for reference.

|    Models   | srd | lcf | lce | lcp |sigma(σ)|   use_bert_spc  | hlcf |  
|:-----------:|:---:|:---:|:---:|:---:| :----: | :-------------: |:----:|
|  BERT-BASE  |  X  |  X  |  X  |  X  |    X   |        X        |   X  |
|  BERT-SPC   |  X  |  X  |  X  |  X  |    X   |        X        |   X  |
|  LCF-GloVe  |  √  |  √  |  X  |  X  |    X   |        X        |   X  |
|  LCF-BERT   |  √  |  √  |  X  |  X  |    X   |        √        |   X  |
|  LCE-LSTM   |  √  |  √  |  √  |  X  |    √   |        X        |   X  |
|  LCE-Glove  |  √  |  √  |  √  |  X  |    √   |        X        |   X  |
|  LCE-BERT   |  √  |  √  |  √  |  X  |    √   |        √        |   X  |
|  LCF-ATEPC  |  √  |  √  |  X  |  X  |    X   |   √ (for APC)   |   X  |
|  HLCF-GloVe |  √  |  √  |  X  |  X  |    X   |        X        |   √  |
|  HLCF-BERT  |  √  |  √  |  X  |  X  |    X   |        √        |   √  |


## Datasets

* SemEval-2014 task4(Restaurant and Laptop datasets) 
* ACL Twitter dataset
* Chinese Review Datasets (Temporarily Untested)

## Performance of BERT-based Models

The state-of-the-art benchmarks of the ABSA task can be found at [NLP-progress](https://nlpprogress.com) (See Section of SemEval-2014 subtask4)
"D", "S" and "A" denote dual-BERT, single-BERT and adapted-BERT, respectively. 

|      Models          | Restaurant (acc) |  Laptop (acc) | Twitter(acc) | Memory Usage |
| :------------------: | :--------------: | :-----------: |:------------:|:------------:|
|  LCF-BERT-CDM (D+A)  |      89.11       |     82.92     |    77.89     |      10 G    | 
|  LCF-BERT-CDW (D+A)  |      89.38       |     82.76     |    77.17     |      10 G    |
|  LCF-BERT-CDM (S+A)  |      89.22       |     80.72     |    75.72     |      6 G     |
|  LCF-BERT-CDW (S+A)  |      88.57       |     80.88     |    75.58     |      6 G     |
|   LCF-BERT-CDM (S)   |      -           |      -        |    -         |      6 G     |
|   LCF-BERT-CDW (S)   |      -           |     -         |    -         |      6 G     |
|   LCE-BERT (S+A)     |      88.93       |     82.45     |    77.46     |      6 G     |
|    LCE-BERT (S)      |      86.07       |     81.66     |    76.59     |      6 G     |
|      HLCF-BERT       |      N/A         |     N/A       |    N/A       |      N/A     |
|      AEN-BERT        |      83.12       |     79.93     |    74.71     |      6 G     |

We provides a training [log](logs/train.log.dat) of LCF-BERT based on [domain-adapted BERT](https://arxiv.org/pdf/1908.11860.pdf) to guide reproductions.
The results in the above table are the best of five training processes (random seed 0, 1, 2, 3, 4), Try to set other random seeds to explore different results.
Learn to train the domain adapted BERT pretrained models from [domain-adapted-atsc](https://github.com/deepopinion/domain-adapted-atsc), and place the pre-trained models in bert_pretrained_models. 

## Training

Train the model with cmd:

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
If this repository is helpful to you, please cite our paper:

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

## Related Repositories


[ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch)

[domain-adapted-atsc](https://github.com/deepopinion/domain-adapted-atsc)
