# LCF-based Aspect Polarity Classification (基于局部上下文专注机制的方面级情感分类模型库)

> Training & Inferring & Reproducing SOTA models of ABSA

> Aspect-based Sentiment Analysis (BERT).

> Chinese Aspect-based Sentiment Analysis (中文方面级情感分类)

> PyTorch Implementations.
> 

## Want a packaged aspect polarity infer tool using BERT?

If you only want to train on your on dataset or predict aspect sentiments, and never mind of the implementation of the models, you can use the packaged tool [**pyabsa**](https://github.com/yangheng95/pyabsa) which is more convenient.

如果你只需要训练和预测方面情感，不关注模型的实现方式，可以使用提供的开箱即用的方面级情感分类工具 [**pyabsa**](https://github.com/yangheng95/pyabsa)


Codes for our paper(s): 

- Yang H, Zeng B. [Enhancing Fine-grained Sentiment Classification Exploiting Local Context Embedding[J]](https://arxiv.org/abs/2010.00767). arXiv preprint arXiv:2010.00767, 2020.

- Yang H, Zeng B, Yang J, et al. [A multi-task learning model for Chinese-oriented aspect polarity classification and aspect term extraction[J]](https://www.sciencedirect.com/science/article/abs/pii/S0925231220312534). Neurocomputing, 419: 344-356.

- Zeng B, Yang H, Xu R, et al. [Lcf: A local context focus mechanism for aspect-based sentiment classification[J]](https://www.mdpi.com/2076-3417/9/16/3389).  Applied Sciences, 2019, 9(16): 3389.


## Requirement
* Python 3.7 + (recommended)
* PyTorch >= 1.0
* [transformers](https://github.com/huggingface/transformers) 
```pip install transformers==4.4.2 or conda install transformers==4.4.2```
* Try to set ```batch_size=8``` or ```max_seq_len=40``` while out-of-memory error occurs.


## Before Training
* Download the [domain-adapted BERT](https://github.com/deepopinion/domain-adapted-atsc) if you want to use state-of-the-art bert-based models.
* Set `use_bert_spc=True` to employ BERT-SPC input format and improve model performance.
* Set `use_dual_bert=True` to use dual BERTs for modeling local context and global context, respectively.
Bert-based models need more computational resources, e.g. system memory.

## Model Introduction 
This repository provides a variety of APC models, 
especially the those based on the local context focus mechanisms, including:
 
### Our LCF-based APC models

Try our best models `SLIDE-LCFS-BERT` and `SLIDE-LCF-BERT`

- **[SLIDE-LCF-BERT](modules/models/slide_lcf_bert.py)** 
  
- **[SLIDE-LCFS-BERT](modules/models/slide_lcf_bert.py)** 

- **[LCA-BERT](modules/models/lca_bert.py)**

- **[LCF-BERT](modules/models/lcf_bert.py)**

Note that GloVe-based models have been removed.

### Other famous APC models

- **[LCFS-BERT](modules/models/lcf-bert.py)** 

Phan M H, Ogunbona P O. [Modelling context and syntactical features for aspect-based sentiment analysis[C]](https://www.aclweb.org/anthology/2020.acl-main.293/)//Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020: 3211-3220.

  The following models are forked from [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch).

-  **[BERT-BASE](modules/models/bert_base.py)** 
-  **[BERT-SPC](modules/models/bert_spc.py)**


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
|  LCF-BERT   |  √  |  √  |  X  |  X  |    X   |        √        |  
|  LCA-BERT   |  √  |  √  |  √  |  √  |    √   |        √        |  
|  LCFS-BERT  |  √  |  √  |  X  |  X  |    X   |        √        |  
|  SLIDE-LCF-BERT   |  √  |  √  |  X  |  X  |    X   |        √        | 
|  SLIDE-LCFS-BERT   |  √  |  √  |  X  |  X  |    X   |        √        |  
|  AEN-BERT   |  X  |  X  |  X  |  X  |    X   |        √        |  

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

## Training

Training single model with cmd:

```
python train.py
```
or running multiple experiments using config file 
```
python train.py --config experiments_apc.json
```

## [Inferring](./batch_inferring/README.md)

We release the universal batch inferring of aspect polarity for all listed APC models! 
Check [here](./batch_inferring/README.md) and follow the instructions to do batch inferring.

## Acknowledgement

This work is based on the repositories of [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) and the [pytorch-transformers](https://github.com/huggingface/transformers). Thanks to the authors for their devotion and Thanks to everyone who offered assistance.
Feel free to report any bug or discussing with us. 

## Contributions & Bug Reports.

This Repository is under development. There may be unknown problems in the code. Please do feel free to report any problem, and PRs are welcome.
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
