# LCF-ATEPC

codes for our
paper [A Multi-task Learning Model for Chinese-oriented Aspect Polarity Classification and Aspect Term Extraction](https://arxiv.org/abs/1912.07976)

> LCF-ATEPC，面向中文及多语言的ATE和APC联合学习模型，基于PyTorch和pytorch-transformers.

> LCF-ATEPC, a multi-task learning model for Chinese and multilingual-oriented ATE and APC task, based on PyTorch

![LICENSE](https://img.shields.io/packagist/l/doctrine/orm.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-multi-task-learning-model-for-chinese/aspect-based-sentiment-analysis-on-semeval)](https://paperswithcode.com/sota/aspect-based-sentiment-analysis-on-semeval?p=a-multi-task-learning-model-for-chinese)

## Requirement

* Python >= 3.7
* PyTorch >= 1.0
* [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) = 1.2.0 (Package ``transformers`` is not
  recommended for our code)
* Set `use_bert_spc = True` to improve the APC performance while only APC is evaluated.

## Training

We use the configuration file to manage experiments setting.

Training in batches by experiments configuration file, refer to the [experiments.json](training/experiments.json) to
manage experiments.

Then,

```sh
python train.py --config_path experiments.json
```

## About dataset

If you want to build your dataset, please find the description of the
dataset [here](https://github.com/yangheng95/LCF-ATEPC/issues/25)

## Out of Memory

Since BERT models require a lot of memory. If the out-of-memory problem while training the model, here are the ways to
mitigate the problem:

1. Reduce the training batch size ( train_batch_size = 4 or 8 )
2. Reduce the longest input sequence ( max_seq_length = 40 or 60 )
3. Set `use_unique_bert = true` to use a unique BERT layer to model for both local and global contexts

## Model Performance

We made our efforts to make our benchmarks reproducible. However, the performance of the LCF-ATEPC models fluctuates and
any slight changes in the model structure could also influence performance. Try different random seed to achieve optimal
results.

### Performance on Chinese Datasets

![chinese](assets/Chinese-results.png)

### Performance on Multilingual Datasets

![multilingual](assets/multilingual-results.png)

### Optimal Performance on Laptop and Restaurant Datasets

![semeval2014](assets/SemEval-2014-results.png)

## Model Architecture

![lcf](assets/lcf-atepc.png)

## Notice

We cleaned up and refactored the original codes for easy understanding and reproduction. However, we didn't test all the
training situations for the refactored codes. If you find any issue in this repo, You can raise an issue or submit a
pull request, whichever is more convenient for you.

Due to the busy schedule, some module may not update for long term, such as saving and loading module for trained
models, inferring module, etc. If possible, we sincerely request for someone to accomplish these work.

## Citation

If this repository is helpful to you, please cite our paper:

    @misc{yang2019multitask,
        title={A Multi-task Learning Model for Chinese-oriented Aspect Polarity Classification and Aspect Term Extraction},
        author={Heng Yang and Biqing Zeng and JianHao Yang and Youwei Song and Ruyang Xu},
        year={2019},
        eprint={1912.07976},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }

## Licence

MIT License

