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
# | [Overview](../README.MD)  |  [About HuggingfaceHub](huggingface_readme.md)  |  [About ABDADatasets](dataset_readme.md)  |  [About Models](model_readme.md) |  [About Application](tutorial_readme.md) |  

## Our Models for ABSA

### ATEPC

1. [LCF-ATEPC](../pyabsa/core/atepc/models/lcf_atepc.py)
2. [LCF-ATEPC-LARGE](../pyabsa/core/atepc/models/lcf_atepc_large.py) (Dual BERT)
2. [FAST-LCF-ATEPC](../pyabsa/core/atepc/models/fast_lcf_atepc.py)
3. [LCFS-ATEPC](../pyabsa/core/atepc/models/lcfs_atepc.py)
4. [LCFS-ATEPC-LARGE](../pyabsa/core/atepc/models/lcfs_atepc_large.py) (Dual BERT)
5. [FAST-LCFS-ATEPC](../pyabsa/core/atepc/models/fast_lcfs_atepc.py)
6. [BERT-BASE](../pyabsa/core/atepc/models/bert_base_atepc.py)

### APC

#### Bert-based APC models

1. [SLIDE-LCF-BERT](../pyabsa/core/apc/models/lsa_t.py) (Faster & Performs Better than LCF/LCFS-BERT)
2. [SLIDE-LCFS-BERT](../pyabsa/core/apc/models/lsa_s.py) (Faster & Performs Better than LCF/LCFS-BERT)
3. [LCF-BERT](../pyabsa/core/apc/models/lcf_bert.py) (Reimplemented & Enhanced)
4. [LCFS-BERT](../pyabsa/core/apc/models/lcfs_bert.py) (Reimplemented & Enhanced)
5. [FAST-LCF-BERT](../pyabsa/core/apc/models/fast_lcf_bert.py) (Faster with slightly performance loss)
6. [FAST_LCFS-BERT](../pyabsa/core/apc/models/fast_lcfs_bert.py) (Faster with slightly performance loss)
7. [LCF-DUAL-BERT](../pyabsa/core/apc/models/lcf_dual_bert.py) (Dual BERT)
8. [LCFS-DUAL-BERT](../pyabsa/core/apc/models/lcfs_dual_bert.py) (Dual BERT)
9. [BERT-BASE](../pyabsa/core/apc/models/bert_base.py)
10. [BERT-SPC](../pyabsa/core/apc/models/bert_spc.py)
11. [LCA-Net](../pyabsa/core/apc/models/lca_bert.py)
12. [DLCF-DCA-BERT *](../pyabsa/core/apc/models/dlcf_dca_bert.py)

#### Bert-based APC baseline models

1. [AOA_BERT](../pyabsa/core/apc/classic/__bert__/models/aoa_bert.py)
2. [ASGCN_BERT](../pyabsa/core/apc/classic/__bert__/models/asgcn_bert.py)
3. [ATAE_LSTM_BERT](../pyabsa/core/apc/classic/__bert__/models/atae_lstm_bert.py)
4. [Cabasc_BERT](../pyabsa/core/apc/classic/__bert__/models/cabasc_bert.py)
5. [IAN_BERT](../pyabsa/core/apc/classic/__bert__/models/ian_bert.py)
6. [LSTM_BERT](../pyabsa/core/apc/classic/__bert__/models/lstm_bert.py)
7. [MemNet_BERT](../pyabsa/core/apc/classic/__bert__/models/memnet_bert.py)
8. [MGAN_BERT](../pyabsa/core/apc/classic/__bert__/models/mgan_bert.py)
9. [RAM_BERT](../pyabsa/core/apc/classic/__bert__/models/ram_bert.py)
10. [TD_LSTM_BERT](../pyabsa/core/apc/classic/__bert__/models/td_lstm_bert.py)
11. [TC_LSTM_BERT](../pyabsa/core/apc/classic/__bert__/models/tc_lstm_bert.py)
12. [TNet_LF_BERT](../pyabsa/core/apc/classic/__bert__/models/tnet_lf_bert.py)

#### GloVe-based APC baseline models

1. [AOA](../pyabsa/core/apc/classic/__glove__/models/aoa.py)
2. [ASGCN](../pyabsa/core/apc/classic/__glove__/models/asgcn.py)
3. [ATAE-LSTM](../pyabsa/core/apc/classic/__glove__/models/atae_lstm.py)
4. [Cabasc](../pyabsa/core/apc/classic/__glove__/models/cabasc.py)
5. [IAN](../pyabsa/core/apc/classic/__glove__/models/ian.py)
6. [LSTM](../pyabsa/core/apc/classic/__glove__/models/lstm.py)
7. [MemNet](../pyabsa/core/apc/classic/__glove__/models/memnet.py)
8. [MGAN](../pyabsa/core/apc/classic/__glove__/models/mgan.py)
9. [RAM](../pyabsa/core/apc/classic/__glove__/models/ram.py)
10. [TD-LSTM](../pyabsa/core/apc/classic/__glove__/models/td_lstm.py)
11. [TD-LSTM](../pyabsa/core/apc/classic/__glove__/models/tc_lstm.py)
12. [TNet_LF](../pyabsa/core/apc/classic/__glove__/models/tnet_lf.py)
