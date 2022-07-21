# MetricVisualizer - for easy managing performance metric
![PyPI - Python Version](https://img.shields.io/badge/python-3.6-blue.svg)
[![PyPI](https://img.shields.io/pypi/v/metric-visualizer)](https://pypi.org/project/metric-visualizer/)
[![Downloads](https://pepy.tech/badge/metric-visualizer)](https://pepy.tech/project/metric-visualizer)
[![Downloads](https://pepy.tech/badge/metric-visualizer/month)](https://pepy.tech/project/metric-visualizer)
[![Downloads](https://pepy.tech/badge/metric-visualizer/week)](https://pepy.tech/project/metric-visualizer)

## Automated metric visualization for comparison experiments
- Box plot
- Trajectory plot
- Scatter plot
- Bar plot
- Violin plot
- Rank test
- On the way

## Install
If you want to make tikz(latex) plots, you need to install texlive (other latex release version are not tested).

```bash
pip install metric_visualizer
```

## 用法说明 Usage
假设存在多组对比实验(或者一组参数设置)，则称之为trial，每组实验存在多个metric(例如AUC，Accuracy，F1，Loss等)，
每组参照实验重复n词，则使用以下方法监听实验结果：
Assume that there exist multiple sets of comparison experiments (or a set of parameter settings), called trials, with multiple metrics (e.g., AUC, accuracy, F1, loss, etc.) for each set of experiments.
Repeat n words for each set of reference experiments, and then listen to the results of the experiments using the following method.


```python
import numpy as np
from metric_visualizer import MetricVisualizer

MV = MetricVisualizer()

trial_num = 5  # number of different trials,
repeat = 10  # number of repeats
metric_num = 3  # number of metrics

for trial in range(trial_num):
    for r in range(repeat):  # repeat the experiments to plot violin or box figure
        metrics = [(np.random.random() + n) for n in range(metric_num)]  # n is metric scale factor
        for i, m in enumerate(metrics):
            MV.add_metric('Metric-{}'.format(i + 1), round(m, 2))  # Add metric by metric name
    MV.next_trial()  # move to next trial

```

画图代码如下：
```python

save_prefix = None
MV.summary(save_path=save_prefix, no_print=True)  # save fig into .tex and .pdf format
MV.traj_plot_by_trial(save_name=save_prefix, xlabel='', xrotation=30, minorticks_on=True)  # save fig into .tex and .pdf format
MV.violin_plot_by_trial(save_name=save_prefix)  # save fig into .tex and .pdf format
MV.box_plot_by_trial(save_name=save_prefix)  # save fig into .tex and .pdf format
MV.avg_bar_plot_by_trial(save_name=save_prefix)  # save fig into .tex and .pdf format
MV.sum_bar_plot_by_trial(save_name=save_prefix)  # save fig into .tex and .pdf format

# 此函数适合对比不同模型性能，每个模型代表一个trial，综合多个metric进行Scott-Knott Rank Test，并绘制箱型图
MV.scott_knott_plot(save_name=save_prefix, minorticks_on=False)  

print(MV.rank_test_by_trail('trial0'))  # save fig into .tex and .pdf format
print(MV.rank_test_by_metric('metric1'))  # save fig into .tex and .pdf format


# save_path = None
# MV.summary(save_path=save_path)  # save fig into .tex and .pdf format
# MV.traj_plot_by_metric(save_path=save_path, xlabel='', xrotation=30)  # save fig into .tex and .pdf format
# MV.violin_plot_by_metric(save_path=save_path)  # save fig into .tex and .pdf format
# MV.box_plot_by_metric(save_path=save_path)  # save fig into .tex and .pdf format
# MV.avg_bar_plot_by_metric(save_path=save_path)  # save fig into .tex and .pdf format
# MV.sum_bar_plot_by_metric(save_path=save_path)  # save fig into .tex and .pdf format

```

```html
 -------------------- Metric Summary --------------------
╒══════════╤═════════╤══════════════════════════════════════════════════════════════╤═════════════════════════════════════════════════════════════╕
│ Metric   │ Trial   │ Values                                                       │ Summary                                                     │
╞══════════╪═════════╪══════════════════════════════════════════════════════════════╪═════════════════════════════════════════════════════════════╡
│ Metric-1 │ trial-0 │ [0.35, 0.65, 0.67, 0.51, 0.04, 0.43, 0.46, 0.58, 0.11, 0.66] │ ['Avg:0.45, Median: 0.48, IQR: 0.22, Max: 0.67, Min: 0.04'] │
├──────────┼─────────┼──────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Metric-1 │ trial-1 │ [0.52, 0.1, 0.11, 0.86, 0.49, 0.7, 0.77, 0.96, 0.16, 0.65]   │ ['Avg:0.53, Median: 0.58, IQR: 0.41, Max: 0.96, Min: 0.1']  │
├──────────┼─────────┼──────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Metric-1 │ trial-2 │ [0.73, 0.99, 0.13, 0.72, 0.63, 0.61, 0.14, 0.85, 0.71, 0.86] │ ['Avg:0.64, Median: 0.72, IQR: 0.17, Max: 0.99, Min: 0.13'] │
├──────────┼─────────┼──────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Metric-1 │ trial-3 │ [0.99, 0.69, 0.86, 0.2, 0.4, 0.1, 0.05, 0.07, 0.95, 0.31]    │ ['Avg:0.46, Median: 0.36, IQR: 0.62, Max: 0.99, Min: 0.05'] │
├──────────┼─────────┼──────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Metric-1 │ trial-4 │ [0.58, 0.95, 0.73, 0.63, 0.04, 0.19, 0.5, 0.9, 0.64, 0.89]   │ ['Avg:0.6, Median: 0.64, IQR: 0.27, Max: 0.95, Min: 0.04']  │
├──────────┼─────────┼──────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Metric-2 │ trial-0 │ [1.58, 1.32, 1.98, 1.76, 1.31, 1.6, 1.6, 1.22, 1.3, 1.19]    │ ['Avg:1.49, Median: 1.45, IQR: 0.29, Max: 1.98, Min: 1.19'] │
├──────────┼─────────┼──────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Metric-2 │ trial-1 │ [1.88, 1.67, 1.77, 1.94, 1.01, 1.6, 1.25, 1.63, 1.62, 1.91]  │ ['Avg:1.63, Median: 1.65, IQR: 0.21, Max: 1.94, Min: 1.01'] │
├──────────┼─────────┼──────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Metric-2 │ trial-2 │ [1.4, 1.94, 1.28, 1.78, 1.01, 1.08, 1.21, 1.82, 1.78, 1.18]  │ ['Avg:1.45, Median: 1.34, IQR: 0.59, Max: 1.94, Min: 1.01'] │
├──────────┼─────────┼──────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Metric-2 │ trial-3 │ [1.79, 1.35, 1.14, 1.5, 1.73, 1.06, 1.98, 1.75, 1.07, 1.49]  │ ['Avg:1.49, Median: 1.5, IQR: 0.49, Max: 1.98, Min: 1.06']  │
├──────────┼─────────┼──────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Metric-2 │ trial-4 │ [1.93, 1.81, 1.18, 1.08, 1.57, 1.85, 1.95, 1.94, 1.58, 1.35] │ ['Avg:1.62, Median: 1.7, IQR: 0.43, Max: 1.95, Min: 1.08']  │
├──────────┼─────────┼──────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Metric-3 │ trial-0 │ [2.85, 2.87, 2.3, 2.05, 2.86, 2.34, 2.85, 2.3, 2.95, 2.53]   │ ['Avg:2.59, Median: 2.69, IQR: 0.54, Max: 2.95, Min: 2.05'] │
├──────────┼─────────┼──────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Metric-3 │ trial-1 │ [2.31, 2.41, 2.34, 2.96, 2.48, 2.68, 2.99, 2.94, 2.01, 2.46] │ ['Avg:2.56, Median: 2.47, IQR: 0.44, Max: 2.99, Min: 2.01'] │
├──────────┼─────────┼──────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Metric-3 │ trial-2 │ [2.65, 2.5, 2.68, 2.34, 2.32, 2.61, 2.61, 2.88, 2.86, 2.36]  │ ['Avg:2.58, Median: 2.61, IQR: 0.24, Max: 2.88, Min: 2.32'] │
├──────────┼─────────┼──────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Metric-3 │ trial-3 │ [2.29, 2.12, 2.4, 2.81, 2.5, 2.54, 2.82, 2.61, 2.45, 2.44]   │ ['Avg:2.5, Median: 2.48, IQR: 0.16, Max: 2.82, Min: 2.12']  │
├──────────┼─────────┼──────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Metric-3 │ trial-4 │ [2.41, 2.12, 2.31, 2.29, 2.46, 2.95, 2.74, 2.66, 2.34, 2.65] │ ['Avg:2.49, Median: 2.44, IQR: 0.33, Max: 2.95, Min: 2.12'] │
╘══════════╧═════════╧══════════════════════════════════════════════════════════════╧═════════════════════════════════════════════════════════════╛
 -------------------- Metric Summary --------------------
```
## Plot via Matplotlib (or Tikz)

### Traj Plot [tikz version](fig/example_metric_traj_plot.pdf)

![traj_plot_example](fig/traj_plot_example.png)

### Box Plot [tikz version](fig/example_metric_box_plot.pdf)

![box_plot_example](fig/box_plot_example.png)

### Violin Plot [tikz version](fig/example_metric_violin_plot.pdf)

![violin_plot_example](fig/violin_plot_example.png)

### Average Bar Plot [tikz version](fig/example_metric_avg_bar_plot.pdf)

![violin_plot_example](fig/avg_bar_plot_example.png)

### Sum Bar Plot [tikz version](fig/example_metric_sum_bar_plot.pdf)

![violin_plot_example](fig/sum_bar_plot_example.png)

## Real Usage Example in PyABSA

To analyze the impact of max_seq_len, we can use MetricVisualizer as following:

```bash
pip install pyabsa  # install pyabsa
```

```python3
import autocuda
import random

from metric_visualizer import MetricVisualizer

from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList

import warnings

device = autocuda.auto_cuda()
warnings.filterwarnings('ignore')

seeds = [random.randint(0, 10000) for _ in range(3)]

max_seq_lens = [60, 70, 80, 90, 100]

apc_config_english = APCConfigManager.get_apc_config_english()
apc_config_english.model = APCModelList.FAST_LCF_BERT
apc_config_english.lcf = 'cdw'
apc_config_english.max_seq_len = 80
apc_config_english.cache_dataset = False
apc_config_english.patience = 10
apc_config_english.seed = seeds

MV = MetricVisualizer()
apc_config_english.MV = MV

for eta in max_seq_lens:
    apc_config_english.eta = eta
    dataset = ABSADatasetList.Laptop14
    Trainer(config=apc_config_english,
            dataset=dataset,  # train set and test set will be automatically detected
            checkpoint_save_mode=0,  # =None to avoid save model
            auto_device=device  # automatic choose CUDA or CPU
            )
    apc_config_english.MV.next_trial()

apc_config_english.MV.summary(save_path=None, xticks=max_seq_lens)
apc_config_english.MV.traj_plot(save_name=None, xticks=max_seq_lens)
apc_config_english.MV.violin_plot(save_name=None, xticks=max_seq_lens)
apc_config_english.MV.box_plot(save_name=None, xticks=max_seq_lens)

save_path = '{}_{}'.format(apc_config_english.model_name, apc_config_english.dataset_name)
apc_config_english.MV.summary(save_path=save_path)
apc_config_english.MV.traj_plot(save_name=save_path, xticks=max_seq_lens, xlabel=r'$\eta$')
apc_config_english.MV.violin_plot(save_name=save_path, xticks=max_seq_lens, xlabel=r'$\eta$')
apc_config_english.MV.box_plot(save_name=save_path, xticks=max_seq_lens, xlabel=r'$\eta$')
```
