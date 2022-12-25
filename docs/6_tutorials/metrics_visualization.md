Metric Visualizations
=====================
This feature allows you to visualize the metrics of your model. It is based on [metric-visualizer](https://github.com/yangheng95/metric-visualizer).
Here is an example of the visualization using pyabsa.

### Tips
Each run will generate a new visualization file, e.g., `*.mv`
You can use mvis from installed metric-visualizer to visualize the metrics in a bash script, e.g.,

```bash
mvis *.mv  # no need to manually install metric-visualizer, it will be installed by pyabsa
```

### Example of visualization
```python3
import autocuda
import random

from metric_visualizer import MetricVisualizer

from pyabsa import AspectPolarityClassification as APC

import warnings

device = autocuda.auto_cuda()
warnings.filterwarnings('ignore')

seeds = [random.randint(0, 10000) for _ in range(3)]

max_seq_lens = [60, 70, 80, 90, 100]

apc_config_english = APC.APCConfigManager.get_apc_config_english()
apc_config_english.model = APC.APCModelList.FAST_LCF_BERT
apc_config_english.lcf = 'cdw'
apc_config_english.max_seq_len = 80
apc_config_english.cache_dataset = False
apc_config_english.patience = 10
apc_config_english.seed = seeds

MV = MetricVisualizer()
apc_config_english.MV = MV

for eta in max_seq_lens:
    apc_config_english.eta = eta
    dataset = APC.APCDatasetList.Laptop14
    APC.APCTrainer(config=apc_config_english,
                   dataset=dataset,  # train set and test set will be automatically detected
                   checkpoint_save_mode=0,  # =None to avoid save model
                   auto_device=device  # automatic choose CUDA or CPU
                   )
    apc_config_english.MV.next_trial()

save_prefix = '{}_{}'.format(apc_config_english.model_name, apc_config_english.dataset_name)

MV.summary(save_path=save_prefix, no_print=True)  # save fig_preview into .tex and .pdf format
MV.traj_plot_by_trial(save_path=save_prefix, xlabel='', xrotation=30,
                      minorticks_on=True)  # save fig_preview into .tex and .pdf format
MV.violin_plot_by_trial(save_path=save_prefix, xticks=max_seq_lens,
                        xlabel=r'$\eta$')  # save fig_preview into .tex and .pdf format
MV.box_plot_by_trial(save_path=save_prefix, xticks=max_seq_lens,
                     xlabel=r'$\eta$')  # save fig_preview into .tex and .pdf format
MV.avg_bar_plot_by_trial(save_path=save_prefix, xticks=max_seq_lens,
                         xlabel=r'$\eta$')  # save fig_preview into .tex and .pdf format
MV.sum_bar_plot_by_trial(save_path=save_prefix, xticks=max_seq_lens,
                         xlabel=r'$\eta$')  # save fig_preview into .tex and .pdf format
MV.scott_knott_plot(save_path=save_prefix, minorticks_on=False, xticks=max_seq_lens,
                    xlabel=r'$\eta$')  # save fig_preview into .tex and .pdf format

# print(MV.rank_test_by_trail('trial0'))  # save fig_preview into .tex and .pdf format
# print(MV.rank_test_by_metric('metric1'))  # save fig_preview into .tex and .pdf format

```
