# Visualizing Metrics

PyABSA includes a powerful tool for visualizing model performance metrics, which is built on top of the
`metric-visualizer` library. This feature allows you to track and compare the performance of your models across
different runs and configurations, making it easier to analyze and interpret your results.

## How It Works

During training, PyABSA can save the performance metrics from each evaluation step into a special file with a `.mv`
extension. You can then use the `mvis` command-line tool to generate various plots and visualizations from these files.

## Generating Metric Files

To enable metric visualization, you need to create a `MetricVisualizer` object and pass it to your training
configuration.

### Example

Here’s a simple example of how to train a model and generate a metric visualization file.

```python
from pyabsa import AspectPolarityClassification as APC
from pyabsa import ModelSaveOption, DeviceTypeOption
from metric_visualizer import MetricVisualizer

# 1. Create a MetricVisualizer object
mv = MetricVisualizer()

# 2. Get a configuration object
config = APC.APCConfigManager.get_apc_config_english()

# 3. Pass the MetricVisualizer object to the config
config.MV = mv

# Set other configuration options
config.model = APC.APCModelList.FAST_LSA_T_V2
config.num_epoch = 5
config.seed = 42

# Choose a dataset
dataset = APC.APCDatasetList.Laptop14

# Start the training
trainer = APC.APCTrainer(
    config=config,
    dataset=dataset,
    checkpoint_save_mode=ModelSaveOption.DO_NOT_SAVE_MODEL,
    auto_device=DeviceTypeOption.AUTO,
)

# After training, a .mv file will be created in your working directory.
```

## Visualizing the Metrics

Once you have a `.mv` file, you can use the `mvis` command-line tool to visualize the metrics. This tool is
automatically installed as a dependency of PyABSA.

```bash
# Visualize a single metric file
mvis <your_metric_file>.mv

# You can also visualize multiple files at once to compare runs
mvis run1.mv run2.mv run3.mv
```

This will generate a variety of plots, such as trajectory plots, box plots, and violin plots, which can help you
understand your model's performance in detail.

By using the metric visualization tools, you can gain deeper insights into your experiments and make more informed
decisions about your models and hyperparameters.
