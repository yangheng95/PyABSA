# Text Auto-Augmentation for Classification

This guide explains how to use PyABSA's text auto-augmentation features to improve the performance of your
classification models. Augmentation can help increase the diversity of your training data and make your models more
robust.

## Augmentation for Aspect-Based Sentiment Classification

PyABSA provides a simple function to automatically augment your dataset and train an Aspect-Based Sentiment
Classification (ABSC) model.

### Example Usage

Here’s how you can use `auto_aspect_sentiment_classification_augmentation` to train an augmented model.

```python
from pyabsa import AspectPolarityClassification as APC
from pyabsa.augmentation import auto_aspect_sentiment_classification_augmentation

# Get a configuration template
config = APC.APCConfigManager.get_apc_config_english()

# Set the model and BERT checkpoint
config.model = APC.APCModelList.FAST_LSA_T_V2
config.pretrained_bert = 'microsoft/deberta-v3-base'

# Set training hyperparameters
config.num_epoch = 10
config.evaluate_begin = 5
config.max_seq_len = 80
config.log_step = 100
config.dropout = 0.5
config.l2reg = 1e-8
config.seed = 42

# Choose a dataset
dataset = APC.APCDatasetList.Laptop14

# This function will automatically augment the dataset and train the model
auto_aspect_sentiment_classification_augmentation(
    config=config,
    dataset=dataset,
    device='cuda'  # Use 'cpu' if you don't have a GPU
)
```

This function handles the augmentation process behind the scenes, so you don't need to manually generate new training
examples.

## Augmentation for Text Classification

Similarly, you can use auto-augmentation for standard text classification tasks.

### Example Usage

Here’s how to use `auto_classification_augmentation` to train an augmented text classification model.

```python
from pyabsa import TextClassification as TC
from pyabsa.augmentation import auto_classification_augmentation

# Get a configuration template
config = TC.TCConfigManager.get_tc_config_english()

# Set the model and training hyperparameters
config.model = TC.BERTTCModelList.BERT_MLP
config.num_epoch = 5
config.evaluate_begin = 2
config.max_seq_len = 80
config.dropout = 0.5
config.seed = 42
config.l2reg = 1e-5

# Choose a dataset
dataset = TC.TCDatasetList.SST2

# This function will automatically augment the dataset and train the model
auto_classification_augmentation(
    config=config,
    dataset=dataset,
    device='cuda'  # Use 'cpu' if you don't have a GPU
)
```

By using these auto-augmentation functions, you can potentially improve your model's performance with minimal effort.
For more advanced control over the augmentation process, refer to the detailed tutorials in the documentation.
