# Configuration Guide

PyABSA provides a flexible configuration system that allows you to customize every aspect of model training, evaluation,
and inference. This guide explains how to work with configurations effectively.

## Understanding Configuration Objects

Configuration objects in PyABSA are Python namespace objects that contain all the parameters needed for a specific task.
They provide default values that work well out of the box, but can be easily customized for your specific needs.

### Checking Available Attributes

To see what attributes are available in a configuration object, you can use:

```python
# Check if an attribute exists
config.get('attribute_name', None)

# Print all attributes (for debugging)
print(vars(config))
```

### Setting and Getting Values

Configuration attributes can be accessed and modified like regular Python object attributes:

```python
# Set a value
config.learning_rate = 1e-5

# Get a value
print(config.learning_rate)

# Set any custom attribute
config.my_custom_parameter = "custom_value"
```

## Getting Configuration Objects

Each task in PyABSA provides a configuration manager with pre-defined templates.

### Aspect Polarity Classification (APC)

```python
from pyabsa import AspectPolarityClassification as APC

# Get default configurations for different languages
config_english = APC.APCConfigManager.get_apc_config_english()
config_chinese = APC.APCConfigManager.get_apc_config_chinese()
config_multilingual = APC.APCConfigManager.get_apc_config_multilingual()
```

### Text Classification

```python
from pyabsa import TextClassification as TC

# Get default text classification configuration
config = TC.TCConfigManager.get_tc_config_english()
```

### Aspect Term Extraction (ATE)

```python
from pyabsa import AspectTermExtraction as ATE

# Get default ATE configuration
config = ATE.ATEConfigManager.get_ate_config_english()
```

## Common Configuration Parameters

Here are the most frequently used configuration parameters across different tasks:

### Model and Architecture

```python
# Choose the model architecture
config.model = APC.APCModelList.FAST_LSA_T_V2

# Set the pre-trained backbone model
config.pretrained_bert = 'microsoft/deberta-v3-base'

# Maximum sequence length
config.max_seq_len = 80
```

### Training Parameters

```python
# Training epochs
config.num_epoch = 10

# When to start evaluation during training
config.evaluate_begin = 2

# Learning rate
config.learning_rate = 1e-5

# L2 regularization
config.l2reg = 1e-8

# Dropout rate
config.dropout = 0.5

# Batch size
config.train_batch_size = 16
config.eval_batch_size = 32

# Random seed for reproducibility
config.seed = 42
```

### Performance Optimization

```python
# Use Automatic Mixed Precision for faster training
config.use_amp = True

# Cache dataset in memory for faster data loading
config.cache_dataset = True

# Number of workers for data loading
config.num_workers = 4

# Device selection (auto-detected by default)
config.device = 'cuda'  # or 'cpu'
```

### Logging and Saving

```python
# How often to print training logs
config.log_step = 100

# Model saving options
config.model_path_to_save = './checkpoints'

# Whether to save the model state dict only
config.save_mode = 1  # 0: full model, 1: state dict only
```

## Advanced Configuration Examples

### Fine-tuning a Pre-trained Model

```python
from pyabsa import AspectPolarityClassification as APC

config = APC.APCConfigManager.get_apc_config_english()

# Use a more powerful backbone
config.pretrained_bert = 'microsoft/deberta-v3-large'

# Adjust learning rate for large models
config.learning_rate = 5e-6

# Increase max sequence length for longer texts
config.max_seq_len = 128

# Use gradient accumulation for large batch effects
config.gradient_accumulation_steps = 2
```

### Training for High Performance

```python
# Configuration for achieving best possible results
config.num_epoch = 20
config.evaluate_begin = 5
config.learning_rate = 1e-5
config.l2reg = 1e-8
config.dropout = 0.1

# Enable data augmentation
config.load_aug = True

# Use ensemble during evaluation
config.use_ensemble_inference = True
```

### Fast Prototyping Configuration

```python
# Configuration for quick experimentation
config.num_epoch = 3
config.evaluate_begin = 1
config.max_seq_len = 64
config.train_batch_size = 32
config.cache_dataset = True
config.use_amp = True
```

## Task-Specific Configurations

### For Aspect Term Extraction

```python
from pyabsa import AspectTermExtraction as ATE

config = ATE.ATEConfigManager.get_ate_config_english()

# ATE-specific parameters
config.window_size = 5  # Context window for aspect detection
config.use_syntax_based_srd = True  # Use syntax-based semantic relative distance
```

### For Text Classification

```python
from pyabsa import TextClassification as TC

config = TC.TCConfigManager.get_tc_config_english()

# TC-specific parameters
config.use_bert_spc = True  # Use BERT for sentence pair classification
config.class_dim = 3  # Number of classes in your classification task
```

## Saving and Loading Configurations

You can save your custom configurations for later use:

```python
import pickle

# Save configuration
with open('my_config.pkl', 'wb') as f:
    pickle.dump(config, f)

# Load configuration
with open('my_config.pkl', 'rb') as f:
    loaded_config = pickle.load(f)
```

## Best Practices

1. **Start with defaults**: Use the pre-defined configuration templates as starting points
2. **Incremental changes**: Modify one parameter at a time to understand its impact
3. **Document changes**: Keep track of which parameters you've modified
4. **Reproducibility**: Always set a fixed seed for reproducible results
5. **Validation**: Test your configuration on a small dataset first

## Troubleshooting

### Common Issues

- **Out of memory**: Reduce `train_batch_size`, `max_seq_len`, or enable `use_amp`
- **Slow training**: Enable `cache_dataset`, increase `num_workers`, or use `use_amp`
- **Poor performance**: Try different `learning_rate`, increase `num_epoch`, or use a better `pretrained_bert`

### Getting Help

If you encounter issues with configurations, check:

1. The parameter spelling and type
2. Compatibility between different parameters
3. Hardware limitations (GPU memory, CPU cores)
4. The PyABSA GitHub issues for similar problems
