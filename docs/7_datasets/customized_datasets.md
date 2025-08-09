# Working with Datasets

PyABSA provides a comprehensive system for working with datasets, supporting both built-in benchmark datasets and custom
data. This guide covers everything you need to know about preparing, loading, and using datasets effectively.

## Built-in Datasets

PyABSA includes access to standard benchmark datasets that are commonly used in ABSA research. These datasets are
automatically downloaded and cached when you first use them.

### Available Dataset Categories

#### Aspect Polarity Classification (APC)

```python
from pyabsa import AspectPolarityClassification as APC

# Popular benchmark datasets
dataset = APC.APCDatasetList.Restaurant14
dataset = APC.APCDatasetList.Laptop14
dataset = APC.APCDatasetList.Restaurant15
dataset = APC.APCDatasetList.Restaurant16

# Multilingual datasets
dataset = APC.APCDatasetList.Multilingual
```

#### Aspect Term Extraction (ATE)

```python
from pyabsa import AspectTermExtraction as ATE

# Standard ATE datasets
dataset = ATE.ATEDatasetList.Restaurant14
dataset = ATE.ATEDatasetList.Laptop14
```

#### Aspect Sentiment Triplet Extraction (ASTE)

```python
from pyabsa import AspectSentimentTripletExtraction as ASTE

# ASTE benchmark datasets
dataset = ASTE.ASTEDatasetList.Restaurant14
dataset = ASTE.ASTEDatasetList.Restaurant15
dataset = ASTE.ASTEDatasetList.Restaurant16
```

### Using Built-in Datasets

```python
# Simply pass the dataset to your trainer
trainer = APC.APCTrainer(
    config=config,
    dataset=APC.APCDatasetList.Restaurant14,
    auto_device=DeviceTypeOption.AUTO
)
```

## Custom Dataset Preparation

To use your own datasets, you need to follow specific formatting conventions that PyABSA expects.

### Data Format Specifications

#### For Aspect Polarity Classification (APC)

Each line should contain a sentence with marked aspects followed by sentiment labels:

```
[sentence with marked aspects] $LABEL$ [sentiment1, sentiment2, ...]
```

**Example:**

```
The [B-ASP]food[E-ASP] was excellent, but the [B-ASP]service[E-ASP] was terrible. $LABEL$ Positive, Negative
```

**Aspect Marking Rules:**

- Use `[B-ASP]` to mark the beginning of an aspect term
- Use `[E-ASP]` to mark the end of an aspect term
- Multiple aspects in one sentence are supported

**Sentiment Labels:**

- `Positive` or `1` for positive sentiment
- `Negative` or `-1` for negative sentiment
- `Neutral` or `0` for neutral sentiment

#### For Aspect Term Extraction (ATE)

Format is similar but without sentiment labels:

```
The [B-ASP]food[E-ASP] was excellent, but the [B-ASP]service[E-ASP] was terrible. $LABEL$
```

#### For Aspect Sentiment Triplet Extraction (ASTE)

Uses BIO tagging for aspects and opinions:

```
The food was excellent , but the service was terrible . $LABEL$ B-ASP I-ASP O B-OP O O B-ASP I-ASP O B-OP O
```

### File Organization

Organize your dataset files using this structure:

```
my_dataset/
├── train.dat.apc         # Training data
├── test.dat.apc          # Test data
├── valid.dat.apc         # Validation data (optional)
└── readme.txt            # Dataset description (optional)
```

**File Naming Conventions:**

- Training files: `train.dat.[task]` (e.g., `train.dat.apc`, `train.dat.ate`)
- Test files: `test.dat.[task]`
- Validation files: `valid.dat.[task]` (optional)

### Loading Custom Datasets

#### Method 1: Direct Path

```python
from pyabsa import AspectPolarityClassification as APC

# Point to your dataset directory
dataset_path = "path/to/my_dataset"

trainer = APC.APCTrainer(
    config=config,
    dataset=dataset_path,
    auto_device=DeviceTypeOption.AUTO
)
```

#### Method 2: Using DatasetItem

```python
from pyabsa.utils.data_utils.dataset_item import DatasetItem

# Create a dataset item
my_dataset = DatasetItem(
    dataset_name="MyCustomDataset",
    dataset_path="path/to/my_dataset"
)

trainer = APC.APCTrainer(
    config=config,
    dataset=my_dataset,
    auto_device=DeviceTypeOption.AUTO
)
```

## Data Preprocessing and Validation

### Automatic Data Validation

PyABSA automatically validates your dataset format when loading:

```python
# The framework will check for:
# - Correct file naming
# - Valid data format
# - Consistent labeling
# - Missing files
```

### Manual Data Validation

You can validate your dataset before training:

```python
from pyabsa.utils.data_utils import validate_dataset

# Validate your dataset format
is_valid, errors = validate_dataset("path/to/my_dataset", task="apc")

if not is_valid:
    print("Dataset validation errors:")
    for error in errors:
        print(f"- {error}")
```

## Data Augmentation

PyABSA supports automatic data augmentation to improve model robustness:

### Enabling Augmentation

```python
# Enable augmentation during training
config.load_aug = True

# Or specify augmentation parameters
config.augmentation_ratio = 0.2  # Augment 20% of training data
config.augmentation_methods = ["synonym", "back_translation"]
```

### Custom Augmentation

```python
from pyabsa.augmentation import TextAugmentation

# Create custom augmentation pipeline
augmenter = TextAugmentation(
    methods=["synonym_replacement", "random_insertion"],
    ratio=0.15
)

# Apply to your dataset
augmented_data = augmenter.augment_dataset("path/to/dataset")
```

## Advanced Dataset Features

### Multi-language Support

```python
# Specify language for your dataset
config.language = "en"  # English
config.language = "zh"  # Chinese
config.language = "multilingual"  # Multi-language

# Use appropriate tokenizer
config.pretrained_bert = "bert-base-multilingual-cased"
```

### Handling Imbalanced Data

```python
# Enable class balancing
config.balance_classes = True

# Use weighted loss
config.use_class_weights = True

# Oversample minority classes
config.oversample_ratio = 2.0
```

### Cross-domain Datasets

```python
# Training on multiple domains
config.cross_domain_training = True
config.domain_adaptation = True

# Specify source and target domains
config.source_domain = "restaurant"
config.target_domain = "laptop"
```

## Best Practices

### Data Quality Guidelines

1. **Consistent Labeling**: Ensure sentiment labels are consistent across your dataset
2. **Balanced Distribution**: Aim for balanced sentiment distribution when possible
3. **Quality Control**: Review a sample of your data manually before training
4. **Sufficient Size**: Use at least 1000 examples for training, more for better performance

### Performance Optimization

```python
# Cache processed datasets for faster loading
config.cache_dataset = True

# Use multiple workers for data loading
config.num_workers = 4

# Enable dataset preprocessing
config.preprocess_data = True
```

### Error Handling

```python
# Handle missing files gracefully
config.ignore_error = True

# Skip malformed examples
config.skip_errors = True

# Log data loading issues
config.verbose_data_loading = True
```

## Troubleshooting Common Issues

### Format Errors

**Problem**: "Invalid data format" error  
**Solution**: Check that aspects are properly marked with `[B-ASP]` and `[E-ASP]` tags

**Problem**: "Missing label separator" error  
**Solution**: Ensure each line contains `$LABEL$` separator

### Loading Errors

**Problem**: "Dataset not found" error  
**Solution**: Verify file paths and naming conventions

**Problem**: "Empty dataset" error  
**Solution**: Check that data files contain actual content

### Performance Issues

**Problem**: Slow data loading  
**Solution**: Enable `cache_dataset=True` and increase `num_workers`

**Problem**: Out of memory during data loading  
**Solution**: Reduce `max_seq_len` or `train_batch_size`

For additional help with dataset issues, check
the [PyABSA GitHub repository](https://github.com/yangheng95/PyABSA/issues) or refer to the example datasets in the
`examples-v2` directory.
