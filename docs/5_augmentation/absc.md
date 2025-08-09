# Data Augmentation for Aspect-Based Sentiment Classification

Data augmentation is a powerful technique to improve model performance by artificially expanding your training dataset.
PyABSA provides built-in augmentation capabilities specifically designed for ABSA tasks that preserve aspect-sentiment
relationships while introducing beneficial variations.

## Why Use Data Augmentation for ABSA?

ABSA models often suffer from:

- **Limited training data** in specific domains
- **Class imbalance** between different sentiment polarities
- **Lack of linguistic diversity** in training examples
- **Poor generalization** to unseen text patterns

Data augmentation addresses these issues by creating additional training examples that maintain semantic meaning while
introducing useful variations.

## Automatic Augmentation Pipeline

PyABSA provides an automated augmentation system that handles the entire process for you.

### Basic Usage

```python
from pyabsa import AspectPolarityClassification as APC
from pyabsa.augmentation import auto_aspect_sentiment_classification_augmentation
import warnings

warnings.filterwarnings('ignore')

# Configure your model
config = APC.APCConfigManager.get_apc_config_english()
config.model = APC.APCModelList.FAST_LSA_T_V2
config.pretrained_bert = 'microsoft/deberta-v3-base'
config.num_epoch = 20
config.evaluate_begin = 5
config.max_seq_len = 80
config.dropout = 0.1
config.l2reg = 1e-8

# Automatic augmentation and training
auto_aspect_sentiment_classification_augmentation(
    config=config,
    dataset=APC.APCDatasetList.Restaurant14,
    device='cuda'
)
```

### Advanced Configuration

```python
# Fine-tune augmentation parameters
config.augmentation_ratio = 0.3  # Augment 30% of original data
config.augmentation_methods = [
    'synonym_replacement',
    'back_translation',
    'contextual_word_embedding',
    'random_insertion'
]

# Control augmentation quality
config.min_augmentation_confidence = 0.8
config.preserve_aspect_terms = True  # Keep aspect terms unchanged
config.max_augmentations_per_sample = 2
```

## Manual Augmentation Control

For more control over the augmentation process, you can use individual augmentation methods.

### Synonym Replacement

Replaces words with semantically similar synonyms while preserving aspects and sentiment.

```python
from pyabsa.augmentation import SynonymAugmentation

# Initialize synonym augmenter
syn_augmenter = SynonymAugmentation(
    preserve_aspects=True,
    replacement_ratio=0.15,
    min_confidence=0.8
)

# Apply to text
original = "The [B-ASP]food[E-ASP] was delicious and the [B-ASP]service[E-ASP] was excellent."
augmented = syn_augmenter.augment(original)
# Result: "The [B-ASP]food[E-ASP] was tasty and the [B-ASP]service[E-ASP] was outstanding."
```

### Back Translation

Translates text to another language and back to introduce natural variations.

```python
from pyabsa.augmentation import BackTranslationAugmentation

# Initialize back-translation augmenter
bt_augmenter = BackTranslationAugmentation(
    source_lang='en',
    intermediate_langs=['fr', 'de', 'es'],
    preserve_aspects=True
)

# Apply augmentation
augmented_texts = bt_augmenter.augment_batch([
    "The [B-ASP]pizza[E-ASP] was amazing!",
    "I hate the [B-ASP]waiting time[E-ASP] here."
])
```

### Contextual Word Embedding

Uses contextual embeddings to find suitable word replacements.

```python
from pyabsa.augmentation import ContextualAugmentation

# Initialize contextual augmenter
ctx_augmenter = ContextualAugmentation(
    model_name='bert-base-uncased',
    top_k=5,
    preserve_aspects=True,
    replacement_ratio=0.1
)

# Apply augmentation
augmented = ctx_augmenter.augment(
    "The [B-ASP]atmosphere[E-ASP] was cozy and inviting."
)
```

## Domain-Specific Augmentation

### Restaurant Domain

```python
from pyabsa.augmentation import DomainSpecificAugmentation

# Restaurant-focused augmentation
restaurant_augmenter = DomainSpecificAugmentation(
    domain='restaurant',
    aspect_categories=['food', 'service', 'ambiance', 'price'],
    domain_vocabulary='restaurant_terms.txt'
)

# Apply domain-specific transformations
augmented = restaurant_augmenter.augment(
    "The [B-ASP]pasta[E-ASP] was overpriced."
)
```

### Technology Domain

```python
# Technology product reviews
tech_augmenter = DomainSpecificAugmentation(
    domain='technology',
    aspect_categories=['performance', 'design', 'battery', 'display'],
    technical_terms=True
)
```

## Augmentation Quality Control

### Filtering Low-Quality Augmentations

```python
from pyabsa.augmentation import AugmentationFilter

# Initialize quality filter
quality_filter = AugmentationFilter(
    min_semantic_similarity=0.85,
    max_syntactic_distance=0.3,
    preserve_sentiment=True,
    aspect_consistency_check=True
)

# Filter augmented examples
filtered_data = quality_filter.filter_augmentations(augmented_dataset)
```

### Evaluation Metrics

```python
from pyabsa.augmentation import evaluate_augmentation_quality

# Evaluate augmentation quality
quality_metrics = evaluate_augmentation_quality(
    original_data=original_dataset,
    augmented_data=augmented_dataset,
    metrics=['semantic_similarity', 'syntactic_diversity', 'sentiment_preservation']
)

print(f"Semantic Similarity: {quality_metrics['semantic_similarity']:.3f}")
print(f"Syntactic Diversity: {quality_metrics['syntactic_diversity']:.3f}")
print(f"Sentiment Preservation: {quality_metrics['sentiment_preservation']:.3f}")
```

## Integration with Training Pipeline

### During Training

```python
# Enable augmentation during training
config.load_aug = True
config.augmentation_ratio = 0.25

# Train with automatic augmentation
trainer = APC.APCTrainer(
    config=config,
    dataset=APC.APCDatasetList.Laptop14,
    auto_device='cuda'
)
```

### Batch Augmentation

```python
from pyabsa.augmentation import BatchAugmentation

# Process entire datasets
batch_augmenter = BatchAugmentation(
    methods=['synonym', 'back_translation'],
    ratio=0.2,
    parallel=True,
    num_workers=4
)

# Augment dataset
augmented_dataset = batch_augmenter.augment_dataset(
    dataset_path="path/to/dataset",
    output_path="path/to/augmented_dataset"
)
```

## Best Practices

### Augmentation Guidelines

1. **Start Conservative**: Begin with low augmentation ratios (10-20%) and increase gradually
2. **Preserve Aspects**: Always maintain aspect term integrity during augmentation
3. **Quality over Quantity**: Focus on high-quality augmentations rather than large volumes
4. **Domain Relevance**: Use domain-specific augmentation for better results
5. **Validation**: Always validate augmented data before using in production

### Performance Optimization

```python
# Optimize for speed
config.augmentation_cache = True  # Cache augmented examples
config.parallel_augmentation = True  # Use multiple cores
config.augmentation_batch_size = 32  # Process in batches

# Memory management
config.augmentation_memory_limit = "4GB"
config.clear_augmentation_cache_frequency = 1000
```

### Common Pitfalls to Avoid

- **Over-augmentation**: Too much augmentation can introduce noise
- **Aspect corruption**: Ensure augmentation preserves aspect terms
- **Semantic drift**: Monitor that augmented text maintains original meaning
- **Class imbalance**: Don't amplify existing class imbalances through augmentation

## Measuring Augmentation Impact

### Before/After Comparison

```python
from pyabsa.evaluation import compare_model_performance

# Train baseline model (no augmentation)
baseline_results = train_baseline_model(config, dataset)

# Train augmented model
config.load_aug = True
augmented_results = train_augmented_model(config, dataset)

# Compare results
improvement = compare_model_performance(baseline_results, augmented_results)
print(f"Accuracy improvement: {improvement['accuracy']:.2%}")
print(f"F1-score improvement: {improvement['f1']:.2%}")
```

This comprehensive augmentation system helps you build more robust ABSA models that generalize better to real-world text
variations while maintaining high accuracy on aspect-sentiment relationships.
