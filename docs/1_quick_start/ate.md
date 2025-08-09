# Aspect Term Extraction (ATE)

This guide provides a quick introduction to performing Aspect Term Extraction (ATE) using PyABSA. You'll learn how to
use pre-trained models to extract aspect terms from text.

## Inference with Pre-trained Models

PyABSA includes pre-trained models that can identify and extract aspect terms from sentences. Here’s how to get started.

### Loading an Extractor

First, import the necessary components and load an aspect extractor. PyABSA will automatically download the required
checkpoint if it's not available locally.

```python
from pyabsa import AspectTermExtraction as ATEPC

# Load a pre-trained aspect extractor
aspect_extractor = ATEPC.AspectExtractor('multilingual')
```

### Running Predictions

Once the extractor is loaded, you can use it to extract aspect terms from a sentence.

```python
# Extract aspect terms from a single sentence
aspect_extractor.predict(
    'The food was good, but the service was terrible.'
)
```

You can also run predictions on a batch of sentences:

```python
# Extract aspect terms from multiple sentences
examples = [
    'The food was good, but the service was terrible.',
    'The screen is amazing, but the battery life is short.'
]
aspect_extractor.predict(examples)
```

### Batch Prediction on a Dataset

For larger datasets, you can use the `batch_predict` method. PyABSA provides several built-in datasets for convenience.

```python
# Use a built-in dataset for batch prediction
inference_set = ATEPC.ATEPCDatasetList.SemEval

# Set pred_sentiment to False to only perform aspect extraction
results = aspect_extractor.batch_predict(
    target_file=inference_set,
    print_result=True,
    save_result=True,
    pred_sentiment=False,
)
```

By following these steps, you can easily integrate Aspect Term Extraction into your projects using PyABSA. For more
advanced use cases, such as training your own models, refer to the detailed tutorials in the documentation.
