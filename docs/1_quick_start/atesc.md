# Aspect Term Extraction and Sentiment Classification (ATESC)

This guide demonstrates how to perform both Aspect Term Extraction and Sentiment Classification (ATESC) in a single step
using PyABSA. You'll learn how to use pre-trained models for inference and how to train your own models.

## Inference with Pre-trained Models

PyABSA's pre-trained models can extract aspect terms and classify their sentiment simultaneously. Here’s how to get
started.

### Loading an Extractor

First, import the necessary components and load an aspect extractor. PyABSA will automatically download the required
checkpoint if it's not available locally.

```python
from pyabsa import AspectTermExtraction as ATEPC

# Load a pre-trained aspect extractor
aspect_extractor = ATEPC.AspectExtractor('multilingual')
```

### Running Predictions

Once the extractor is loaded, you can use it to extract aspect terms and their corresponding sentiments from a sentence.

```python
# Extract aspects and sentiments from a single sentence
aspect_extractor.predict(
    'The food was good, but the service was terrible.'
)
```

You can also run predictions on a batch of sentences:

```python
# Extract aspects and sentiments from multiple sentences
examples = [
    'The food was good, but the service was terrible.',
    'The screen is amazing, but the battery life is short.'
]
aspect_extractor.predict(examples)
```

### Batch Prediction on a Dataset

For larger datasets, you can use the `batch_predict` method. By default, it performs both aspect extraction and
sentiment classification.

```python
# Use a built-in dataset for batch prediction
inference_set = ATEPC.ATEPCDatasetList.SemEval

# Set pred_sentiment to True to perform both extraction and classification
results = aspect_extractor.batch_predict(
    target_file=inference_set,
    print_result=True,
    save_result=True,
    pred_sentiment=True,
)
```

## Training a New Model

You can also train your own ATESC model using PyABSA. Here’s a simple example to get you started.

### Configuring the Training

First, set up the configuration for your model. You can choose a model architecture, a pre-trained BERT model, and other
hyperparameters.

```python
from pyabsa import AspectTermExtraction as ATEPC
from pyabsa import ModelSaveOption, DeviceTypeOption

# Get a configuration template
config = ATEPC.ATEPCConfigManager.get_atepc_config_english()

# Set the model and BERT checkpoint
config.model = ATEPC.ATEPCModelList.FAST_LCF_ATEPC
config.pretrained_bert = 'microsoft/deberta-v3-base'

# Set training hyperparameters
config.num_epoch = 5
config.evaluate_begin = 2
config.max_seq_len = 80
config.log_step = 100
config.dropout = 0.5
config.learning_rate = 1e-5
config.l2reg = 1e-8
config.cache_dataset = False
config.use_amp = True
config.seed = 42
```

### Starting the Training

Next, choose a dataset and start the training process. PyABSA will handle the data loading, training loop, and
evaluation.

```python
# Choose a dataset
dataset = ATEPC.ATEPCDatasetList.Laptop14

# Start the training
trainer = ATEPC.ATEPCTrainer(
    config=config,
    dataset=dataset,
    checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
    auto_device=DeviceTypeOption.AUTO,
)
```

### Using Your Trained Model

After training, you can load your model from the checkpoint and use it for inference, just like you would with a
pre-trained model.

```python
# The trained model is saved in the "checkpoints" directory
# You can load it by providing the path to the checkpoint
my_extractor = ATEPC.AspectExtractor('checkpoints/fast_lcf_atepc_Laptop14_acc_..._f1_...')

# Now you can use your own model for predictions
my_extractor.predict('The screen is amazing, but the battery life is short.')
```
