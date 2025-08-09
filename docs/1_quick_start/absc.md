# Quickstart: Aspect-Based Sentiment Classification (ABSC)

This guide provides a hands-on introduction to Aspect-Based Sentiment Classification (ABSC) with PyABSA. You'll learn
how to perform inference with pre-trained models and how to train your own custom models.

ABSC is the task of identifying the sentiment polarity (e.g., positive, negative, or neutral) towards a specific aspect
mentioned in the text.

---

## Inference with a Pre-trained Model

PyABSA provides pre-trained models that you can use for inference right away.

### 1. Load the Classifier

First, import `AspectPolarityClassification` and load a pre-trained sentiment classifier. PyABSA will automatically
download the model checkpoint if it's not found locally.

```python
from pyabsa import AspectPolarityClassification as APC

# Load a pre-trained multilingual model
# The first time you run this, it will download the model checkpoint
sentiment_classifier = APC.SentimentClassifier("multilingual")
```

### 2. Prepare Your Data

To get a prediction, you need to specify the aspect in your text. The aspect should be enclosed within `[B-ASP]` (
beginning of aspect) and `[E-ASP]` (end of aspect) tags.

```python
# Define a single sentence for prediction
text = "The [B-ASP]food[E-ASP] was great, but the [B-ASP]service[E-ASP] was slow."

# You can also prepare a list of sentences for batch prediction
examples = [
    "The [B-ASP]pizza[E-ASP] is delicious, but the [B-ASP]staff[E-ASP] is rude.",
    "I love the [B-ASP]location[E-ASP], but the [B-ASP]price[E-ASP] is too high."
]
```

### 3. Run Predictions

Use the `predict` method to get the sentiment for the specified aspects.

```python
# Run prediction on the single sentence
result = sentiment_classifier.predict(text)
# Expected output will show sentiment for 'food' and 'service'

# Run prediction on the list of sentences
results = sentiment_classifier.predict(examples)
# Expected output will be a list of predictions for each sentence
```

### 4. Batch Prediction on a Dataset File

For larger datasets, you can point the classifier to a file. PyABSA includes several standard datasets you can use for
this.

```python
# Select a built-in dataset for batch prediction
# You can replace this with the path to your own dataset file
inference_set = APC.APCDatasetList.Restaurant14

# Run batch prediction
# The results will be saved to a file and printed to the console
batch_results = sentiment_classifier.batch_predict(
    target_file=inference_set,
    print_result=True,
    save_result=True,
    ignore_error=True,
)
```

---

## Training a Custom Model

Training your own ABSC model is straightforward with PyABSA.

### 1. Configure the Training

Start by creating a configuration object. This allows you to define the model architecture, pre-trained backbone, and
various hyperparameters.

```python
from pyabsa import AspectPolarityClassification as APC
from pyabsa import ModelSaveOption, DeviceTypeOption

# Get a pre-defined configuration template for English
config = APC.APCConfigManager.get_apc_config_english()

# Choose a model from the available list
config.model = APC.APCModelList.FAST_LSA_T_V2

# Specify the pre-trained BERT model to use
config.pretrained_bert = 'microsoft/deberta-v3-base'

# Set key training hyperparameters
config.num_epoch = 5
config.evaluate_begin = 2  # Start evaluation after 2 epochs
config.max_seq_len = 80
config.log_step = 100
config.dropout = 0.5
config.learning_rate = 1e-5
config.l2reg = 1e-8
config.cache_dataset = False  # Cache dataset in memory for faster training
config.use_amp = True  # Use Automatic Mixed Precision for faster training
config.seed = 42
```

### 2. Choose a Dataset

PyABSA provides access to standard benchmark datasets. You can also use your own custom dataset (see documentation on
using custom datasets).

```python
# Select a dataset for training
# You can also provide a path to your own dataset
dataset = APC.APCDatasetList.Laptop14
```

### 3. Start the Training

Finally, pass your configuration and dataset to the `APCTrainer` to start the training process.

```python
# Create a trainer
trainer = APC.APCTrainer(
    config=config,
    dataset=dataset,
    # from_checkpoint="path/to/your/checkpoint" # Optionally load a checkpoint to resume training
    auto_device=DeviceTypeOption.AUTO, # Automatically choose GPU if available
    checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT, # Save model state_dict only
    load_aug=False # Do not load augmented data
)

# The training process will start here
# The best model checkpoint will be saved automatically
```

After training is complete, you can find the trained model in the `checkpoints` directory and load it for inference as
shown in the first section.
