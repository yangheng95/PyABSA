# Adversarial Defense for Text Classification

This guide provides a quick introduction to using PyABSA for adversarial defense in text classification. You'll learn
how to use pre-trained models to detect and defend against adversarial attacks.

## Inference with Adversarial Defense

PyABSA's `TextAdversarialDefense` module can classify a piece of text while also detecting and defending against
potential adversarial attacks. Here’s how to get started.

### Loading a Classifier

First, import the necessary components and load a pre-trained text classifier with adversarial defense capabilities.

```python
from pyabsa import TextAdversarialDefense as TAD

# Load a pre-trained classifier
classifier = TAD.TADTextClassifier('tad-sst2')
```

### Running Predictions with Defense

Once the classifier is loaded, you can use it to predict the label of a sentence. To enable adversarial defense, you can
specify a defense method.

```python
# An example of an adversarial text
adversarial_text = 'The movie is not good. $LABEL$ 1'

# Predict the label with PWWS defense enabled
result = classifier.predict(adversarial_text, defense='pwws')

# The result will indicate if the text was identified as adversarial
# and if the prediction was repaired.
print(result)
```

### Understanding the Output

The output from the `predict` method contains several key pieces of information:

- `label`: The predicted label for the text.
- `confidence`: The confidence score for the prediction.
- `is_adv_label`: Indicates whether the input text was detected as adversarial.
- `is_fixed`: Indicates whether the model was able to repair the prediction.
- `restored_text`: The version of the text after the defense mechanism is applied.

## Generating Adversarial Examples

To test the defense mechanisms, you may need to generate adversarial examples. PyABSA is compatible with libraries like
`textattack` for this purpose. Here is a simplified example of how you might set up an attacker.

### Setting up an Attacker

You can use `textattack` to generate adversarial examples for a given model. First, you'll need to wrap your PyABSA
model.

```python
from textattack import Attacker
from textattack.attack_recipes import PWWSRen2019
from textattack.models.wrappers import HuggingFaceModelWrapper


class ModelWrapper(HuggingFaceModelWrapper):
    def __init__(self, model):
        self.model = model

    def __call__(self, text_inputs, **kwargs):
        outputs = [self.model.infer(text, print_result=False, **kwargs)['probs'] for text in text_inputs]
        return outputs


# Wrap your classifier
model_wrapper = ModelWrapper(classifier)

# Create an attacker
attacker = Attacker(PWWSRen2019.build(model_wrapper))
```

### Attacking a Sentence

Once the attacker is set up, you can use it to generate an adversarial version of a sentence.

```python
# Original text and label
text = "The movie is fantastic."
label = 1

# Generate an adversarial example
attack_result = attacker.attack(text, label)

if hasattr(attack_result, 'perturbed_text'):
    adversarial_text = attack_result.perturbed_text()
    print(f"Adversarial Text: {adversarial_text}")
else:
    print("Attack failed or text is not adversarial.")

```

By following these steps, you can explore and utilize the adversarial defense capabilities of PyABSA. For more advanced
use cases, please refer to the detailed tutorials in the documentation.
