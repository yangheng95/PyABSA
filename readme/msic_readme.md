# PyABSA - Open Framework for Aspect-based Sentiment Analysis

![PyPI - Python Version](https://img.shields.io/badge/python-3.6-blue.svg)
[![PyPI](https://img.shields.io/pypi/v/pyabsa)](https://pypi.org/project/pyabsa/)
[![PyPI_downloads](https://img.shields.io/pypi/dm/pyabsa)](https://pypi.org/project/pyabsa/)
![License](https://img.shields.io/pypi/l/pyabsa?logo=PyABSA)

[![total views](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_views.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)
[![total views per week](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_views_per_week.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)
[![total clones](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_clones.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)
[![total clones per week](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_clones_per_week.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/back-to-reality-leveraging-pattern-driven/aspect-based-sentiment-analysis-on-semeval)](https://paperswithcode.com/sota/aspect-based-sentiment-analysis-on-semeval?p=back-to-reality-leveraging-pattern-driven)

**Hi, there!** Please star this repo if it helps you! Each Star helps PyABSA go further, many thanks.

# | [Overview](../README.MD) | [HuggingfaceHub](huggingface_readme.md) | [ABDADatasets](dataset_readme.md) | [ABSA Models](model_readme.md) | [Colab Tutorials](tutorial_readme.md) |

## For Syntax-Parsing Models

The default SpaCy english model is en_core_web_sm, if you didn't install it, PyABSA will download/install it
automatically.

If you would like to change english model (or other pre-defined options), you can get/set as following:

```python3
from pyabsa.framework.configuration_class.apc_config_manager import APCConfigManager
from pyabsa.framework.configuration_class.atepc_config_manager import ATEPCConfigManager
from pyabsa.framework.configuration_class.classification_config_manager import ClassificationConfigManager

# Set
APCConfigManager.set_apc_config_english({'spacy_model': 'en_core_web_lg'})
ATEPCConfigManager.set_atepc_config_english({'spacy_model': 'en_core_web_lg'})
ClassificationConfigManager.set_classification_config_english({'spacy_model': 'en_core_web_lg'})

# Get
APCConfigManager.get_apc_config_english()
ATEPCConfigManager.get_atepc_config_english()
ClassificationConfigManager.get_classification_config_english()

# Manually Set spaCy nlp Language object
from pyabsa.core.apc.dataset_utils.apc_utils import configure_spacy_model

nlp = configure_spacy_model(APCConfigManager.get_apc_config_english())
```
