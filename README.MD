# PyABSA: A Modularized Framework for Reproducible Aspect-based Sentiment Analysis

[![PyPI](https://img.shields.io/pypi/v/pyabsa.svg)](https://pypi.org/project/pyabsa/)
[![Paper](https://img.shields.io/badge/CIKM-2023-blue)](https://dl.acm.org/doi/10.1145/3583780.3614752)
[![Views/Week](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_views_per_week.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)
[![Clones/Week](https://raw.githubusercontent.com/yangheng95/PyABSA/traffic/total_clones_per_week.svg)](https://github.com/yangheng95/PyABSA/tree/traffic#-total-traffic-data-badge)
[![License](https://img.shields.io/pypi/l/pyabsa.svg)](./LICENSE)

**PyABSA** is a **modular, reproducible** framework for **Aspect-based Sentiment Analysis (ABSA)** — from research to
production.
It unifies training/evaluation/inference across ABSA subtasks, ships with ready-to-use checkpoints, and offers dataset
tooling and metric visualization.

- 📄 **Paper**: CIKM 2023 [[ACM DL]](https://dl.acm.org/doi/10.1145/3583780.3614752)
- 📚 **Docs**: https://pyabsa.readthedocs.io/
- 🧪 **Examples**: [`examples-v2/`](./examples-v2)
- 🧰 **Dataset hub**: [ABSADatasets](https://github.com/yangheng95/ABSADatasets)
- 🌐 **Online demos**: see links below

---

## Getting Started

Welcome to PyABSA! This guide will walk you through the initial steps to get you up and running with the framework.

### Prerequisites

Make sure you have Python 3.8 or later installed on your system. You can check your Python version by running:

`python --version`

### Installation

For a straightforward installation, you can use pip:

`pip install -U pyabsa`

This command installs the core components of PyABSA. For more advanced features like text augmentation and
visualization, you may need to install additional dependencies.

### Your First Code

After installation, you can start using PyABSA with just a few lines of code. Here’s a simple example to get you
started:

```python3
from pyabsa import AspectTermExtraction as ATEPC
```

```python3
# Initialize the aspect extractor
aspect_extractor = ATEPC.AspectExtractor('multilingual', auto_device=True)
```


```python3
# Perform aspect extraction on a sample sentence
result = aspect_extractor.predict(
    ['I love this movie, it is so great!'],
    save_result=True,
    print_result=True
)
```

---

## Features at a Glance

- **Unified API** for training / evaluation / inference across ABSA tasks
- **Model Zoo** with `available_checkpoints()` and auto-download
- **Visualization** for evaluation metrics
- **Human-in-the-loop** dataset annotation helpers
- **Text augmentation** for classification & adversarial defense
- Automatic device selection; simple CPU/GPU switching

> See the [Introduction](https://pyabsa.readthedocs.io/en/latest/0_intro/introduction.html#features) for the full
> feature list.

---

## Supported Tasks

| Task                                                         | What it does                                                | Python API entry                                                  | Demo                                                                                                    |
|--------------------------------------------------------------|-------------------------------------------------------------|-------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| **APC** (Aspect Polarity Classification)                     | Classify sentiment for a given aspect                       | `pyabsa.AspectPolarityClassification`                             | [Multilingual APC (HF Space)](https://huggingface.co/spaces/yangheng/PyABSA-APC)                        |
| **ATEPC** (Aspect Term Extraction & Polarity Classification) | Extract aspect terms and their sentiment                    | `pyabsa.AspectTermExtraction`                                     | [ATEPC (HF Space)](Gradio-Blocks/Multilingual-Aspect-Based-Sentiment-Analysis)                          |
| **ASTE** (Aspect Sentiment Triplet Extraction)               | Extract *(aspect, opinion, sentiment)* triplets             | `pyabsa.AspectSentimentTripletExtraction`                         | [Triplet Extraction (HF Space)](Gradio-Blocks/Multilingual-Aspect-Based-Sentiment-Analysis)             |
| **ASQP / ACOS**                                              | Extract *(aspect, category, opinion, sentiment)* quadruples | `pyabsa.AspectCategoryOpinionSentimentTripletExtraction`          | [Quadruple Extraction (HF Space)](Gradio-Blocks/Multilingual-Aspect-Based-Sentiment-Analysis)           |
| **Others**                                                   | Text classification, adversarial defense, etc.              | `pyabsa.TextClassification`, `pyabsa.TextAdversarialDefense`, ... | –                                                                                                       |

> Full list and
>
tutorials: [Supported Tasks](https://pyabsa.readthedocs.io/en/latest/0_intro/introduction.html) · [Tutorials](https://pyabsa.readthedocs.io/en/latest/3_tutorials/index.html)

---

## Installation

PyPI (recommended):

```bash
pip install -U pyabsa
```

From source (latest mainline):

```bash
git clone https://github.com/yangheng95/PyABSA --depth=1
cd PyABSA
python setup.py install
```

**Requirements**: Python **>= 3.8**; PyTorch and Transformers will be installed as dependencies. For advanced/optional
dependencies (augmentation, visualization, demos), see
the [Installation guide](https://pyabsa.readthedocs.io/en/latest/1_installation/index.html).

---

## Quickstart

### 1) Extract aspect terms and classify their sentiments (ATEPC)

```python
from pyabsa import AspectTermExtraction as ATEPC, available_checkpoints

# View available checkpoints (local + remote)
print(available_checkpoints())

aspect_extractor = ATEPC.AspectExtractor(
    'multilingual',
    auto_device=True,  # False -> force CPU
    cal_perplexity=True
)

# Single instance
aspect_extractor.predict(
    ['I love this movie, it is so great!'],
    save_result=True,
    print_result=True,
    ignore_error=True
)

# Batch inference from a built-in dataset
inference_source = ATEPC.ATEPCDatasetList.Restaurant16
result = aspect_extractor.batch_predict(
    target_file=inference_source,
    save_result=True,
    print_result=True,
    pred_sentiment=True
)
print(result)
```

### 2) Aspect-based sentiment classification (APC)

```python
from pyabsa import AspectPolarityClassification as APC, available_checkpoints

print(available_checkpoints(show_ckpts=True))

classifier = APC.SentimentClassifier(
    'multilingual',
    auto_device=True,
    cal_perplexity=True
)

classifier.predict(
    ['I love this movie, it is so great!'],
    save_result=True,
    print_result=True,
    ignore_error=True
)

inference_source = APC.APCDatasetList.Laptop14
apc_result = classifier.batch_predict(
    target_file=inference_source,
    save_result=True,
    print_result=True,
    pred_sentiment=True
)
print(apc_result)
```

> More examples (training, evaluation, visualization, deployment): see [`examples-v2/`](./examples-v2)
> and [Tutorials](https://pyabsa.readthedocs.io/en/latest/3_tutorials/index.html).

---

## Model Zoo & Checkpoints

- List all available checkpoints:
  ```python
  from pyabsa import available_checkpoints
  print(available_checkpoints())
  ```
- PyABSA resolves checkpoints across local/remote registries and auto-downloads when needed. See the docs for *
  *CheckpointManager** and **API Reference**.

---

## Datasets

- Public & community-contributed datasets: **[ABSADatasets](https://github.com/yangheng95/ABSADatasets)**
- To prepare your own datasets (format, semi-automatic annotation, naming conventions), see **Integrated Datasets** and
  **Notice** in the docs.
- You can also use built-in dataset enums (e.g., `APC.APCDatasetList.Laptop14`, `ATEPC.ATEPCDatasetList.Restaurant16`)
  to run quick experiments.

---

## Documentation

- Overview & features: [Introduction](https://pyabsa.readthedocs.io/en/latest/0_intro/introduction.html)
- Installation & optional components: [Installation](https://pyabsa.readthedocs.io/en/latest/1_installation/index.html)
- Task tutorials (train/infer/deploy): [Tutorials](https://pyabsa.readthedocs.io/en/latest/3_tutorials/index.html)
- API Reference: [API Reference](https://pyabsa.readthedocs.io/en/latest/4_api_reference/index.html)

---

## Roadmap (indicative)

- Python 3.13 compatibility verification and wheels
- Extended dataset templates & validators
- Streamlined model registry and checkpoint metadata
- Better Hugging Face integration (Spaces & model cards)
- Optional plugins: advanced augmentation, evaluation dashboards

> Have a suggestion? Please open a GitHub Discussion or Issue.

---

## Known Limitations

- `v2` introduced breaking API changes; older scripts may need updates.
- Some checkpoints require a one-time download at first use.
- GPU is optional but recommended for training and large-scale inference.
- Certain advanced features have extra dependencies; see the Installation guide.

---

## Citation

If you use PyABSA in your research or products, please cite:

**CIKM 2023**

```bibtex
@inproceedings{YangZL23,
    author = {Heng Yang and Chen Zhang and Ke Li},
    title = {PyABSA: A Modularized Framework for Reproducible Aspect-based Sentiment Analysis},
    booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM 2023)},
    pages = {5117--5122},
    year = {2023},
    doi = {10.1145/3583780.3614752}
}
```

**arXiv 2022 (optional)**

```bibtex
@article{YangL22,
    author = {Heng Yang and Ke Li},
    title = {PyABSA: Open Framework for Aspect-based Sentiment Analysis},
    journal = {CoRR},
    volume = {abs/2208.01368},
    year = {2022},
    doi = {10.48550/arXiv.2208.01368}
}
```

---

## Contributing

Contributions are welcome! You can:

- Share custom datasets via [ABSADatasets](https://github.com/yangheng95/ABSADatasets)
- Integrate your models (with or without PyABSA base—we can help adapt)
- Report bugs, improve messages & docs, or add example scripts
- Propose features or refactors

**Guidelines**

- Use Python 3.8+; please run at least one GPU and one CPU pass for examples before submitting.
- Keep changes reproducible (seeds, configs) and scoped.
- In PR description, summarize motivation and impact.

---

## Community and Support

Join our community to stay updated, ask questions, and contribute to the project.

- **GitHub Discussions**: For questions, feature requests, and discussions.
- **Issue Tracker**: To report bugs and track issues.
- **Contributing**: We welcome contributions! Please see our contributing guidelines for more details.

---

## License

MIT License © PyABSA contributors

---


