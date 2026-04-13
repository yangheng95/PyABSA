# Installation

This guide will walk you through the process of installing PyABSA. To avoid dependency conflicts, it is highly
recommended to use a virtual environment.

## Prerequisites

- Python 3.10+
- PyTorch 2.0.0+
- Transformers 4.44.0 – 5.x
- huggingface_hub 0.23.0+

PyABSA will automatically install the required versions of PyTorch, Transformers and
huggingface_hub. The Hub client is required because checkpoints and datasets are
distributed via native Hugging Face Model / Dataset repos since 2.5.0.

## Setting up a Virtual Environment

A virtual environment is a self-contained directory that holds a specific Python interpreter and its own set of
libraries. This is the recommended way to manage project dependencies.

### macOS / Linux

```bash
python3 -m venv pyabsa-env
source pyabsa-env/bin/activate
```

### Windows

```bash
python -m venv pyabsa-env
.\pyabsa-env\Scripts\activate
```

Once activated, your terminal prompt will be prefixed with `(pyabsa-env)`.

## Installing PyABSA

You can install PyABSA either from the Python Package Index (PyPI) or from the source code on GitHub.

### From PyPI

For the latest stable version, use `pip`:

```bash
pip install pyabsa -U
```

### From Source

If you need the latest features or want to contribute to the project, you can install from the source:

```bash
git clone https://github.com/yangheng95/PyABSA.git
cd PyABSA
pip install -e .
```

The `-e` flag installs the package in "editable" mode, which means that any changes you make to the source code will be
immediately available in your environment.

## Verifying the Installation

To make sure PyABSA is installed correctly, run the following command:

```bash
python -c "from pyabsa import available_checkpoints, TaskCodeOption; \
           print(available_checkpoints(TaskCodeOption.Aspect_Polarity_Classification))"
```

A successful installation will fetch the checkpoint index from the
Hugging Face Hub (`yangheng/pyabsa-index` by default; override via the
`PYABSA_INDEX_REPO` env var) and print a dict of available checkpoint
entries.

## Publishing checkpoints / datasets

PyABSA ships three console scripts that wrap the official `huggingface_hub`
SDK so contributors can push artefacts to the Hub without any custom
hosting:

```bash
huggingface-cli login                 # one-time auth (or set HF_TOKEN)

pyabsa-upload-ckpt  ./local_ckpt --repo <user>/pyabsa-<task>-<name>
pyabsa-upload-data  ./local_data --repo <user>/pyabsa-<task>-<dataset>
pyabsa-index-update ./checkpoints.json    # publish a schema-v2 index
```

## Optional Dependencies

Some PyABSA features require additional packages.

### Text Augmentation

To use the text augmentation features, you need to install `textaugment`:

```bash
pip install textaugment
```

### Metric Visualization

For visualizing metrics, you will need `matplotlib` and `seaborn`:

```bash
pip install matplotlib seaborn
```

You are now ready to start using PyABSA for your Aspect-Based Sentiment Analysis projects!
