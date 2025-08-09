# Installation

This guide will walk you through the process of installing PyABSA. To avoid dependency conflicts, it is highly
recommended to use a virtual environment.

## Prerequisites

- Python 3.8+
- PyTorch 1.10.0+
- Transformers 4.0.0+

PyABSA will automatically install the required versions of PyTorch and Transformers.

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
python -c "from pyabsa import available_checkpoints; print(available_checkpoints())"
```

A successful installation will print a list of available checkpoints.

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
