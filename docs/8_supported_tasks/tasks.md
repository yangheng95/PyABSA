# Supported Tasks

PyABSA is a comprehensive framework that supports a wide range of natural language processing tasks, with a primary
focus on aspect-based sentiment analysis. This section provides an overview of all supported tasks and their
applications.

## Core ABSA Tasks

### Aspect-Based Sentiment Classification (ABSC)

**What it does**: Determines the sentiment polarity (positive, negative, neutral) towards specific aspects mentioned in
text.

**Use cases**:

- Product review analysis
- Customer feedback processing
- Social media sentiment monitoring

**Example**: "The [B-ASP]food[E-ASP] was excellent, but the [B-ASP]service[E-ASP] was slow."

- Aspect: "food" → Sentiment: Positive
- Aspect: "service" → Sentiment: Negative

[→ See ABSC Examples](https://github.com/yangheng95/PyABSA/tree/v2/examples-v2/aspect_polarity_classification)

### Aspect Term Extraction (ATE)

**What it does**: Automatically identifies and extracts aspect terms from text without requiring pre-marked aspects.

**Use cases**:

- Automated opinion mining
- Market research
- Content analysis

**Example**: "The pizza was delicious and the atmosphere was cozy."

- Extracted aspects: "pizza", "atmosphere"

[→ See ATE Examples](https://github.com/yangheng95/PyABSA/tree/v2/examples-v2/aspect_term_extraction)

### Aspect Sentiment Triplet Extraction (ASTE)

**What it does**: Simultaneously extracts aspect terms, opinion terms, and their sentiment relationships in a single
model.

**Use cases**:

- Comprehensive opinion analysis
- Fine-grained sentiment understanding
- Advanced text analytics

**Example**: "The pizza was delicious but the service was terrible."

- Triplet 1: (pizza, delicious, positive)
- Triplet 2: (service, terrible, negative)

[→ See ASTE Examples](https://github.com/yangheng95/PyABSA/tree/v2/examples-v2/aspect_sentiment_triplet_extration/Aspect_Sentiment_Triplet_Extraction.ipynb)

## Extended Tasks

### Text Classification

**What it does**: General-purpose text classification for various domains and use cases.

**Use cases**:

- Document categorization
- Topic classification
- Intent detection

[→ See Text Classification Examples](https://github.com/yangheng95/PyABSA/tree/v2/examples-v2/text_classification)

### Textual Adversarial Defense

**What it does**: Protects models from adversarial attacks that attempt to fool sentiment classifiers with subtle text
modifications.

**Use cases**:

- Robust model deployment
- Security-critical applications
- Model reliability enhancement

[→ See Adversarial Defense Examples](https://github.com/yangheng95/PyABSA/tree/v2/examples-v2/text_adversarial_defense)

## Specialized Domains

### RNA Sequence Classification

**What it does**: Classifies RNA sequences for biological research applications.

**Use cases**:

- Bioinformatics research
- Genomic analysis
- Functional annotation

[→ See RNA Classification Examples](https://github.com/yangheng95/PyABSA/tree/v2/examples-v2/rna_sequence_classification)

### RNA Sequence Regression

**What it does**: Predicts continuous values for RNA sequences, such as expression levels or binding affinities.

**Use cases**:

- Gene expression prediction
- Protein-RNA interaction analysis
- Regulatory element scoring

[→ See RNA Regression Examples](https://github.com/yangheng95/PyABSA/tree/v2/examples-v2/rna_sequence_regression)

## Getting Started with Tasks

Each task in PyABSA follows a similar workflow:

1. **Load or configure a model** using the task-specific API
2. **Prepare your data** in the required format
3. **Train** (optional) or **load a pre-trained model**
4. **Run inference** on your data
5. **Evaluate results** and visualize metrics

For detailed examples and code snippets, click on the links above or refer to the tutorials section in this
documentation.

## Model Architecture Support

PyABSA supports various state-of-the-art model architectures for each task:

- **Transformer-based models**: BERT, RoBERTa, DeBERTa, and more
- **Task-specific architectures**: FAST-LSA, LCF-BERT, LCFS-BERT
- **Custom models**: Easy integration of your own architectures

## Performance and Benchmarks

PyABSA models are evaluated on standard benchmark datasets and consistently achieve competitive or state-of-the-art
performance. For detailed benchmark results, please refer to our research papers and the performance metrics in
individual task examples.
