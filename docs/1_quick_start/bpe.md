# Training Custom Tokenizers and Word Embeddings

For specialized domains or languages that are not well-represented by standard tokenizers, PyABSA allows you to train
your own BPE (Byte-Pair Encoding) tokenizer and Word2Vec embeddings. This can significantly improve model performance on
your specific data.

## Training a BPE Tokenizer

A custom BPE tokenizer can learn the subword vocabulary of your corpus, which is especially useful for languages without
clear word boundaries or for domains with unique jargon (e.g., bioinformatics, legal documents).

### Example

Here’s how you can train a BPE tokenizer on a collection of text files.

```python
from pyabsa.utils import train_bpe_tokenizer
import findfile

# 1. Find all your text files
# Replace '.txt' with the extension of your corpus files
corpus_files = findfile.find_cwd_files('.txt')

# 2. Train the tokenizer
# The trained tokenizer files will be saved to the 'bpe_tokenizer' directory
train_bpe_tokenizer(
    paths=corpus_files,
    save_path='bpe_tokenizer',
    base_tokenizer='roberta-base'  # You can choose a base tokenizer to build upon
)
```

After running this script, you can load your custom tokenizer using
`transformers.AutoTokenizer.from_pretrained('bpe_tokenizer')`.

## Training Word2Vec Embeddings

Once you have a custom tokenizer, you can use it to train Word2Vec embeddings. These embeddings will capture the
semantic relationships between the tokens in your specific corpus.

### Example

Here’s how to train Word2Vec embeddings using the BPE tokenizer you created above.

```python
from pyabsa.utils import train_word2vec
from transformers import AutoTokenizer
import findfile

# 1. Find all your text files
corpus_files = findfile.find_cwd_files('.txt')

# 2. Load your custom BPE tokenizer
pre_tokenizer = AutoTokenizer.from_pretrained('bpe_tokenizer')

# 3. Train the Word2Vec embeddings
# The trained embedding file will be saved as 'word2vec.vec'
train_word2vec(
    paths=corpus_files,
    save_path='word2vec',
    pre_tokenizer=pre_tokenizer
)
```

You can then use these custom embeddings in your models by setting the `glove_or_word2vec_path` in your configuration to
the path of the generated `.vec` file. This is particularly useful for non-Transformer-based models in PyABSA.
