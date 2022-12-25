BPE Tokenizer and Word2Vec Training
======================
## BPE Tokenizer Training
If your text cannot be tokenized by the whitespace tokenizer, you can train a BPE tokenizer by yourself.
```python3
import findfile
from transformers import AutoTokenizer

from pyabsa.utils import train_word2vec, train_bpe_tokenizer

if __name__ == '__main__':
    """
    This script is used to train word2vec and bpe tokenizer for rna/protein classification/regression tasks.
    For example:
    MQFKVYTYKRESRYRLFCDVQSDIIDTPGRRMVIPLASARLLSDKVSRELYPVVHIGDESWRMMTTDMASVPVSVIGEEVADLSHRENDIKNAINLMFWGI
    -> Tokenize
    MQFK VYTYKR ESRY RLFCDV QSDIIDT PGRRM VIP LASARLLSD KVSRELYPV VHIGDESW RMMTTDM ASVPV SVIGEE VADLSH RENDI KNAIN LMFWGI
    -> Word2Vec Embedding
    [1*768] or [1*300]
    This is a not a real protein sequence, just for example.
    """
    paths = findfile.find_cwd_files('.txt')

    # train bpe tokenizer for protein or rna sequence
    train_bpe_tokenizer(paths, save_path='bpe_tokenizer', base_tokenizer='roberta-base')

    # then you can use the bpe_tokenizer to train a protein or rna sequence word2vec embedding
    pre_tokenizer = AutoTokenizer.from_pretrained('bpe_tokenizer')
    train_word2vec(paths, save_path='word2vec', pre_tokenizer=pre_tokenizer)
```
