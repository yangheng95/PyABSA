# -*- coding: utf-8 -*-
# file: tokenizer_class.py
# time: 03/11/2022 21:44
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import os
import pickle
from typing import Union, List

import numpy as np
import tqdm
from numpy import ndarray
from termcolor import colored
from transformers import AutoTokenizer

from pyabsa.utils.file_utils.file_utils import prepare_glove840_embedding
from pyabsa.utils.pyabsa_utils import fprint


class Tokenizer(object):
    def __init__(self, config):
        # Constructor for Tokenizer class
        self.config = config
        self.max_seq_len = self.config.max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1
        self.pre_tokenizer = None
        self.pad_token_id = 0
        self.unk_token_id = 0
        self.cls_token_id = 0
        self.sep_token_id = 0
        self.mask_token_id = 0

    @staticmethod
    def build_tokenizer(config, cache_path=None, pre_tokenizer=None, **kwargs):
        # Build the tokenizer from a given config file
        Tokenizer.pre_tokenizer = pre_tokenizer
        dataset_name = os.path.basename(config.dataset_name)
        if not os.path.exists("run/{}".format(dataset_name)):
            os.makedirs("run/{}".format(dataset_name))
        tokenizer_path = "run/{}/{}".format(dataset_name, cache_path)
        if cache_path and os.path.exists(tokenizer_path) and not config.overwrite_cache:
            config.logger.info("Loading tokenizer on {}".format(tokenizer_path))
            tokenizer = pickle.load(open(tokenizer_path, "rb"))
        else:
            words = set()
            if hasattr(config, "dataset_file"):
                config.logger.info(
                    "Building tokenizer for {} on {}".format(
                        config.dataset_file, tokenizer_path
                    )
                )
                for dataset_type in config.dataset_file:
                    for file in config.dataset_file[dataset_type]:
                        # Open the file and tokenize each line
                        fin = open(
                            file, "r", encoding="utf-8", newline="\n", errors="ignore"
                        )
                        lines = fin.readlines()
                        fin.close()
                        for i in range(0, len(lines)):
                            # Tokenize the line using the pre tokenizer or split by spaces
                            if pre_tokenizer:
                                words.update(pre_tokenizer.tokenize(lines[i].strip()))
                            else:
                                words.update(lines[i].strip().split())
            elif hasattr(config, "dataset_dict"):
                config.logger.info(
                    "Building tokenizer for {} on {}".format(
                        config.dataset_name, tokenizer_path
                    )
                )
                for dataset_type in ["train", "test", "valid"]:
                    for i, data in enumerate(config.dataset_dict[dataset_type]):
                        # Tokenize each sample in the data
                        if pre_tokenizer:
                            words.update(pre_tokenizer.tokenize(data["data"]))
                        else:
                            words.update(data["data"].split())
            tokenizer = Tokenizer(config)
            tokenizer.pre_tokenizer = pre_tokenizer
            tokenizer.fit_on_text(list(words))
            # Cache the tokenizer if required
            if config.cache_dataset:
                pickle.dump(tokenizer, open(tokenizer_path, "wb"))

        return tokenizer

    def fit_on_text(self, text: Union[str, List[str]], **kwargs):
        # Tokenize the given text and fit it to the tokenizer
        if isinstance(text, str):
            if self.pre_tokenizer:
                words = self.pre_tokenizer.tokenize(text)
            else:
                words = text.split()
            for word in words:
                if self.config.do_lower_case:
                    word = word.lower()
                if word not in self.word2idx:
                    self.word2idx[word] = self.idx
                    self.idx2word[self.idx] = word
                    self.idx += 1
        elif isinstance(text, list):
            for t in text:
                self.fit_on_text(t)
        else:
            raise ValueError("Text must be a string or a list of strings.")

    def text_to_sequence(
        self, text: Union[str, List[str]], padding="max_length", **kwargs
    ):
        """
        Convert input text to a sequence of token IDs.

        Parameters:
        - `text` : str or list of str
            Input text to be converted to a sequence of token IDs.
        - `padding` : str, optional (default="max_length")
            Padding method to use when the sequence is shorter than the `max_seq_len` parameter.
        - `**kwargs`:
            Additional arguments that can be passed, such as `reverse`.

        Returns:
        - `sequence`: list of int or list of list of int
            Sequence of token IDs or list of sequences of token IDs, depending on whether the input text is a string or a list of strings.
        """
        if isinstance(text, str):
            if self.config.do_lower_case:
                text = text.lower()
            if self.pre_tokenizer:
                words = self.pre_tokenizer.tokenize(text)
            else:
                words = text.split()
            sequence = [self.word2idx[w] if w in self.word2idx else 0 for w in words]
            if len(sequence) == 0:
                sequence = [0]
            if kwargs.get("reverse", False):
                sequence = sequence[::-1]
            if padding == "max_length":
                return pad_and_truncate(sequence, self.max_seq_len, self.pad_token_id)
            else:
                return sequence

        elif isinstance(text, list):
            sequences = []
            for t in text:
                sequences.append(self.text_to_sequence(t, **kwargs))
            return sequences
        else:
            raise ValueError("text_to_sequence only support str or list of str")

    def sequence_to_text(self, sequence):
        """
        Convert a sequence of token IDs to text.

        Parameters:
        - `sequence` : list of int
            Sequence of token IDs to be converted to text.

        """
        # Convert a sequence of token IDs to text
        words = [
            self.idx2word[idx] if idx in self.idx2word else "<unk>" for idx in sequence
        ]
        # Join the words to form a sentence
        return " ".join(words)


class PretrainedTokenizer:
    def __init__(self, config, **kwargs):
        """
        Constructor for PretrainedTokenizer class
            Args:
            - config: A configuration object that includes parameters for the tokenizer
            - **kwargs: Other keyword arguments to be passed to the AutoTokenizer class

            Returns:
            - None
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_bert, **kwargs)
        self.max_seq_len = self.config.max_seq_len
        self.pad_token_id = self.tokenizer.pad_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.eos_token_id = (
            self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id
            else self.tokenizer.sep_token_id
        )

        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.mask_token = self.tokenizer.mask_token
        self.eos_token = (
            self.tokenizer.eos_token
            if self.tokenizer.eos_token
            else self.tokenizer.sep_token
        )

    def text_to_sequence(self, text, **kwargs):
        return self.tokenizer.encode(
            text,
            truncation=kwargs.pop("truncation", True),
            padding=kwargs.pop("padding", "max_length"),
            max_length=kwargs.pop("max_length", self.max_seq_len),
            return_tensors=kwargs.pop("return_tensors", None),
            **kwargs
        )

    def text_to_sequence(self, text, **kwargs):
        """
        Encodes the given text into a sequence of token IDs.

        Args:
            text (str): Text to be encoded.
            **kwargs: Additional arguments to be passed to the tokenizer.

        Returns:
            torch.Tensor: Encoded sequence of token IDs.
        """
        return self.tokenizer.encode(
            text,
            truncation=kwargs.pop("truncation", True),
            padding=kwargs.pop("padding", "max_length"),
            max_length=kwargs.pop("max_length", self.max_seq_len),
            return_tensors=kwargs.pop("return_tensors", None),
            **kwargs
        )

    def sequence_to_text(self, sequence, **kwargs):
        """
        Decodes the given sequence of token IDs into text.

        Args:
            sequence (list): Sequence of token IDs.
            **kwargs: Additional arguments to be passed to the tokenizer.

        Returns:
            str: Decoded text.
        """
        return self.tokenizer.decode(sequence, **kwargs)

    def tokenize(self, text, **kwargs):
        """
        Tokenizes the given text into subwords.

        Args:
            text (str): Text to be tokenized.
            **kwargs: Additional arguments to be passed to the tokenizer.

        Returns:
            list: List of subwords.
        """
        return self.tokenizer.tokenize(text, **kwargs)

    def convert_tokens_to_ids(self, return_tensors=None, **kwargs):
        """
        Converts the given tokens into token IDs.

        Args:
            return_tensors (str): Type of tensor to be returned.

        Returns:
            list or torch.Tensor: List or tensor of token IDs.
        """
        return self.tokenizer.convert_tokens_to_ids(return_tensors, **kwargs)

    def convert_ids_to_tokens(self, ids, **kwargs):
        """
        Converts the given token IDs into tokens.

        Args:
            ids (list): List of token IDs.

        Returns:
            list: List of tokens.
        """
        return self.tokenizer.convert_ids_to_tokens(ids, **kwargs)

    def encode_plus(self, text, **kwargs):
        """
        Encodes the given text into a sequence of token IDs along with additional information.

        Args:
            text (str): Text to be encoded.
            **kwargs: Additional arguments to be passed to the tokenizer.
        """
        return self.tokenizer.encode_plus(
            text,
            truncation=kwargs.pop("truncation", True),
            padding=kwargs.pop("padding", "max_length"),
            max_length=kwargs.pop("max_length", self.max_seq_len),
            return_tensors=kwargs.pop("return_tensors", None),
            **kwargs
        )

    def encode(self, text, **kwargs):
        """
        Encodes the given text into a sequence of token IDs.

        Args:
            text (str): Text to be encoded.
            **kwargs: Additional arguments to be passed to the tokenizer.

        Returns:
            torch.Tensor: Encoded sequence of token IDs.
        """
        return self.tokenizer.encode(
            text,
            truncation=kwargs.pop("truncation", True),
            padding=kwargs.pop("padding", "max_length"),
            max_length=kwargs.pop("max_length", self.max_seq_len),
            return_tensors=kwargs.pop("return_tensors", None),
            **kwargs
        )

    def decode(self, sequence, **kwargs):
        # Decode the given sequence to its corresponding text using the tokenizer
        return self.tokenizer.decode(sequence, **kwargs)


def build_embedding_matrix(config, tokenizer, cache_path=None):
    """
    Function to build an embedding matrix for a given tokenizer and config.

    Args:
    - config: A configuration object.
    - tokenizer: A tokenizer object.
    - cache_path: A string that specifies the cache path.

    Returns:
    - embedding_matrix: A numpy array of shape (len(tokenizer.word2idx)+1, config.embed_dim)
                        containing the embedding matrix for the given tokenizer and config.

    """
    if not os.path.exists("run/{}".format(config.dataset_name)):
        os.makedirs("run/{}".format(config.dataset_name))
    embed_matrix_path = "run/{}".format(os.path.join(config.dataset_name, cache_path))
    if cache_path and os.path.exists(embed_matrix_path) and not config.overwrite_cache:
        fprint(
            colored(
                "Loading cached embedding_matrix from {} (Please remove all cached files if there is any problem!)".format(
                    embed_matrix_path
                ),
                "green",
            )
        )
        embedding_matrix = pickle.load(open(embed_matrix_path, "rb"))
    else:
        glove_path = prepare_glove840_embedding(
            embed_matrix_path, config.embed_dim, config=config
        )
        embedding_matrix = np.zeros(
            (len(tokenizer.word2idx) + 1, config.embed_dim)
        )  # idx 0 and len(word2idx)+1 are all-zeros

        word_vec = _load_word_vec(
            glove_path, word2idx=tokenizer.word2idx, embed_dim=config.embed_dim
        )

        for word, i in tqdm.tqdm(
            tokenizer.word2idx.items(),
            desc=colored("Building embedding_matrix {}".format(cache_path), "yellow"),
        ):
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        if config.cache_dataset:
            pickle.dump(embedding_matrix, open(embed_matrix_path, "wb"))
    return embedding_matrix


def pad_and_truncate(sequence, max_seq_len, value, **kwargs):
    """
    Pad or truncate a sequence to a specified maximum sequence length.

    Args:
        sequence (list or np.ndarray): The sequence of elements to be padded or truncated.
        max_seq_len (int): The maximum sequence length to pad or truncate to.
        value: The value to use for padding.
        **kwargs: Additional keyword arguments to ignore.

    Returns:
        np.ndarray or list: The padded or truncated sequence, as a list or numpy array, depending on the type of the input sequence.
    """
    if isinstance(sequence, ndarray):
        sequence = list(sequence)
        if len(sequence) > max_seq_len:
            sequence = sequence[:max_seq_len]
        else:
            sequence = sequence + [value] * (max_seq_len - len(sequence))
        return np.array(sequence)
    else:
        if len(sequence) > max_seq_len:
            sequence = sequence[:max_seq_len]
        else:
            sequence = sequence + [value] * (max_seq_len - len(sequence))
        return sequence


def _load_word_vec(path, word2idx=None, embed_dim=300):
    """
    Loads word vectors from a given embedding file and returns a dictionary of word to vector mappings.

    Args:
        path (str): Path to the embedding file.
        word2idx (dict): A dictionary containing word to index mappings.
        embed_dim (int): The dimension of the word embeddings.

    Returns:
        word_vec (dict): A dictionary containing word to vector mappings.
    """
    fin = open(path, "r", encoding="utf-8", newline="\n", errors="ignore")
    # Open the embedding file for reading
    word_vec = {}
    # Initialize an empty dictionary to store word to vector mappings
    for line in tqdm.tqdm(fin.readlines(), desc="Loading embedding file"):
        # Iterate over each line of the file
        tokens = line.rstrip().split()
        # Split the line by space characters and strip the newline character
        word, vec = " ".join(tokens[:-embed_dim]), tokens[-embed_dim:]
        # Split the tokens into word and vector
        if word in word2idx.keys():
            # Check if the word is in the given word to index mappings
            word_vec[word] = np.asarray(vec, dtype="float32")
            # Add the word to vector mapping to the dictionary
    return word_vec
