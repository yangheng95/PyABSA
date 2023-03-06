# -*- coding: utf-8 -*-
# file: data_utils_for_training.py
# time: 2021/5/31 0031
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import json
import re
from collections import Counter, OrderedDict

import numpy as np
import tqdm
from pyabsa.tasks.AspectPolarityClassification.dataset_utils.__lcf__.apc_utils import (
    configure_spacy_model,
)
from termcolor import colored

from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.tasks.AspectSentimentTripletExtraction.dataset_utils.aste_utils import (
    VocabHelp,
    Instance,
    load_tokens,
)
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from pyabsa.utils.pyabsa_utils import check_and_fix_labels, fprint


def generate_tags(tokens, start, end, scheme):
    # print('Generating tags for tokens: ', tokens)
    if scheme == "BIO":
        tags = ["O"] * len(tokens)
        tags[start] = "B"
        for i in range(start + 1, end + 1):
            tags[i] = "I"
        return " ".join([f"{token}\\{tag}" for token, tag in zip(tokens, tags)])
    elif scheme == "IOB2":
        tags = ["O"] * len(tokens)
        tags[start] = "B"
        for i in range(start + 1, end + 1):
            tags[i] = "I"
        if end < len(tokens) - 1 and tags[end + 1] == "I":
            tags[end] = "B"
        return " ".join([f"{token}\\{tag}" for token, tag in zip(tokens, tags)])
    else:
        raise ValueError(f"Invalid tagging scheme '{scheme}'.")


def load_tokens(data):
    tokens = []
    deprel = []
    postag = []
    postag_ca = []

    max_len = 0
    sentence = data["sentence"].split()
    tokens.extend(sentence)
    deprel.extend(data["deprel"])
    postag_ca.extend(data["postag"])
    # postag.extend(d['postag'])
    n = len(data["postag"])
    tmp_pos = []
    for i in range(n):
        for j in range(n):
            tup = tuple(sorted([data["postag"][i], data["postag"][j]]))
            tmp_pos.append(tup)
    postag.extend(tmp_pos)

    max_len = max(len(sentence), max_len)
    return tokens, deprel, postag, postag_ca, max_len


class ASTEDataset(PyABSADataset):
    syn_post_vocab = None
    postag_vocab = None
    deprel_vocab = None
    post_vocab = None
    syn_post_vocab = None
    token_vocab = None

    all_tokens = []
    all_deprel = []
    all_postag = []
    all_postag_ca = []
    all_max_len = []

    labels = [
        "N",
        "B-A",
        "I-A",
        "A",
        "B-O",
        "I-O",
        "O",
        "Negative",
        "Neutral",
        "Positive",
    ]
    label_to_index, index_to_label = OrderedDict(), OrderedDict()
    for i, v in enumerate(labels):
        label_to_index[v] = i
        index_to_label[i] = v

    def load_data_from_dict(self, data_dict, **kwargs):
        pass

    def load_data_from_file(self, file_path, **kwargs):
        lines = load_dataset_from_file(
            self.config.dataset_file[self.dataset_type], config=self.config
        )

        all_data = []
        # record polarities type to update output_dim
        label_set = set()

        for ex_id in tqdm.tqdm(range(0, len(lines)), desc="preparing dataloader"):
            if lines[ex_id].count("####"):
                sentence, annotations = lines[ex_id].split("####")
            elif lines[ex_id].count("$LABEL$"):
                sentence, annotations = lines[ex_id].split("$LABEL$")
            else:
                raise ValueError(
                    "Invalid annotations format, please check your dataset file."
                )

            sentence = sentence.strip()
            annotations = annotations.strip()
            annotations = eval(annotations)

            prepared_data = self.get_syntax_annotation(sentence, annotations)
            prepared_data["id"] = ex_id
            tokens, deprel, postag, postag_ca, max_len = load_tokens(prepared_data)
            self.all_tokens.extend(tokens)
            self.all_deprel.extend(deprel)
            self.all_postag.extend(postag)
            self.all_postag_ca.extend(postag_ca)
            self.all_max_len.append(max_len)
            label_set.add(annotation[-1] for annotation in annotations)
            all_data.append(prepared_data)

        self.data = all_data

    def __init__(self, config, tokenizer, dataset_type="train"):
        self.nlp = configure_spacy_model(config)
        super().__init__(config=config, tokenizer=tokenizer, dataset_type=dataset_type)
        self.config.label_to_index = self.label_to_index
        self.config.index_to_label = self.index_to_label
        self.config.output_dim = len(self.label_to_index)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def convert_examples_to_features(self):
        self.get_vocabs()
        _data = []
        for data in tqdm.tqdm(self.data, desc="converting data to features"):
            try:
                feat = Instance(
                    self.tokenizer,
                    data,
                    self.post_vocab,
                    self.deprel_vocab,
                    self.postag_vocab,
                    self.syn_post_vocab,
                    self.config,
                )
                _data.append(feat)
            except Exception as e:
                fprint(
                    "Processing error for: {}. Exception: {}".format(
                        data["sentence"], e
                    )
                )
        self.data = _data

    def get_syntax_annotation(self, sentence, annotation):
        # Extract aspect and opinion terms from annotation
        aspect_spans = [
            (aspect_span[0], aspect_span[-1]) for (aspect_span, _, _) in annotation
        ]
        opinion_spans = [
            (opinion_span[0], opinion_span[-1]) for (_, opinion_span, _) in annotation
        ]
        sentiments = [sentiment_label for (_, _, sentiment_label) in annotation]

        # Tokenize sentence
        # tokens = re.findall(r'\w+|[^\w\s]', sentence)
        tokens = sentence.split()
        postags, heads, deprels = self.get_dependencies(tokens)

        # Generate triples
        triples = []
        for i, aspect_span in enumerate(aspect_spans):
            for j, opinion_span in enumerate(opinion_spans):
                if aspect_span == opinion_span:
                    continue
                aspect_start, aspect_end = aspect_span
                opinion_start, opinion_end = opinion_span
                # if aspect_start > opinion_start:
                #     aspect_start, opinion_start = opinion_start, aspect_start
                #     aspect_end, opinion_end = opinion_end, aspect_end
                # if aspect_end >= opinion_start:
                #     continue
                uid = f"{i}-{j}"
                target_tags = generate_tags(tokens, aspect_start, aspect_end, "BIO")
                opinion_tags = generate_tags(tokens, opinion_start, opinion_end, "BIO")
                triples.append(
                    {
                        "uid": uid,
                        "target_tags": target_tags,
                        "opinion_tags": opinion_tags,
                        "sentiment": sentiments[j]
                        .replace("POS", "Positive")
                        .replace("NEG", "Negative")
                        .replace("NEU", "Neutral"),
                    }
                )

        # Generate output dictionary
        output = {
            "id": "",
            "sentence": sentence,
            "postag": postags,
            "head": heads,
            "deprel": deprels,
            "triples": triples,
        }

        return output

    def generate_tags(self, tokens, start, end, scheme):
        if scheme == "BIO":
            tags = ["O"] * len(tokens)
            tags[start] = "B"
            for i in range(start + 1, end + 1):
                tags[i] = "I"
            return " ".join([f"{token}\\{tag}" for token, tag in zip(tokens, tags)])
        elif scheme == "IOB2":
            tags = ["O"] * len(tokens)
            tags[start] = "B"
            for i in range(start + 1, end + 1):
                tags[i] = "I"
            if end < len(tokens) - 1 and tags[end + 1] == "I":
                tags[end] = "B"
            return " ".join([f"{token}\\{tag}" for token, tag in zip(tokens, tags)])
        else:
            raise ValueError(f"Invalid tagging scheme '{scheme}'.")

    def get_dependencies(self, tokens):
        # Replace special characters in tokens with placeholders
        placeholder_tokens = []
        for token in tokens:
            if re.search(r"[^\w\s]", token):
                placeholder = f"__{token}__"
                placeholder_tokens.append(placeholder)
            else:
                placeholder_tokens.append(token)

        # Get part-of-speech tags and dependencies using spaCy
        doc = self.nlp(" ".join(tokens))
        postags = [token.pos_ for token in doc]
        heads = [token.head.i for token in doc]
        deprels = [token.dep_ for token in doc]

        return postags, heads, deprels

    def get_vocabs(self):
        if (
            self.syn_post_vocab is None
            and self.postag_vocab is None
            and self.deprel_vocab is None
            and self.syn_post_vocab is None
            and self.token_vocab is None
        ):
            token_counter = Counter(self.all_tokens)
            deprel_counter = Counter(self.all_deprel)
            postag_counter = Counter(self.all_postag)
            postag_ca_counter = Counter(self.all_postag_ca)
            # deprel_counter['ROOT'] = 1
            deprel_counter["self"] = 1

            max_len = max(self.all_max_len)
            # post_counter = Counter(list(range(-max_len, max_len)))
            post_counter = Counter(list(range(0, max_len)))
            syn_post_counter = Counter(list(range(0, 5)))

            # build vocab
            print("building vocab...")
            token_vocab = VocabHelp(token_counter, specials=["<pad>", "<unk>"])
            post_vocab = VocabHelp(post_counter, specials=["<pad>", "<unk>"])
            deprel_vocab = VocabHelp(deprel_counter, specials=["<pad>", "<unk>"])
            postag_vocab = VocabHelp(postag_counter, specials=["<pad>", "<unk>"])
            syn_post_vocab = VocabHelp(syn_post_counter, specials=["<pad>", "<unk>"])
            # print("token_vocab: {}, post_vocab: {}, syn_post_vocab: {}, deprel_vocab: {}, postag_vocab: {}".format(
            #     len(token_vocab), len(post_vocab), len(syn_post_vocab), len(deprel_vocab), len(postag_vocab)))

            self.token_vocab = token_vocab
            self.post_vocab = post_vocab
            self.deprel_vocab = deprel_vocab
            self.postag_vocab = postag_vocab
            self.syn_post_vocab = syn_post_vocab
            self.config.post_size = len(post_vocab)
            self.config.deprel_size = len(deprel_vocab)
            self.config.postag_size = len(postag_vocab)
            self.config.synpost_size = len(syn_post_vocab)
