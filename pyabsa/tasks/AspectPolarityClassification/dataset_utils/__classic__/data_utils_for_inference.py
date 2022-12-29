# -*- coding: utf-8 -*-
# file: data_utils_for_inference.py
# time: 02/11/2022 15:39
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import numpy as np
import tqdm
from torch.utils.data import Dataset

from pyabsa import LabelPaddingOption
from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.utils.pyabsa_utils import validate_example, fprint
from .classic_glove_apc_utils import build_sentiment_window
from .dependency_graph import dependency_adj_matrix, configure_spacy_model
from ..__lcf__.data_utils_for_inference import ABSAInferenceDataset


class GloVeABSAInferenceDataset(ABSAInferenceDataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        configure_spacy_model(config)

        self.data = []

    def process_data(self, samples, ignore_error=True):
        all_data = []

        if len(samples) > 100:
            it = tqdm.tqdm(samples, desc="preparing apc inference dataloader")
        else:
            it = samples
        for ex_id, text in enumerate(it):
            try:
                # handle for empty lines in inference dataset
                if text is None or "" == text.strip():
                    raise RuntimeError("Invalid Input!")

                # check for given polarity
                if "$LABEL$" in text:
                    text, polarity = (
                        text.split("$LABEL$")[0].strip(),
                        text.split("$LABEL$")[1].strip(),
                    )
                    text = text.replace("[PADDING]", "")

                    polarity = (
                        polarity if polarity else LabelPaddingOption.LABEL_PADDING
                    )

                else:
                    polarity = str(LabelPaddingOption.LABEL_PADDING)

                # simply add padding in case of some aspect is at the beginning or ending of a sentence
                text_left, aspect, text_right = text.split("[ASP]")
                text_left = text_left.replace("[PADDING] ", "").lower().strip()
                text_right = text_right.replace(" [PADDING]", "").lower().strip()
                aspect = aspect.lower().strip()
                text = text_left + " " + aspect + " " + text_right

                if validate_example(text, aspect, polarity, self.config) or not aspect:
                    continue

                text_indices = self.tokenizer.text_to_sequence(
                    text_left + " " + aspect + " " + text_right
                )
                context_indices = self.tokenizer.text_to_sequence(
                    text_left + " " + text_right
                )
                left_indices = self.tokenizer.text_to_sequence(text_left)
                left_with_aspect_indices = self.tokenizer.text_to_sequence(
                    text_left + " " + aspect
                )
                right_indices = self.tokenizer.text_to_sequence(
                    text_right, reverse=True
                )
                right_with_aspect_indices = self.tokenizer.text_to_sequence(
                    aspect + " " + text_right, reverse=True
                )
                aspect_indices = self.tokenizer.text_to_sequence(aspect)
                left_len = np.count_nonzero(left_indices)
                aspect_len = np.count_nonzero(aspect_indices)
                aspect_boundary = np.asarray(
                    [
                        left_len,
                        min(left_len + aspect_len - 1, self.config.max_seq_len - 1),
                    ]
                )

                idx2graph = dependency_adj_matrix(
                    text_left + " " + aspect + " " + text_right
                )
                dependency_graph = np.pad(
                    idx2graph,
                    (
                        (0, max(0, self.config.max_seq_len - idx2graph.shape[0])),
                        (0, max(0, self.config.max_seq_len - idx2graph.shape[0])),
                    ),
                    "constant",
                )
                dependency_graph = dependency_graph[
                    :, range(0, self.config.max_seq_len)
                ]
                dependency_graph = dependency_graph[
                    range(0, self.config.max_seq_len), :
                ]

                aspect_begin = np.count_nonzero(
                    self.tokenizer.text_to_sequence(text_left)
                )
                aspect_position = set(
                    range(aspect_begin, aspect_begin + np.count_nonzero(aspect_indices))
                )
                if len(aspect_position) < 1:
                    raise RuntimeError("Invalid Input: {}".format(text))
                validate_example(text, aspect, polarity, config=self.config)

                data = {
                    "ex_id": ex_id,
                    "text_indices": text_indices
                    if "text_indices" in self.config.inputs_cols
                    else 0,
                    "context_indices": context_indices
                    if "context_indices" in self.config.inputs_cols
                    else 0,
                    "left_indices": left_indices
                    if "left_indices" in self.config.inputs_cols
                    else 0,
                    "left_with_aspect_indices": left_with_aspect_indices
                    if "left_with_aspect_indices" in self.config.inputs_cols
                    else 0,
                    "right_indices": right_indices
                    if "right_indices" in self.config.inputs_cols
                    else 0,
                    "right_with_aspect_indices": right_with_aspect_indices
                    if "right_with_aspect_indices" in self.config.inputs_cols
                    else 0,
                    "aspect_indices": aspect_indices
                    if "aspect_indices" in self.config.inputs_cols
                    else 0,
                    "aspect_len": aspect_len
                    if "aspect_len" in self.config.inputs_cols
                    else 0,
                    "aspect_boundary": aspect_boundary
                    if "aspect_boundary" in self.config.inputs_cols
                    else 0,
                    "aspect_position": np.array(list(aspect_position)),
                    "dependency_graph": dependency_graph
                    if "dependency_graph" in self.config.inputs_cols
                    else 0,
                    "text_raw": text,
                    "aspect": aspect,
                    "polarity": polarity,
                }

                all_data.append(data)

            except Exception as e:
                if ignore_error:
                    fprint(
                        "Ignore error while processing: {} Error info:{}".format(
                            text, e
                        )
                    )
                else:
                    raise RuntimeError(
                        "Catch Exception: {}, use ignore_error=True to remove error samples.".format(
                            e
                        )
                    )

        all_data = build_sentiment_window(
            all_data,
            self.tokenizer,
            self.config.similarity_threshold,
            input_demands=self.config.inputs_cols,
        )
        for data in all_data:

            cluster_ids = []
            for pad_idx in range(self.config.max_seq_len):
                if pad_idx in data["cluster_ids"]:
                    cluster_ids.append(
                        self.config.label_to_index.get(
                            self.config.index_to_label.get(data["polarity"], "N.A."),
                            LabelPaddingOption.SENTIMENT_PADDING,
                        )
                    )
                else:
                    cluster_ids.append(-100)
                    # cluster_ids.append(3)

            data["cluster_ids"] = np.asarray(cluster_ids, dtype=np.int64)
            data["side_ex_ids"] = np.array(0)
            data["aspect_position"] = np.array(0)
        self.data = all_data

        self.data = PyABSADataset.covert_to_tensor(self.data)

        return self.data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
