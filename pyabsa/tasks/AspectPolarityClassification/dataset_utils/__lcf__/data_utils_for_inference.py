# -*- coding: utf-8 -*-
# file: data_utils_for_inferring.py
# time: 2021/4/22 0022
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import re
from typing import Union, List

import numpy as np

from pyabsa import LabelPaddingOption
from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from torch.utils.data import Dataset
import tqdm

from pyabsa.utils.pyabsa_utils import fprint
from .apc_utils import (
    build_sentiment_window,
    build_spc_mask_vec,
    prepare_input_for_apc,
    configure_spacy_model,
)
from .apc_utils_for_dlcf_dca import (
    prepare_input_for_dlcf_dca,
    configure_dlcf_spacy_model,
)


def parse_sample(text):
    if "[B-ASP]" not in text and "[ASP]" not in text:
        # if '[B-ASP]' not in text or '[E-ASP]' not in text:
        text = " [B-ASP]Global Sentiment[E-ASP] " + text

    _text = text
    samples = []

    if "$LABEL$" not in text:
        text += "$LABEL$"
    text, _, ref_sent = text.partition("$LABEL$")
    if "[B-ASP]" in text:
        ref_sent = ref_sent.split(",") if ref_sent else []
        aspects = re.findall(r"\[B\-ASP\](.*?)\[E\-ASP\]", text)

        for i, aspect in enumerate(aspects):
            sample = (
                text.replace(f"[B-ASP]{aspect}[E-ASP]", f"[TEMP]{aspect}[TEMP]", 1)
                .replace("[B-ASP]", "")
                .replace("[E-ASP]", "")
            )
            if len(aspects) == len(ref_sent):
                sample += f"$LABEL${ref_sent[i]}"
                samples.append(sample.replace("[TEMP]", "[ASP]"))
            else:
                fprint(
                    f"Warning: aspect number {len(aspects)} not equal to reference sentiment number {len(ref_sent)}, text: {_text}"
                )
                samples.append(sample.replace("[TEMP]", "[ASP]"))

    else:
        fprint(
            "[ASP] tag is detected, please use [B-ASP] and [E-ASP] to annotate aspect terms."
        )
        splits = text.split("[ASP]")
        ref_sent = ref_sent.split(",") if ref_sent else []

        if ref_sent and int((len(splits) - 1) / 2) == len(ref_sent):
            for i in range(0, len(splits) - 1, 2):
                sample = text.replace(
                    "[ASP]" + splits[i + 1] + "[ASP]",
                    "[TEMP]" + splits[i + 1] + "[TEMP]",
                    1,
                ).replace("[ASP]", "")
                sample += " $LABEL$ " + str(ref_sent[int(i / 2)])
                samples.append(sample.replace("[TEMP]", "[ASP]"))
        elif not ref_sent or int((len(splits) - 1) / 2) != len(ref_sent):
            # if not ref_sent:
            #     fprint(_text, ' -> No the reference sentiment found')
            if ref_sent:
                fprint(
                    _text,
                    " -> Unequal length of reference sentiment and aspects, ignore the reference sentiment.",
                )

            for i in range(0, len(splits) - 1, 2):
                sample = text.replace(
                    "[ASP]" + splits[i + 1] + "[ASP]",
                    "[TEMP]" + splits[i + 1] + "[TEMP]",
                    1,
                ).replace("[ASP]", "")
                samples.append(sample.replace("[TEMP]", "[ASP]"))
        else:
            raise ValueError("Invalid Input:{}".format(text))

    return samples


class ABSAInferenceDataset(Dataset):
    def __init__(self, config, tokenizer):
        configure_spacy_model(config)
        self.tokenizer = tokenizer
        self.config = config
        self.data = []

    def prepare_infer_sample(self, text: Union[str, List[str]], ignore_error=True):
        if isinstance(text, str):
            self.process_data(parse_sample(text), ignore_error=ignore_error)
        elif isinstance(text, list):
            examples = []
            for sample in text:
                examples.extend(parse_sample(sample))
            self.process_data(examples, ignore_error=ignore_error)

    def prepare_infer_dataset(self, infer_file, ignore_error):
        lines = load_dataset_from_file(infer_file, config=self.config)
        samples = []
        for sample in lines:
            if sample:
                samples.extend(parse_sample(sample))
        self.process_data(samples, ignore_error)

    def process_data(self, samples, ignore_error=True):
        all_data = []
        label_set = set()
        ex_id = 0
        if len(samples) > 100:
            it = tqdm.tqdm(samples, desc="preparing apc inference dataloader")
        else:
            it = samples
        for i, text in enumerate(it):
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
                    polarity = (
                        polarity if polarity else LabelPaddingOption.LABEL_PADDING
                    )
                    text = text.replace("[PADDING]", "")

                else:
                    polarity = str(LabelPaddingOption.LABEL_PADDING)

                # simply add padding in case of some aspect is at the beginning or ending of a sentence
                text_left, aspect, text_right = text.split("[ASP]")
                text_left = text_left.replace("[PADDING] ", "")
                text_right = text_right.replace(" [PADDING]", "")
                text = text_left + " " + aspect + " " + text_right

                prepared_inputs = prepare_input_for_apc(
                    self.config,
                    self.tokenizer,
                    text_left,
                    text_right,
                    aspect,
                    input_demands=self.config.inputs_cols,
                )

                text_raw = prepared_inputs["text_raw"]
                aspect = prepared_inputs["aspect"]
                aspect_position = prepared_inputs["aspect_position"]
                text_indices = prepared_inputs["text_indices"]
                text_raw_bert_indices = prepared_inputs["text_raw_bert_indices"]
                aspect_bert_indices = prepared_inputs["aspect_bert_indices"]

                lcf_cdw_vec = prepared_inputs["lcf_cdw_vec"]
                lcf_cdm_vec = prepared_inputs["lcf_cdm_vec"]
                lcf_vec = prepared_inputs["lcf_vec"]

                lcfs_cdw_vec = prepared_inputs["lcfs_cdw_vec"]
                lcfs_cdm_vec = prepared_inputs["lcfs_cdm_vec"]
                lcfs_vec = prepared_inputs["lcfs_vec"]

                if (
                    self.config.model_name == "dlcf_dca_bert"
                    or self.config.model_name == "dlcfs_dca_bert"
                ):
                    configure_dlcf_spacy_model(self.config)
                    prepared_inputs = prepare_input_for_dlcf_dca(
                        self.config, self.tokenizer, text_left, text_right, aspect
                    )
                    dlcf_vec = (
                        prepared_inputs["dlcf_cdm_vec"]
                        if self.config.lcf == "cdm"
                        else prepared_inputs["dlcf_cdw_vec"]
                    )
                    dlcfs_vec = (
                        prepared_inputs["dlcfs_cdm_vec"]
                        if self.config.lcf == "cdm"
                        else prepared_inputs["dlcfs_cdw_vec"]
                    )
                    depend_vec = prepared_inputs["depend_vec"]
                    depended_vec = prepared_inputs["depended_vec"]
                data = {
                    "ex_id": ex_id,
                    "text_raw": text_raw,
                    "aspect": aspect,
                    "aspect_position": aspect_position,
                    "lca_ids": lcf_vec,
                    # the lca indices are the same as the refactored CDM (lcf != CDW or Fusion) lcf vec
                    "lcf_vec": lcf_vec if "lcf_vec" in self.config.inputs_cols else 0,
                    "lcf_cdw_vec": lcf_cdw_vec
                    if "lcf_cdw_vec" in self.config.inputs_cols
                    else 0,
                    "lcf_cdm_vec": lcf_cdm_vec
                    if "lcf_cdm_vec" in self.config.inputs_cols
                    else 0,
                    "lcfs_vec": lcfs_vec
                    if "lcfs_vec" in self.config.inputs_cols
                    else 0,
                    "lcfs_cdw_vec": lcfs_cdw_vec
                    if "lcfs_cdw_vec" in self.config.inputs_cols
                    else 0,
                    "lcfs_cdm_vec": lcfs_cdm_vec
                    if "lcfs_cdm_vec" in self.config.inputs_cols
                    else 0,
                    "dlcf_vec": dlcf_vec
                    if "dlcf_vec" in self.config.inputs_cols
                    else 0,
                    "dlcfs_vec": dlcfs_vec
                    if "dlcfs_vec" in self.config.inputs_cols
                    else 0,
                    "depend_vec": depend_vec
                    if "depend_vec" in self.config.inputs_cols
                    else 0,
                    "depended_vec": depended_vec
                    if "depended_vec" in self.config.inputs_cols
                    else 0,
                    "spc_mask_vec": build_spc_mask_vec(
                        self.config, text_raw_bert_indices
                    )
                    if "spc_mask_vec" in self.config.inputs_cols
                    else 0,
                    "text_indices": text_indices
                    if "text_indices" in self.config.inputs_cols
                    else 0,
                    "aspect_bert_indices": aspect_bert_indices
                    if "aspect_bert_indices" in self.config.inputs_cols
                    else 0,
                    "text_raw_bert_indices": text_raw_bert_indices
                    if "text_raw_bert_indices" in self.config.inputs_cols
                    else 0,
                    "polarity": polarity,
                }

                ex_id += 1
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
                        "Ignore error while processing: {} Catch Exception: {}, use ignore_error=True to remove error samples.".format(
                            text, e
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
                    # fprint(data['polarity'])
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
