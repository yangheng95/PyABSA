# -*- coding: utf-8 -*-
# file: data_utils_for_training.py
# time: 2021/5/31 0031
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import numpy as np
import tqdm
from termcolor import colored

from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from pyabsa.utils.pyabsa_utils import check_and_fix_labels, fprint
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
from pyabsa.utils.pyabsa_utils import validate_absa_example


class ABSADataset(PyABSADataset):
    def load_data_from_dict(self, data_dict, **kwargs):
        pass

    def load_data_from_file(self, file_path, **kwargs):
        configure_spacy_model(self.config)

        lines = load_dataset_from_file(
            self.config.dataset_file[self.dataset_type], config=self.config
        )

        if len(lines) % 3 != 0 or len(lines) == 0:
            fprint(
                colored(
                    "ERROR: one or more datasets are corrupted, make sure the number of lines in a dataset should be multiples of 3.",
                    "red",
                )
            )

        all_data = []
        # record polarities type to update output_dim
        label_set = set()

        ex_id = 0

        for i in tqdm.tqdm(range(0, len(lines), 3), desc="preparing dataloader"):
            if lines[i].count("$T$") > 1:
                continue

            text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].strip()
            polarity = lines[i + 2].strip()

            prepared_inputs = prepare_input_for_apc(
                self.config,
                self.tokenizer,
                text_left,
                text_right,
                aspect,
                input_demands=self.config.inputs_cols,
            )

            text_raw = prepared_inputs["text_raw"]
            text_spc = prepared_inputs["text_spc"]
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

            # if validate_absa_example(text_raw, aspect, polarity, self.config):
            #     continue

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
                "text_spc": text_spc,
                "aspect": aspect,
                "aspect_position": aspect_position,
                "lca_ids": lcf_vec,  # the lca indices are the same as the refactored CDM (lcf != CDW or Fusion) lcf vec
                "lcf_vec": lcf_vec if "lcf_vec" in self.config.inputs_cols else 0,
                "lcf_cdw_vec": lcf_cdw_vec
                if "lcf_cdw_vec" in self.config.inputs_cols
                else 0,
                "lcf_cdm_vec": lcf_cdm_vec
                if "lcf_cdm_vec" in self.config.inputs_cols
                else 0,
                "lcfs_vec": lcfs_vec if "lcfs_vec" in self.config.inputs_cols else 0,
                "lcfs_cdw_vec": lcfs_cdw_vec
                if "lcfs_cdw_vec" in self.config.inputs_cols
                else 0,
                "lcfs_cdm_vec": lcfs_cdm_vec
                if "lcfs_cdm_vec" in self.config.inputs_cols
                else 0,
                "dlcf_vec": dlcf_vec if "dlcf_vec" in self.config.inputs_cols else 0,
                "dlcfs_vec": dlcfs_vec if "dlcfs_vec" in self.config.inputs_cols else 0,
                "depend_vec": depend_vec
                if "depend_vec" in self.config.inputs_cols
                else 0,
                "depended_vec": depended_vec
                if "depended_vec" in self.config.inputs_cols
                else 0,
                "spc_mask_vec": build_spc_mask_vec(self.config, text_raw_bert_indices)
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

            label_set.add(polarity)
            all_data.append(data)

        check_and_fix_labels(label_set, "polarity", all_data, self.config)
        self.config.output_dim = len(label_set)

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
                    cluster_ids.append(data["polarity"])
                else:
                    cluster_ids.append(-100)
                    # cluster_ids.append(3)

            data["cluster_ids"] = np.asarray(cluster_ids, dtype=np.int64)
            data["side_ex_ids"] = np.array(0)
            data["aspect_position"] = np.array(0)
        self.data = all_data

    def __init__(self, config, tokenizer, dataset_type="train"):
        super().__init__(config=config, tokenizer=tokenizer, dataset_type=dataset_type)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
