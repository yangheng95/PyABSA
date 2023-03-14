# -*- coding: utf-8 -*-
# file: train.py
# time: 11:30 2023/3/13
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2023. All Rights Reserved.
import os
import warnings

import pandas as pd

from .data_utils import DatasetLoader, read_json
from .utils import T5Generator, T5Classifier
from .instructions import InstructionsHandler

warnings.filterwarnings("ignore")


task_name = "multitask"
experiment_name = "instruction"
model_checkpoint = "allenai/tk-instruct-base-def-pos"
print("Experiment Name: ", experiment_name)
model_out_path = "checkpoints"
model_out_path = os.path.join(
    model_out_path, task_name, f"{model_checkpoint.replace('/', '')}-{experiment_name}"
)
print("Model output path: ", model_out_path)

# Load the data
id_train_file_path = "./integrated_datasets"
id_test_file_path = "./integrated_datasets"
id_tr_df = read_json(id_train_file_path, "train")
id_te_df = read_json(id_test_file_path, "test")

id_tr_df = pd.DataFrame(id_tr_df)
id_te_df = pd.DataFrame(id_te_df)

# Get the input text into the required format using Instructions
instruct_handler = InstructionsHandler()
instruct_handler.load_instruction_set1()

loader = DatasetLoader(id_tr_df, id_te_df)

if loader.train_df_id is not None:
    loader.train_df_id = loader.create_data_for_multitask(
        loader.train_df_id,
        instruct_handler.multitask["bos_instruct3"],
        instruct_handler.multitask["eos_instruct"],
    )
if loader.test_df_id is not None:
    loader.test_df_id = loader.create_data_for_multitask(
        loader.test_df_id,
        instruct_handler.multitask["bos_instruct3"],
        instruct_handler.multitask["eos_instruct"],
    )
if loader.train_df_ood is not None:
    loader.train_df_ood = loader.create_data_for_multitask(
        loader.train_df_ood,
        instruct_handler.multitask["bos_instruct3"],
        instruct_handler.multitask["eos_instruct"],
    )
if loader.test_df_ood is not None:
    loader.test_df_ood = loader.create_data_for_multitask(
        loader.test_df_ood,
        instruct_handler.multitask["bos_instruct3"],
        instruct_handler.multitask["eos_instruct"],
    )

# Create T5 utils object
t5_exp = T5Generator(model_checkpoint)

# Tokenize Dataset
id_ds, id_tokenized_ds, ood_ds, ood_tokenzed_ds = loader.set_data_for_training_semeval(
    t5_exp.tokenize_function_inputs
)

# Training arguments
training_args = {
    "output_dir": model_out_path,
    "evaluation_strategy": "epoch",
    "save_strategy": "no",
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 12,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 8,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "load_best_model_at_end": False,
    "push_to_hub": False,
    "eval_accumulation_steps": 1,
    "predict_with_generate": True,
    "logging_steps": 1000000000,
    "use_mps_device": False,
    "fp16": False,
}

# Train model
model_trainer = t5_exp.train(id_tokenized_ds, **training_args)

# Model inference - Trainer object - (Pass model trainer as predictor)

# Get prediction labels - Training set
id_tr_pred_labels = t5_exp.get_labels(
    predictor=model_trainer, tokenized_dataset=id_tokenized_ds, sample_set="train"
)
id_tr_labels = [i.strip() for i in id_ds["train"]["labels"]]

# Get prediction labels - Testing set
id_te_pred_labels = t5_exp.get_labels(
    predictor=model_trainer, tokenized_dataset=id_tokenized_ds, sample_set="test"
)
id_te_labels = [i.strip() for i in id_ds["test"]["labels"]]

# Compute Metrics
metrics = t5_exp.get_metrics(id_tr_labels, id_tr_pred_labels)
print("----------------------- Training Set Metrics -----------------------")
print(metrics)

metrics = t5_exp.get_metrics(id_te_labels, id_te_pred_labels)
print("----------------------- Testing Set Metrics -----------------------")
print(metrics)

# Compute Metrics
metrics = t5_exp.get_classic_metrics(id_tr_labels, id_tr_pred_labels)
print("----------------------- Classic Training Set Metrics -----------------------")
print(metrics)

metrics = t5_exp.get_classic_metrics(id_te_labels, id_te_pred_labels)
print("----------------------- Classic Testing Set Metrics -----------------------")
print(metrics)
