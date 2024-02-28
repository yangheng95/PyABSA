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

warnings.filterwarnings("ignore")
import pandas as pd


task_name = "multitask"
experiment_name = "instruction"
# model_checkpoint = 'allenai/tk-instruct-base-def-pos'
model_checkpoint = "kevinscaria/ate_tk-instruct-base-def-pos-neg-neut-combined"
# model_checkpoint = 'allenai/tk-instruct-large-def-pos'
# model_checkpoint = 'allenai/tk-instruct-3b-def-pos'
# model_checkpoint = 'google/mt5-base'

print("Experiment Name: ", experiment_name)
model_out_path = "checkpoints"
model_out_path = os.path.join(
    model_out_path, task_name, f"{model_checkpoint.replace('/', '')}-{experiment_name}"
)
print("Model output path: ", model_out_path)

# Load the data
# id_train_file_path = './integrated_datasets'
# id_test_file_path = './integrated_datasets'
id_train_file_path = "./integrated_datasets/acos_datasets/"
id_test_file_path = "./integrated_datasets/acos_datasets"
# id_train_file_path = './integrated_datasets/acos_datasets/501.Laptop14'
# id_test_file_path = './integrated_datasets/acos_datasets/501.Laptop14'
# id_train_file_path = './integrated_datasets/acos_datasets/504.Restaurant16'
# id_test_file_path = './integrated_datasets/acos_datasets/504.Restaurant16'


id_tr_df = read_json(id_train_file_path, "train")
id_te_df = read_json(id_test_file_path, "test")

id_tr_df = pd.DataFrame(id_tr_df)
id_te_df = pd.DataFrame(id_te_df)

loader = InstructDatasetLoader(id_tr_df, id_te_df)

if loader.train_df_id is not None:
    loader.train_df_id = loader.prepare_instruction_dataloader(loader.train_df_id)
if loader.test_df_id is not None:
    loader.test_df_id = loader.prepare_instruction_dataloader(loader.test_df_id)
if loader.train_df_ood is not None:
    loader.train_df_ood = loader.prepare_instruction_dataloader(loader.train_df_ood)
if loader.test_df_ood is not None:
    loader.test_df_ood = loader.prepare_instruction_dataloader(loader.test_df_ood)

# Create T5 utils object
t5_exp = T5Generator(model_checkpoint)

# Tokenize Dataset
id_ds, id_tokenized_ds, ood_ds, ood_tokenzed_ds = loader.create_datasets(
    t5_exp.tokenize_function_inputs
)

# Training arguments
training_args = {
    "output_dir": model_out_path,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 6,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "load_best_model_at_end": True,
    "push_to_hub": False,
    "eval_accumulation_steps": 1,
    "predict_with_generate": True,
    "logging_steps": 1000000000,
    "use_mps_device": False,
    # 'fp16': True,
    "fp16": False,
}

# Train model
model_trainer = t5_exp.train(id_tokenized_ds, **training_args)

# Model inference - Trainer object - (Pass model trainer as predictor)

# model_checkpoint = findfile.find_cwd_dir('tk-instruct-base-def-pos')
# t5_exp = T5Generator(model_checkpoint)

# Get prediction labels - Training set
id_tr_pred_labels = t5_exp.get_labels(
    predictor=model_trainer,
    tokenized_dataset=id_tokenized_ds,
    sample_set="train",
    batch_size=16,
)
id_tr_labels = [i.strip() for i in id_ds["train"]["labels"]]

# Get prediction labels - Testing set
id_te_pred_labels = t5_exp.get_labels(
    predictor=model_trainer,
    tokenized_dataset=id_tokenized_ds,
    sample_set="test",
    batch_size=16,
)
id_te_labels = [i.strip() for i in id_ds["test"]["labels"]]

# # Compute Metrics
# metrics = t5_exp.get_metrics(id_tr_labels, id_tr_pred_labels)
# print('----------------------- Training Set Metrics -----------------------')
# print(metrics)
#
# metrics = t5_exp.get_metrics(id_te_labels, id_te_pred_labels)
# print('----------------------- Testing Set Metrics -----------------------')
# print(metrics)

# Compute Metrics
metrics = t5_exp.get_classic_metrics(id_tr_labels, id_tr_pred_labels)
print("----------------------- Classic Training Set Metrics -----------------------")
print(metrics)

metrics = t5_exp.get_classic_metrics(id_te_labels, id_te_pred_labels)
print("----------------------- Classic Testing Set Metrics -----------------------")
print(metrics)
