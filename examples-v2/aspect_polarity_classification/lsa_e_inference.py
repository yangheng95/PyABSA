# -*- coding: utf-8 -*-
# file: inference.py
# time: 05/11/2022 19:48
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import random

import findfile
import tqdm
from sklearn import metrics
from sklearn.metrics import classification_report

from pyabsa import AspectPolarityClassification as APC

from pyabsa import (
    AspectPolarityClassification as APC,
    ModelSaveOption,
    DeviceTypeOption,
)
import warnings

from pyabsa.tasks.AspectPolarityClassification import APCDatasetList
from pyabsa.utils import VoteEnsemblePredictor
from pyabsa.utils.pyabsa_utils import fprint, rprint

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # Training the models before ensemble inference, take Laptop14 as an example

    # for dataset in [
    #     APCDatasetList.Laptop14,
    #     # APCDatasetList.Restaurant14,
    #     # APCDatasetList.Restaurant15,
    #     # APCDatasetList.Restaurant16,
    #     # APCDatasetList.MAMS
    # ]:
    #     for model in [
    #         APC.APCModelList.FAST_LSA_T_V2,
    #         APC.APCModelList.FAST_LSA_S_V2,
    #         # APC.APCModelList.BERT_SPC_V2  # BERT_SPC_V2 is slow in ensemble inference so we don't use it
    #     ]:
    #         config = APC.APCConfigManager.get_apc_config_english()
    #         config.model = model
    #         config.pretrained_bert = 'microsoft/deberta-v3-base'
    #         config.evaluate_begin = 5
    #         config.max_seq_len = 80
    #         config.num_epoch = 30
    #         config.log_step = 10
    #         config.patience = 10
    #         config.dropout = 0
    #         config.cache_dataset = False
    #         config.l2reg = 1e-8
    #         config.lsa = True
    #         config.seed = [random.randint(0, 10000) for _ in range(3)]
    #
    #         APC.APCTrainer(config=config,
    #                        dataset=dataset,
    #                        checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
    #                        auto_device=DeviceTypeOption.AUTO,
    #                        ).destroy()
    # Ensemble inference
    dataset_file_dict = {
        # 'laptop14': findfile.find_cwd_files(['laptop14', '.inference'], exclude_key=[]),
        "laptop14": "integrated_datasets/apc_datasets/110.SemEval/113.laptop14/Laptops_Test_Gold.xml.seg.inference",
        "restaurant14": "integrated_datasets/apc_datasets/110.SemEval/114.restaurant14/Restaurants_Test_Gold.xml.seg.inference",
        "restaurant15": "integrated_datasets/apc_datasets/110.SemEval/115.restaurant15/restaurant_test.raw.inference",
        "restaurant16": "integrated_datasets/apc_datasets/110.SemEval/116.restaurant16/restaurant_test.raw.inference",
        "twitter": "integrated_datasets/apc_datasets/120.Twitter/120.twitter/twitter_test.raw.inference",
        "mams": "integrated_datasets/apc_datasets/109.MAMS/test.xml.dat.inference",
    }
    for model_name in ["bert-base-uncased"]:
        for dataset in [
            "laptop14",
            "restaurant14",
            "restaurant15",
            "restaurant16",
            "mams",
        ]:
            if len(findfile.find_cwd_dirs(key=[f"{dataset}_acc", model_name])) == 0:
                rprint(f"No checkpoints found for {dataset} {model_name}")
                continue

            checkpoints = {
                ckpt: APC.SentimentClassifier(checkpoint=ckpt)
                for ckpt in findfile.find_cwd_dirs(key=[f"{dataset}_acc", model_name])
            }

            ensemble_predictor = VoteEnsemblePredictor(
                checkpoints, weights=None, numeric_agg="mean", str_agg="max_vote"
            )

            files = dataset_file_dict[dataset]
            text_classifiers = {}

            lines = []
            if isinstance(files, str):
                files = [files]
                for file in files:
                    with open(file, "r") as f:
                        lines.extend(f.readlines())

            # 测试总体准确率 batch predict
            # eval acc
            count1 = 0
            accuracy = 0
            batch_pred = []
            batch_gold = []

            # do not merge the same sentence
            results = ensemble_predictor.batch_predict(
                lines, ignore_error=False, print_result=False
            )
            it = tqdm.tqdm(results, ncols=100)
            for i, result in enumerate(it):
                label = result["sentiment"]
                if label == lines[i].split("$LABEL$")[-1].strip():
                    count1 += 1
                batch_pred.append(label)
                batch_gold.append(lines[i].split("$LABEL$")[-1].strip().split(","))
                accuracy = count1 / (i + 1)
                it.set_description(f"Accuracy: {accuracy:.4f}")

                fprint(f"{model_name} {dataset} Accuracy: {accuracy:.4f}")

            # while True:
            #     text = input("Please input your text sequence: ")
            #     if text == "exit":
            #         break
            #     if text == "":
            #         continue
            #     _, _, true_label = text.partition("$LABEL$")
            #     try:
            #         result = ensemble_predictor.predict(
            #             text, ignore_error=False, print_result=False
            #         )
            #         print(result)
            #         pred_label = result["label"]
            #         confidence = result["confidence"]
            #         fprint(
            #             "Predicted Label:",
            #             pred_label,
            #             "Reference Label: ",
            #             true_label,
            #             "Correct: ",
            #             pred_label == true_label,
            #         )
            #     except Exception as e:
            #         fprint(e)
