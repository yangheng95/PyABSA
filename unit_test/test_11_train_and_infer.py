# -*- coding: utf-8 -*-
# file: test_11_train_and_infer.py
# author: PyABSA Contributors
# Copyright (C) 2021. All Rights Reserved.
#
# Smoke tests that verify training and inference work end-to-end using a small
# locally-built BERT model so that no network access is required.

import json
import os
import tempfile
import warnings

import pytest

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TINY_MODEL_DIR = os.path.join(tempfile.gettempdir(), "pyabsa-tiny-bert")


def _build_tiny_bert(model_dir: str) -> None:
    """Create a minimal BertModel + BertTokenizer saved to *model_dir*.

    The model is tiny on purpose (64-dim, 2 layers) so training/inference
    is fast even on CPU with no GPU available.
    """
    if os.path.exists(os.path.join(model_dir, "config.json")):
        return  # already built in a previous test run / test session

    from transformers import BertConfig, BertModel

    os.makedirs(model_dir, exist_ok=True)

    config = BertConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    config.save_pretrained(model_dir)
    BertModel(config).save_pretrained(model_dir)

    # Minimal vocabulary: special tokens + alphabet + a handful of content words
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    alpha = list("abcdefghijklmnopqrstuvwxyz")
    content_words = [
        "good",
        "bad",
        "great",
        "poor",
        "excellent",
        "terrible",
        "food",
        "battery",
        "service",
        "nice",
        "staff",
        "the",
        "is",
        "not",
        "very",
        "was",
        "it",
        "life",
        "laptop",
        "screen",
        "keyboard",
        "price",
        "quality",
        "speed",
        "memory",
        "camera",
        "display",
        "software",
        "support",
        "build",
        "strong",
        "fast",
        "slow",
        "easy",
        "hard",
    ]
    vocab_tokens = special_tokens + alpha + content_words
    with open(os.path.join(model_dir, "vocab.txt"), "w") as fh:
        for tok in vocab_tokens:
            fh.write(tok + "\n")

    tok_cfg = {
        "do_lower_case": True,
        "model_max_length": 128,
        "tokenizer_class": "BertTokenizer",
    }
    with open(os.path.join(model_dir, "tokenizer_config.json"), "w") as fh:
        json.dump(tok_cfg, fh)


# ---------------------------------------------------------------------------
# APC (Aspect Polarity Classification) – train + infer
# ---------------------------------------------------------------------------

APC_INFER_EXAMPLES = [
    "The [ASP]battery life[ASP] is excellent .",
    "Strong [ASP]build[ASP] though which really adds to its durability .",
    "The [ASP]food[ASP] is great .",
]


def test_apc_train_and_infer():
    """Train an APC model for 1 epoch on a small slice of Laptop14 using a
    tiny local BERT checkpoint, then run inference on a few sentences."""
    _build_tiny_bert(_TINY_MODEL_DIR)

    from pyabsa import AspectPolarityClassification as APC
    from pyabsa import DeviceTypeOption, ModelSaveOption

    config = APC.APCConfigManager.get_apc_config_english()
    config.model = APC.APCModelList.FAST_LSA_T_V2
    config.pretrained_bert = _TINY_MODEL_DIR
    config.num_epoch = 1
    config.max_seq_len = 32
    config.batch_size = 4
    config.evaluate_begin = 0
    config.log_step = -1
    config.cache_dataset = False
    # Limit to 30 raw lines (= 10 complete 3-line examples after truncation)
    config.data_num = 30

    trainer = APC.APCTrainer(
        config=config,
        dataset=APC.APCDatasetList.Laptop14,
        checkpoint_save_mode=ModelSaveOption.DO_NOT_SAVE_MODEL,
        auto_device=DeviceTypeOption.CPU,
    )
    classifier = trainer.load_trained_model()

    for example in APC_INFER_EXAMPLES:
        result = classifier.predict(example, print_result=False, ignore_error=False)
        assert result is not None
        assert "sentiment" in result

    trainer.destroy()
    classifier.destroy()


# ---------------------------------------------------------------------------
# TC (Text Classification) – train + infer
# ---------------------------------------------------------------------------

TC_INFER_EXAMPLES = [
    "I love this laptop very much !",
    "This product is terrible and disappointing .",
]


def test_tc_train_and_infer():
    """Train a TC model for 1 epoch on a small slice of SST2 using a tiny
    local BERT checkpoint, then run inference."""
    _build_tiny_bert(_TINY_MODEL_DIR)

    from pyabsa import TextClassification as TC
    from pyabsa import DeviceTypeOption, ModelSaveOption

    config = TC.TCConfigManager.get_tc_config_english()
    config.model = TC.BERTTCModelList.BERT_MLP
    config.pretrained_bert = _TINY_MODEL_DIR
    config.num_epoch = 1
    config.max_seq_len = 32
    config.batch_size = 4
    config.evaluate_begin = 0
    config.log_step = -1
    config.cache_dataset = False
    config.data_num = 30

    trainer = TC.TCTrainer(
        config=config,
        dataset=TC.TCDatasetList.SST2,
        checkpoint_save_mode=ModelSaveOption.DO_NOT_SAVE_MODEL,
        auto_device=DeviceTypeOption.CPU,
    )
    classifier = trainer.load_trained_model()

    for example in TC_INFER_EXAMPLES:
        result = classifier.predict(example, print_result=False, ignore_error=False)
        assert result is not None

    trainer.destroy()
    classifier.destroy()


# ---------------------------------------------------------------------------
# ATEPC (Aspect Term Extraction + Polarity Classification) – train + infer
# ---------------------------------------------------------------------------

ATEPC_INFER_EXAMPLES = [
    "But the staff was so nice to us .",
    "The food is absolutely delicious and the service is great .",
]


def test_atepc_train_and_infer():
    """Train an ATEPC model for 1 epoch using a tiny local BERT checkpoint,
    then run batch inference."""
    _build_tiny_bert(_TINY_MODEL_DIR)

    from pyabsa import AspectTermExtraction as ATEPC
    from pyabsa import DeviceTypeOption

    config = ATEPC.ATEPCConfigManager.get_atepc_config_english()
    config.model = ATEPC.ATEPCModelList.FAST_LCF_ATEPC
    config.pretrained_bert = _TINY_MODEL_DIR
    config.num_epoch = 1
    config.max_seq_len = 32
    config.batch_size = 4
    config.evaluate_begin = 0
    config.log_step = -1
    config.cache_dataset = False
    config.data_num = 60
    config.cross_validate_fold = -1

    trainer = ATEPC.ATEPCTrainer(
        config=config,
        dataset=ATEPC.ATEPCDatasetList.Restaurant16,
        checkpoint_save_mode=1,
        auto_device=DeviceTypeOption.CPU,
    )
    extractor = trainer.load_trained_model()
    trainer.destroy()

    results = extractor.predict(
        ATEPC_INFER_EXAMPLES,
        print_result=False,
        pred_sentiment=True,
        ignore_error=True,
    )
    assert results is not None

    extractor.destroy()


# ---------------------------------------------------------------------------
# Entry point for running directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_apc_train_and_infer()
    test_tc_train_and_infer()
    test_atepc_train_and_infer()
    print("All training and inference tests passed.")
