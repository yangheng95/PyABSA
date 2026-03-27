# -*- coding: utf-8 -*-
# file: test_12_real_model_train_infer.py
# author: PyABSA Contributors
# Copyright (C) 2021. All Rights Reserved.
#
# Tests that training and inference work with a realistic (non-toy) pre-trained
# BERT model.  A proper English WordPiece tokenizer is trained from an ABSA
# corpus and used to build a BERT checkpoint that the PyABSA trainers consume.
#
# No HuggingFace network access is required.  All data and model artefacts are
# created locally within the test.

import json
import os
import shutil
import tempfile
import warnings

import pytest

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Representative English ABSA/review corpus used to train the WordPiece vocab
_ABSA_CORPUS = """
I charge it at night and skip taking the cord with me because of the good battery life .
The battery life is excellent on this laptop .
The battery drains very fast compared to other laptops .
Strong build quality which really adds to its durability .
The keyboard is hard to type on and feels cheap .
The screen resolution is amazing and very crisp .
The service was terrible and the staff were rude and unhelpful .
The food was absolutely outstanding at this restaurant .
The staff was so nice and welcoming to everyone .
The price is too high for the quality you get .
Software support is poor and updates are buggy .
The camera takes great photos even in low light conditions .
The display quality is not good at all .
The processor speed is very fast for gaming .
Memory management is great and multitasking works well .
I love this laptop and use it every day for work .
The hotel room was clean and the staff were very helpful .
The ambiance was great and the food was delicious .
The delivery was slow and the packaging was damaged .
The battery performance exceeded my expectations .
The build quality feels premium and solid .
I hate the keyboard layout on this model .
The food quality is not consistent at all .
Not only was the food outstanding but the service was great too .
It took half an hour to get our check which was perfect since we could relax .
The tech support team was responsive and fixed my issue quickly .
The price point is reasonable for what you get with this device .
I have had my laptop for two weeks and it works perfectly .
The screen is vivid and colors are accurate for photo editing .
The wifi connection drops frequently and is unreliable .
The touchpad is very responsive and accurate .
The speakers produce decent sound for a laptop .
I was pleasantly surprised by the battery life on long trips .
The hinges feel loose and the lid wobbles when typing .
The USB ports are conveniently located on the left side .
The software comes pre-loaded with useful applications .
The weight is perfect for traveling and carrying around .
Customer service was very helpful in resolving my complaint .
The restaurant has a great atmosphere and good parking .
The menu has a wide variety of options to choose from .
I would highly recommend this product to anyone .
The design is sleek and modern looking .
The instructions were clear and easy to follow .
The material feels durable and high quality .
The color options are limited but the default looks good .
The charger cable is too short for convenient use .
The packaging arrived damaged but the product was fine .
The response time from support was impressive .
The taste was amazing and the portion size was perfect .
The waiter was attentive and the service was prompt .
""".strip()

# APC training data (3-line format: text with $T$, aspect, polarity)
_APC_TRAIN_DATA = """\
I charge it at night and skip taking the cord with me because of the good $T$ .
battery life
Positive
The $T$ drains very fast compared to other laptops .
battery
Negative
Strong $T$ though which really adds to its durability .
build quality
Positive
The $T$ is hard to type on and feels cheap .
keyboard
Negative
The $T$ resolution is amazing and very crisp .
screen
Positive
The $T$ was terrible and the staff were rude .
service
Negative
The food was absolutely outstanding at this $T$ .
restaurant
Positive
The $T$ was so nice and welcoming to everyone .
staff
Positive
The $T$ is too high for the quality you get .
price
Negative
Software $T$ is poor and updates are buggy .
support
Negative
The $T$ takes great photos even in low light .
camera
Positive
The display $T$ is not good at all .
quality
Negative"""

# TC training data (text $LABEL$ label format)
_TC_TRAIN_DATA = """\
the battery life is excellent on this laptop $LABEL$ Positive
the service was terrible and staff were rude $LABEL$ Negative
i love this laptop and use it every day for work $LABEL$ Positive
the food was absolutely outstanding at this restaurant $LABEL$ Positive
the keyboard is hard to type on and feels cheap $LABEL$ Negative
the display quality is not good at all $LABEL$ Negative
the price is reasonable for the quality you get $LABEL$ Positive
the build quality feels premium and solid $LABEL$ Positive
the wifi drops frequently and is unreliable $LABEL$ Negative
the camera takes great photos in low light $LABEL$ Positive
i hate the keyboard layout on this model $LABEL$ Negative
the staff was so nice and welcoming $LABEL$ Positive"""

# ATEPC training data (CoNLL/IOB format: word IOB_tag sentiment_label)
# -999 is the sentinel for non-aspect tokens
_ATEPC_TRAIN_DATA = """\
The O -999
battery B-ASP Positive
life I-ASP Positive
is O -999
excellent O -999
on O -999
this O -999
laptop O -999
. O -999

The O -999
keyboard B-ASP Negative
is O -999
hard O -999
to O -999
type O -999
on O -999
. O -999

The O -999
service B-ASP Negative
was O -999
terrible O -999
. O -999

The O -999
food B-ASP Positive
was O -999
absolutely O -999
outstanding O -999
. O -999

The O -999
staff B-ASP Positive
was O -999
so O -999
nice O -999
. O -999

The O -999
screen B-ASP Positive
resolution O -999
is O -999
amazing O -999
. O -999

The O -999
camera B-ASP Positive
takes O -999
great O -999
photos O -999
. O -999

The O -999
price B-ASP Negative
is O -999
too O -999
high O -999
. O -999

The O -999
build B-ASP Positive
quality I-ASP Positive
feels O -999
premium O -999
. O -999

Software O -999
support B-ASP Negative
is O -999
poor O -999
. O -999

The O -999
battery B-ASP Negative
drains O -999
very O -999
fast O -999
. O -999

The O -999
display B-ASP Negative
quality I-ASP Negative
is O -999
not O -999
good O -999
. O -999"""

# ---------------------------------------------------------------------------
# Model & dataset setup helpers
# ---------------------------------------------------------------------------

_REAL_MODEL_DIR = os.path.join(tempfile.gettempdir(), "pyabsa-real-bert")
_DATASET_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "integrated_datasets",
)

_APC_DATASET_ID = "999.RealModelTest"
_TC_DATASET_ID = "999.RealModelTestTC"
_ATEPC_DATASET_ID = "999.RealModelTestATEPC"


def _build_real_bert(model_dir: str) -> None:
    """Build a realistic BERT model with a proper English WordPiece tokenizer.

    The tokenizer is trained on *_ABSA_CORPUS* using the HuggingFace
    ``tokenizers`` library, giving a genuine subword vocabulary that handles
    all English test sentences without reducing everything to ``[UNK]``.

    Architecture:
        - vocab_size  : derived from the trained WordPiece vocabulary
        - hidden_size : 128   (small but realistic, 2× the toy model)
        - num_layers  : 2
        - num_heads   : 4
        - intermediate: 256
    """
    if os.path.exists(os.path.join(model_dir, "config.json")):
        return  # already built

    from tokenizers import BertWordPieceTokenizer
    from transformers import BertConfig, BertModel

    os.makedirs(model_dir, exist_ok=True)

    # --- Train a real WordPiece tokenizer from the ABSA corpus ---
    tmp_corpus = os.path.join(model_dir, "_corpus.txt")
    with open(tmp_corpus, "w") as fh:
        fh.write(_ABSA_CORPUS)

    tokenizer = BertWordPieceTokenizer(lowercase=True)
    tokenizer.train(
        files=[tmp_corpus],
        vocab_size=1000,
        min_frequency=1,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )
    vocab_size = tokenizer.get_vocab_size()
    tokenizer.save_model(model_dir)  # writes vocab.txt
    os.remove(tmp_corpus)

    # HuggingFace tokenizer_config.json so AutoTokenizer resolves to BertTokenizer
    tok_cfg = {
        "do_lower_case": True,
        "model_max_length": 128,
        "tokenizer_class": "BertTokenizer",
    }
    with open(os.path.join(model_dir, "tokenizer_config.json"), "w") as fh:
        json.dump(tok_cfg, fh)

    # --- Build and save the BERT model ---
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=128,
    )
    config.save_pretrained(model_dir)
    BertModel(config).save_pretrained(model_dir)


def _setup_apc_dataset(root: str, dataset_id: str) -> None:
    """Create a minimal APC dataset under *root*/integrated_datasets/."""
    num_id = dataset_id.split(".")[0]  # e.g. "999"
    data_dir = os.path.join(
        root, "apc_datasets", "990.TestSuite", dataset_id
    )
    os.makedirs(data_dir, exist_ok=True)
    for split, content in [("train", _APC_TRAIN_DATA), ("test", _APC_TRAIN_DATA)]:
        path = os.path.join(data_dir, f"{num_id}.{split}.apc")
        if not os.path.exists(path):
            with open(path, "w") as fh:
                fh.write(content)


def _setup_tc_dataset(root: str, dataset_id: str) -> None:
    """Create a minimal TC dataset under *root*/integrated_datasets/."""
    num_id = dataset_id.split(".")[0]
    data_dir = os.path.join(
        root, "tc_datasets", "990.TestSuite", dataset_id
    )
    os.makedirs(data_dir, exist_ok=True)
    for split, content in [("train", _TC_TRAIN_DATA), ("test", _TC_TRAIN_DATA)]:
        path = os.path.join(data_dir, f"{num_id}.{split}.tc")
        if not os.path.exists(path):
            with open(path, "w") as fh:
                fh.write(content)


def _setup_atepc_dataset(root: str, dataset_id: str) -> None:
    """Create a minimal ATEPC dataset under *root*/integrated_datasets/."""
    num_id = dataset_id.split(".")[0]
    data_dir = os.path.join(
        root, "atepc_datasets", "990.TestSuite", dataset_id
    )
    os.makedirs(data_dir, exist_ok=True)
    for split, content in [("train", _ATEPC_TRAIN_DATA), ("test", _ATEPC_TRAIN_DATA)]:
        path = os.path.join(data_dir, f"{num_id}.{split}.atepc")
        if not os.path.exists(path):
            with open(path, "w") as fh:
                fh.write(content)


# ---------------------------------------------------------------------------
# APC (Aspect Polarity Classification) – real model train + infer
# ---------------------------------------------------------------------------

APC_INFER_EXAMPLES = [
    "The [ASP]battery life[ASP] is excellent on this laptop .",
    "The [ASP]keyboard[ASP] is hard to type on and feels cheap .",
    "The [ASP]food[ASP] was absolutely outstanding .",
    "Strong [ASP]build quality[ASP] which really adds to its durability .",
]


def test_apc_real_model_train_and_infer():
    """Train APC with a realistic BERT model on a few examples, then infer.

    Uses a proper English WordPiece tokenizer (trained from actual ABSA text)
    and a 128-dim BERT architecture — substantially more realistic than the
    64-dim toy model in test_11.
    """
    _build_real_bert(_REAL_MODEL_DIR)
    _setup_apc_dataset(_DATASET_ROOT, _APC_DATASET_ID)

    from pyabsa import AspectPolarityClassification as APC
    from pyabsa import DeviceTypeOption, ModelSaveOption
    from pyabsa.utils.data_utils.dataset_item import DatasetItem

    config = APC.APCConfigManager.get_apc_config_english()
    config.model = APC.APCModelList.FAST_LSA_T_V2
    config.pretrained_bert = _REAL_MODEL_DIR
    config.num_epoch = 2
    config.max_seq_len = 64
    config.batch_size = 4
    config.evaluate_begin = 0
    config.log_step = -1
    config.cache_dataset = False

    trainer = APC.APCTrainer(
        config=config,
        dataset=DatasetItem(_APC_DATASET_ID),
        checkpoint_save_mode=ModelSaveOption.DO_NOT_SAVE_MODEL,
        auto_device=DeviceTypeOption.CPU,
    )
    classifier = trainer.load_trained_model()

    for example in APC_INFER_EXAMPLES:
        result = classifier.predict(example, print_result=False, ignore_error=False)
        assert result is not None, f"predict() returned None for: {example}"
        assert "sentiment" in result, f"'sentiment' key missing from result: {result}"
        assert result["sentiment"][0] in [
            "Positive",
            "Negative",
            "Neutral",
        ], f"Unexpected sentiment value: {result['sentiment']}"

    trainer.destroy()
    classifier.destroy()


# ---------------------------------------------------------------------------
# TC (Text Classification) – real model train + infer
# ---------------------------------------------------------------------------

TC_INFER_EXAMPLES = [
    "I love this laptop and use it every day for work .",
    "The keyboard is hard to type on and feels cheap .",
    "The food was absolutely outstanding at this restaurant .",
]


def test_tc_real_model_train_and_infer():
    """Train TC with a realistic BERT model on a few examples, then infer."""
    _build_real_bert(_REAL_MODEL_DIR)
    _setup_tc_dataset(_DATASET_ROOT, _TC_DATASET_ID)

    from pyabsa import TextClassification as TC
    from pyabsa import DeviceTypeOption, ModelSaveOption
    from pyabsa.utils.data_utils.dataset_item import DatasetItem

    config = TC.TCConfigManager.get_tc_config_english()
    config.model = TC.BERTTCModelList.BERT_MLP
    config.pretrained_bert = _REAL_MODEL_DIR
    config.num_epoch = 2
    config.max_seq_len = 64
    config.batch_size = 4
    config.evaluate_begin = 0
    config.log_step = -1
    config.cache_dataset = False

    trainer = TC.TCTrainer(
        config=config,
        dataset=DatasetItem(_TC_DATASET_ID),
        checkpoint_save_mode=ModelSaveOption.DO_NOT_SAVE_MODEL,
        auto_device=DeviceTypeOption.CPU,
    )
    classifier = trainer.load_trained_model()

    for example in TC_INFER_EXAMPLES:
        result = classifier.predict(example, print_result=False, ignore_error=False)
        assert result is not None, f"predict() returned None for: {example}"

    trainer.destroy()
    classifier.destroy()


# ---------------------------------------------------------------------------
# ATEPC (Aspect Term Extraction + Polarity) – real model train + infer
# ---------------------------------------------------------------------------

ATEPC_INFER_EXAMPLES = [
    "But the staff was so nice to us .",
    "The food is absolutely delicious and the service is great .",
    "The battery drains very fast and the keyboard feels cheap .",
]


def test_atepc_real_model_train_and_infer():
    """Train ATEPC with a realistic BERT model on a few examples, then infer."""
    _build_real_bert(_REAL_MODEL_DIR)
    _setup_atepc_dataset(_DATASET_ROOT, _ATEPC_DATASET_ID)

    from pyabsa import AspectTermExtraction as ATEPC
    from pyabsa import DeviceTypeOption, ModelSaveOption
    from pyabsa.utils.data_utils.dataset_item import DatasetItem

    config = ATEPC.ATEPCConfigManager.get_atepc_config_english()
    config.model = ATEPC.ATEPCModelList.FAST_LCF_ATEPC
    config.pretrained_bert = _REAL_MODEL_DIR
    config.num_epoch = 2
    config.max_seq_len = 64
    config.batch_size = 4
    config.evaluate_begin = 0
    config.log_step = -1
    config.cache_dataset = False
    config.cross_validate_fold = -1

    trainer = ATEPC.ATEPCTrainer(
        config=config,
        dataset=DatasetItem(_ATEPC_DATASET_ID),
        checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
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
    assert results is not None, "ATEPC predict() returned None"

    extractor.destroy()


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_apc_real_model_train_and_infer()
    print("APC passed.")
    test_tc_real_model_train_and_infer()
    print("TC passed.")
    test_atepc_real_model_train_and_infer()
    print("ATEPC passed.")
    print("All real-model training and inference tests passed.")
