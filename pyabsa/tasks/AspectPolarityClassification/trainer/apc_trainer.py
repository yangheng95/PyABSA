# -*- coding: utf-8 -*-
# file: apc_trainer.py
# time: 02/11/2022 21:34
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

from typing import Union

from pyabsa.framework.flag_class.flag_template import (
    DeviceTypeOption,
    ModelSaveOption,
    TaskCodeOption,
    TaskNameOption,
)
from pyabsa.framework.trainer_class.trainer_template import Trainer
from ..configuration.apc_configuration import APCConfigManager
from ..prediction.sentiment_classifier import SentimentClassifier
from ..instructor.apc_instructor import APCTrainingInstructor


class APCTrainer(Trainer):
    """Trainer entry point for Aspect Polarity Classification (APC).

    This wrapper connects configuration, datasets and the APC training
    instructor. After initialization, it triggers the standard training
    pipeline and exposes `load_trained_model()` (in the base trainer) to
    obtain a ready-to-use `SentimentClassifier` for inference.
    """
    def __init__(
        self,
        config: APCConfigManager = None,
        dataset=None,
        from_checkpoint: str = None,
        checkpoint_save_mode: int = ModelSaveOption.SAVE_MODEL_STATE_DICT,
        auto_device: Union[bool, str] = DeviceTypeOption.AUTO,
        path_to_save=None,
        load_aug=False,
    ):
        """Initialize the APC training workflow.

        Args:
            config: An `APCConfigManager` instance describing model, tokenizer
                and training hyperparameters.
            dataset: Dataset name, directory path, `DatasetItem`, or list of
                such entries; see docs for supported formats.
            from_checkpoint: Optional checkpoint path to resume training.
            checkpoint_save_mode: One of `ModelSaveOption.*` controlling what
                to save (state_dict, entire model, or fine-tuned PLM).
            auto_device: Device selection strategy or explicit device string.
            path_to_save: Directory to save checkpoints and logs.
            load_aug: Whether to load available augmentation datasets.

        Notes:
            After training, call `load_trained_model()` to obtain an
            inference-ready `SentimentClassifier`.
        """
        super(APCTrainer, self).__init__(
            config=config,
            dataset=dataset,
            from_checkpoint=from_checkpoint,
            checkpoint_save_mode=checkpoint_save_mode,
            auto_device=auto_device,
            path_to_save=path_to_save,
            load_aug=load_aug,
        )

        self.training_instructor = APCTrainingInstructor
        self.inference_model_class = SentimentClassifier
        self.config.task_code = TaskCodeOption.Aspect_Polarity_Classification
        self.config.task_name = TaskNameOption().get(
            TaskCodeOption.Aspect_Polarity_Classification
        )

        self._run()
