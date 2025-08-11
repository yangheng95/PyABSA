# -*- coding: utf-8 -*-
# file: aste_trainer.py
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
from ..configuration.configuration import USAConfigManager
from ..instructor.instructor import USATrainingInstructor
from ..prediction.predictor import USAPredictor


class USATrainer(Trainer):
    """Trainer entry point for Universal Sentiment Analysis (USA).

    Orchestrates the training of the sequence-to-sequence USA model via the
    task-specific training instructor. After initialization, it launches the
    standard training routine; use `load_trained_model()` to obtain a
    `USAPredictor` for inference.
    """

    def __init__(
        self,
        config: USAConfigManager = None,
        dataset=None,
        from_checkpoint: str = None,
        checkpoint_save_mode: int = ModelSaveOption.SAVE_MODEL_STATE_DICT,
        auto_device: Union[bool, str] = DeviceTypeOption.AUTO,
        path_to_save=None,
        load_aug=False,
    ):
        """Initialize the USA training workflow.

        Args:
            config: A `USAConfigManager` describing model and trainer options.
            dataset: Dataset name, directory path, `DatasetItem`, or list.
            from_checkpoint: Optional checkpoint to resume training.
            checkpoint_save_mode: What to save after/between epochs.
            auto_device: Device strategy or explicit device string.
            path_to_save: Directory for checkpoints.
            load_aug: Whether to include augmentation datasets.

        Notes:
            After training, call `load_trained_model()` to obtain a
            `USAPredictor` for inference.
        """
        super(USATrainer, self).__init__(
            config=config,
            dataset=dataset,
            from_checkpoint=from_checkpoint,
            checkpoint_save_mode=checkpoint_save_mode,
            auto_device=auto_device,
            path_to_save=path_to_save,
            load_aug=load_aug,
        )

        self.training_instructor = USATrainingInstructor
        self.inference_model_class = USAPredictor
        self.config.task_code = TaskCodeOption.Universal_Sentiment_Analysis
        self.config.task_name = TaskNameOption().get(self.config.task_code)

        self._run()
