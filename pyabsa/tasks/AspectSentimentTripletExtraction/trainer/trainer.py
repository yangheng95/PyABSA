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
from ..configuration.configuration import ASTEConfigManager
from ..instructor.instructor import ASTETrainingInstructor
from ..prediction.predictor import AspectSentimentTripletExtractor


class ASTETrainer(Trainer):
    """Trainer entry point for Aspect Sentiment Triplet Extraction (ASTE).

    Coordinates configuration, datasets, and the ASTE training instructor.
    After initialization, it launches the standard training routine. Use
    `load_trained_model()` to obtain an `AspectSentimentTripletExtractor`
    for inference.
    """
    def __init__(
        self,
        config: ASTEConfigManager = None,
        dataset=None,
        from_checkpoint: str = None,
        checkpoint_save_mode: int = ModelSaveOption.SAVE_MODEL_STATE_DICT,
        auto_device: Union[bool, str] = DeviceTypeOption.AUTO,
        path_to_save=None,
        load_aug=False,
    ):
        """Initialize the ASTE training workflow.

        Args:
            config: An `ASTEConfigManager` describing model and trainer options.
            dataset: Dataset name, directory path, `DatasetItem`, or list.
            from_checkpoint: Optional checkpoint path to resume training.
            checkpoint_save_mode: What to save after/between epochs.
            auto_device: Device strategy or explicit device string.
            path_to_save: Directory to store checkpoints.
            load_aug: Whether to include augmentation datasets.

        Notes:
            After training, call `load_trained_model()` to obtain an
            `AspectSentimentTripletExtractor` for inference.
        """
        super(ASTETrainer, self).__init__(
            config=config,
            dataset=dataset,
            from_checkpoint=from_checkpoint,
            checkpoint_save_mode=checkpoint_save_mode,
            auto_device=auto_device,
            path_to_save=path_to_save,
            load_aug=load_aug,
        )

        self.training_instructor = ASTETrainingInstructor
        self.inference_model_class = AspectSentimentTripletExtractor
        self.config.task_code = TaskCodeOption.Aspect_Sentiment_Triplet_Extraction
        self.config.task_name = TaskNameOption().get(self.config.task_code)

        self._run()
