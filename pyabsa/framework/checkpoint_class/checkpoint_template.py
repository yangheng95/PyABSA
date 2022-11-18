# -*- coding: utf-8 -*-
# file: checkpoint_template.py
# time: 2021/6/11 0011
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


import os
import sys
from pathlib import Path
from typing import Union

from findfile import find_file
from termcolor import colored

from pyabsa import TaskCodeOption
from pyabsa.framework.checkpoint_class.checkpoint_utils import available_checkpoints, download_checkpoint
from pyabsa.utils.file_utils.file_utils import unzip_checkpoint


class CheckpointManager:

    def parse_checkpoint(self,
                         checkpoint: Union[str, Path] = None,
                         task_code: str = TaskCodeOption.Aspect_Polarity_Classification
                         ) -> Path:
        """

        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from Google Drive
        This param is for someone wants to load a checkpoint not registered in PyABSA
        :param task_code: task code, e.g. apc, atepc, tad, rnac_datasets, rnar, tc, etc.

        :return:
        """
        if isinstance(checkpoint, str) or isinstance(checkpoint, Path):
            if os.path.exists(checkpoint):
                checkpoint_config = find_file(checkpoint, and_key=['.config'])
            else:
                checkpoint_config = find_file(os.getcwd(), [checkpoint, '.config'])
            if checkpoint_config:
                checkpoint = os.path.dirname(checkpoint_config)
            elif isinstance(checkpoint, str) and checkpoint.endswith('.zip'):
                checkpoint = unzip_checkpoint(
                    checkpoint if os.path.exists(checkpoint) else find_file(os.getcwd(), checkpoint))
            else:
                checkpoint = self._get_remote_checkpoint(checkpoint, task_code)
        return checkpoint

    def _get_remote_checkpoint(self, checkpoint: str = 'multilingual', task_code: str = None) -> Path:
        """
        download the checkpoint and return the path of the downloaded checkpoint
        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from Google Drive
        This param is for someone wants to load a checkpoint not registered in PyABSA
        :return:
        """

        available_checkpoint_by_task = available_checkpoints(task_code)
        if checkpoint.lower() in [k.lower() for k in available_checkpoint_by_task.keys()]:
            print(colored('Downloading checkpoint:{} ...'.format(checkpoint), 'green'))
        else:
            print(colored(
                'Checkpoint:{} is not found, you can raise an issue for requesting shares of checkpoints'.format(
                    checkpoint), 'red'))
            sys.exit(-1)
        return download_checkpoint(task=task_code,
                                   language=checkpoint.lower(),
                                   checkpoint=available_checkpoint_by_task[checkpoint.lower()])


class APCCheckpointManager(CheckpointManager):

    def __init__(self):
        super(APCCheckpointManager, self).__init__()

    @staticmethod
    def get_sentiment_classifier(checkpoint: Union[str, Path] = None, **kwargs):
        """
        This interface is used for compatibility with previous APIs
        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried
        """
        from pyabsa.tasks.AspectPolarityClassification import SentimentClassifier
        return SentimentClassifier(CheckpointManager().parse_checkpoint(checkpoint, TaskCodeOption.Aspect_Polarity_Classification))


class ATEPCCheckpointManager(CheckpointManager):
    def __init__(self):
        super(ATEPCCheckpointManager, self).__init__()

    @staticmethod
    def get_aspect_extractor(checkpoint: Union[str, Path] = None, **kwargs):
        """
        This interface is used for compatibility with previous APIs
        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried
        """
        from pyabsa.tasks.AspectTermExtraction import AspectExtractor
        return AspectExtractor(CheckpointManager().parse_checkpoint(checkpoint, TaskCodeOption.Aspect_Term_Extraction_and_Classification))


class TADCheckpointManager(CheckpointManager):
    def __init__(self):
        super(TADCheckpointManager, self).__init__()

    @staticmethod
    def get_tad_text_classifier(checkpoint: Union[str, Path] = None, **kwargs):
        """
        This interface is used for compatibility with previous APIs
        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried
        """
        from pyabsa.tasks.TextAdversarialDefense import TADTextClassifier
        return TADTextClassifier(CheckpointManager().parse_checkpoint(checkpoint, TaskCodeOption.Text_Adversarial_Defense))


class RNACCheckpointManager(CheckpointManager):
    def __init__(self):
        super(RNACCheckpointManager, self).__init__()

    @staticmethod
    def get_rna_classifier(checkpoint: Union[str, Path] = None, **kwargs):
        """
        This interface is used for compatibility with previous APIs
        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried
        """
        from pyabsa.tasks.RNAClassification import RNAClassifier
        return RNAClassifier(CheckpointManager().parse_checkpoint(checkpoint, TaskCodeOption.RNASequenceClassification))


class RNARCheckpointManager(CheckpointManager):
    def __init__(self):
        super(RNARCheckpointManager, self).__init__()

    @staticmethod
    def get_rna_regressor(checkpoint: Union[str, Path] = None, **kwargs):
        """
        This interface is used for compatibility with previous APIs
        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried
        """
        from pyabsa.tasks.RNARegression import RNARegressor
        return RNARegressor(CheckpointManager().parse_checkpoint(checkpoint, TaskCodeOption.RNASequenceRegression))


class TCCheckpointManager(CheckpointManager):
    def __init__(self):
        super(TCCheckpointManager, self).__init__()

    @staticmethod
    def get_text_classifier(checkpoint: Union[str, Path] = None, **kwargs):
        """
        This interface is used for compatibility with previous APIs
        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried
        """
        from pyabsa.tasks.TextClassification import TextClassifier
        return TextClassifier(CheckpointManager().parse_checkpoint(checkpoint, TaskCodeOption.Text_Classification))
