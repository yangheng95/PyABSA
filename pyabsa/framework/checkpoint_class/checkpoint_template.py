# -*- coding: utf-8 -*-
# file: checkpoint_template.py
# time: 2021/6/11 0011
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


import os
import sys
from pathlib import Path
from typing import Union

from findfile import find_file
from termcolor import colored

from pyabsa import TaskCodeOption
from pyabsa.framework.checkpoint_class.checkpoint_utils import (
    available_checkpoints,
    download_checkpoint,
)
from pyabsa.utils.file_utils.file_utils import unzip_checkpoint
from pyabsa.utils.pyabsa_utils import fprint

from pyabsa.tasks.AspectPolarityClassification import SentimentClassifier
from pyabsa.tasks.AspectTermExtraction import AspectExtractor
from pyabsa.tasks.TextAdversarialDefense import TADTextClassifier
from pyabsa.tasks.RNAClassification import RNAClassifier
from pyabsa.tasks.RNARegression import RNARegressor
from pyabsa.tasks.TextClassification import TextClassifier
from pyabsa.tasks.AspectSentimentTripletExtraction import (
    AspectSentimentTripletExtractor,
)


class CheckpointManager:
    def parse_checkpoint(
        self,
        checkpoint: Union[str, Path] = None,
        task_code: str = TaskCodeOption.Aspect_Polarity_Classification,
    ) -> str:
        """
        Parse a given checkpoint file path or name and returns the path of the checkpoint directory.

        Args:
            checkpoint (Union[str, Path], optional): Zipped checkpoint name, checkpoint path, or checkpoint name queried from Google Drive. Defaults to None.
            task_code (str, optional): Task code, e.g. apc, atepc, tad, rnac_datasets, rnar, tc, etc. Defaults to TaskCodeOption.Aspect_Polarity_Classification.

        Returns:
            Path: The path of the checkpoint directory.

        Example:
            ```
            manager = CheckpointManager()
            checkpoint_path = manager.parse_checkpoint("checkpoint.zip", "apc")
            ```
        """
        if isinstance(checkpoint, str) or isinstance(checkpoint, Path):
            if os.path.exists(checkpoint):
                checkpoint_config = find_file(checkpoint, and_key=[".config"])
            else:
                checkpoint_config = find_file(os.getcwd(), [checkpoint, ".config"])
            if checkpoint_config:
                checkpoint = os.path.dirname(checkpoint_config)
            elif isinstance(checkpoint, str) and checkpoint.endswith(".zip"):
                checkpoint = unzip_checkpoint(
                    checkpoint
                    if os.path.exists(checkpoint)
                    else find_file(os.getcwd(), checkpoint)
                )
            else:
                checkpoint = self._get_remote_checkpoint(checkpoint, task_code)
        return checkpoint

    def _get_remote_checkpoint(
        self, checkpoint: str = "multilingual", task_code: str = None
    ) -> str:
        """
        Downloads a checkpoint file and returns the path of the downloaded checkpoint.

        Args:
            checkpoint (str, optional): Zipped checkpoint name, checkpoint path, or checkpoint name queried from Google Drive. Defaults to "multilingual".
            task_code (str, optional): Task code, e.g. apc, atepc, tad, rnac_datasets, rnar, tc, etc. Defaults to None.

        Returns:
            Path: The path of the downloaded checkpoint.

        Raises:
            SystemExit: If the given checkpoint file is not found.

        Example:
            ```
            manager = CheckpointManager()
            checkpoint_path = manager._get_remote_checkpoint("multilingual", "apc")
            ```
        """
        available_checkpoint_by_task = available_checkpoints(task_code)
        if checkpoint.lower() in [
            k.lower() for k in available_checkpoint_by_task.keys()
        ]:
            fprint(colored("Downloading checkpoint:{} ".format(checkpoint), "green"))
        else:
            fprint(
                colored(
                    "Checkpoint:{} is not found, you can raise an issue for requesting shares of checkpoints".format(
                        checkpoint
                    ),
                    "red",
                )
            )
            sys.exit(-1)
        return download_checkpoint(
            task=task_code,
            language=checkpoint.lower(),
            checkpoint=available_checkpoint_by_task[checkpoint.lower()],
        )


class ASTECheckpointManager(CheckpointManager):
    """
    This class manages the checkpoints for Aspect Sentiment Term Extraction.
    """

    def __init__(self):
        """
        Initializes an instance of the ASTECheckpointManager class.
        """
        super(ASTECheckpointManager, self).__init__()

    @staticmethod
    def get_aspect_sentiment_triplet_extractor(
        checkpoint: Union[str, Path] = None, **kwargs
    ) -> "AspectSentimentTripletExtractor":
        """
        Get an AspectExtractor object initialized with the given checkpoint for Aspect Sentiment Term Extraction.

        :param checkpoint: A string or Path object indicating the path to the checkpoint or a zip file containing the checkpoint.
            If the checkpoint is not registered in PyABSA, it should be the name of the checkpoint queried from Google Drive.
        :param kwargs: Additional keyword arguments to be passed to the AspectExtractor constructor.
        :return: An AspectExtractor object initialized with the given checkpoint.
        """
        return AspectSentimentTripletExtractor(
            CheckpointManager().parse_checkpoint(
                checkpoint, TaskCodeOption.Aspect_Sentiment_Triplet_Extraction
            )
        )


class APCCheckpointManager(CheckpointManager):
    def __init__(self):
        """
        Initializes an instance of the APCCheckpointManager class.
        """
        super(APCCheckpointManager, self).__init__()

    @staticmethod
    def get_sentiment_classifier(
        checkpoint: Union[str, Path] = None, **kwargs
    ) -> "SentimentClassifier":
        """
        Returns a pre-trained aspect sentiment classification model.

        Args:
            checkpoint (Union[str, Path], optional): A string specifying the path to a checkpoint or the name of a
                checkpoint registered in PyABSA. If `None`, the default checkpoint is used.
            **kwargs: Additional keyword arguments.

        Returns:
            SentimentClassifier: A pre-trained aspect sentiment classification model.

        Example:
            from pyabsa import APCCheckpointManager

            sentiment_classifier = APCCheckpointManager.get_sentiment_classifier()

        """

        return SentimentClassifier(
            CheckpointManager().parse_checkpoint(
                checkpoint, TaskCodeOption.Aspect_Polarity_Classification
            )
        )


class ATEPCCheckpointManager(CheckpointManager):
    """
    This class manages the checkpoints for Aspect Term Extraction and Polarity Classification.
    """

    def __init__(self):
        """
        Initializes an instance of the ATEPCCheckpointManager class.
        """
        super(ATEPCCheckpointManager, self).__init__()

    @staticmethod
    def get_aspect_extractor(
        checkpoint: Union[str, Path] = None, **kwargs
    ) -> "AspectExtractor":
        """
        Get an AspectExtractor object initialized with the given checkpoint for Aspect Term Extraction and Polarity Classification.

        :param checkpoint: A string or Path object indicating the path to the checkpoint or a zip file containing the checkpoint.
            If the checkpoint is not registered in PyABSA, it should be the name of the checkpoint queried from Google Drive.
        :param kwargs: Additional keyword arguments to be passed to the function.
        :return: An AspectExtractor object initialized with the given checkpoint.
        """

        return AspectExtractor(
            CheckpointManager().parse_checkpoint(
                checkpoint, TaskCodeOption.Aspect_Term_Extraction_and_Classification
            )
        )


class TADCheckpointManager(CheckpointManager):
    """
    This class manages the checkpoints for text adversarial defense.
    """

    def __init__(self):
        """
        Initializes an instance of the TADCheckpointManager class.
        """
        super(TADCheckpointManager, self).__init__()

    def get_tad_text_classifier(
        checkpoint: Union[str, Path] = None, **kwargs
    ) -> "TADTextClassifier":
        """
        Return a TADTextClassifier object initialized with the specified checkpoint.

        Args:
            checkpoint (Union[str, Path], optional): The path to the checkpoint, the name of the zipped checkpoint, or
                the name of the checkpoint queried from Google Drive. Defaults to None.

        Returns:
            TADTextClassifier: A TADTextClassifier object initialized with the given checkpoint.
        """

        return TADTextClassifier(
            CheckpointManager().parse_checkpoint(
                checkpoint, TaskCodeOption.Text_Adversarial_Defense
            )
        )


class RNACCheckpointManager(CheckpointManager):
    """
    This class manages the checkpoints for RNA sequence classification.
    """

    def __init__(self):
        """
        Initializes an instance of the RNACCheckpointManager class.
        """
        super(RNACCheckpointManager, self).__init__()

    @staticmethod
    def get_rna_classifier(
        checkpoint: Union[str, Path] = None, **kwargs
    ) -> "RNAClassifier":
        """
        This method returns an instance of the RNAClassifier class with a parsed checkpoint for RNA sequence classification.

        Args:
            checkpoint (Union[str, Path], optional): The name of the zipped checkpoint or the path to the checkpoint file. If not provided, the default checkpoint will be used. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            RNAClassifier: An instance of the RNAClassifier class with a parsed checkpoint for RNA sequence classification.

        Raises:
            ValueError: If the provided checkpoint is not found.
        """

        return RNAClassifier(
            CheckpointManager().parse_checkpoint(
                checkpoint, TaskCodeOption.RNASequenceClassification
            )
        )


class RNARCheckpointManager(CheckpointManager):
    """
    This class manages the checkpoints for RNA sequence regression.
    """

    def __init__(self):
        """
        Initializes an instance of the RNARCheckpointManager class.
        """
        super(RNARCheckpointManager, self).__init__()

    @staticmethod
    def get_rna_regressor(
        checkpoint: Union[str, Path] = None, **kwargs
    ) -> "RNARegressor":
        """
        Loads a pre-trained checkpoint for RNA sequence regression and returns an instance of the RNARegressor class
        that is ready to make predictions.

        :param checkpoint: (Optional) The name of a zipped checkpoint file, the path to a checkpoint file, or the name of
            a checkpoint file that can be found in Google Drive. If `checkpoint` is not provided, the default checkpoint
            for RNA sequence regression will be loaded.
        :type checkpoint: Union[str, Path]

        :return: An instance of the RNARegressor class that has been initialized with the specified checkpoint file.
        :rtype: RNARegressor
        """

        return RNARegressor(
            CheckpointManager().parse_checkpoint(
                checkpoint, TaskCodeOption.RNASequenceRegression
            )
        )


class TCCheckpointManager(CheckpointManager):
    """
    This class manages the checkpoints for text classification.
    """

    def __init__(self):
        """
        Initializes an instance of the TCCheckpointManager class.
        """
        super(TCCheckpointManager, self).__init__()

    @staticmethod
    def get_text_classifier(
        checkpoint: Union[str, Path] = None, **kwargs
    ) -> "TextClassifier":
        """
        Returns a TextClassifier instance loaded with a pre-trained checkpoint for text classification.

        Args:
            checkpoint (Union[str, Path], optional): The name of a zipped checkpoint file, a path to a checkpoint file,
                or the name of a checkpoint registered in PyABSA. If None, the latest version of the default checkpoint
                will be used. Defaults to None.
            **kwargs: Additional keyword arguments. Not used in this method.

        Returns:
            TextClassifier: A TextClassifier instance loaded with the specified checkpoint.
        """

        return TextClassifier(
            CheckpointManager().parse_checkpoint(
                checkpoint, TaskCodeOption.Text_Classification
            )
        )
