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
    def __init__(self):
        self.task_code = None

    def parse_checkpoint(self,
                         checkpoint: Union[str, Path] = None,
                         task_code: str = TaskCodeOption.Aspect_Polarity_Classification
                         ) -> Path:
        """

        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from Google Drive
        This param is for someone wants to load a checkpoint not registered in PyABSA
        :param task_code: task code, e.g. apc, atepc, tad, rnac, rnar, tc, etc.

        :return:
        """
        if isinstance(checkpoint, str) or isinstance(checkpoint, Path):
            self.task_code = task_code
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
                checkpoint = self._get_remote_checkpoint(checkpoint)
        return checkpoint

    def _get_remote_checkpoint(self, checkpoint: str = 'multilingual'):
        """
        download the checkpoint and return the path of the downloaded checkpoint
        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from Google Drive
        This param is for someone wants to load a checkpoint not registered in PyABSA
        :return:
        """

        available_checkpoint_by_task = available_checkpoints(self.task_code)
        if checkpoint.lower() in [k.lower() for k in available_checkpoint_by_task.keys()]:
            print(colored('Downloading checkpoint:{} ...'.format(checkpoint), 'green'))
        else:
            print(colored(
                'Checkpoint:{} is not found, you can raise an issue for requesting shares of checkpoints'.format(
                    checkpoint), 'red'))
            sys.exit(-1)
        return download_checkpoint(task=self.task_code,
                                   language=checkpoint.lower(),
                                   checkpoint=available_checkpoint_by_task[checkpoint.lower()])
