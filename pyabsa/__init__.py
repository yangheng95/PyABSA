# -*- coding: utf-8 -*-
# file: __init__.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95

# Copyright (C) 2021. All Rights Reserved.

__version__ = '1.9.4'
__name__ = 'pyabsa'

from termcolor import colored
from update_checker import UpdateChecker

from pyabsa.functional.trainer import APCTrainer, ATEPCTrainer, TextClassificationTrainer
from pyabsa.core.apc.models import (APCModelList,
                                    BERTBaselineAPCModelList,
                                    GloVeAPCModelList)
from pyabsa.core.tc.models import (GloVeClassificationModelList,
                                   BERTClassificationModelList)
from pyabsa.core.atepc.models import ATEPCModelList
from pyabsa.functional import (TextClassifierCheckpointManager,
                               APCCheckpointManager,
                               ATEPCCheckpointManager)
from pyabsa.functional.checkpoint.checkpoint_manager import (APCCheckpointManager,
                                                             ATEPCCheckpointManager,
                                                             available_checkpoints)
from pyabsa.functional.dataset import ABSADatasetList, ClassificationDatasetList
from pyabsa.functional.config import APCConfigManager
from pyabsa.functional.config import ATEPCConfigManager
from pyabsa.functional.config import ClassificationConfigManager
from pyabsa.utils.file_utils import check_update_log, check_dataset
from pyabsa.utils.pyabsa_utils import validate_version

validate_version()

check_dataset()

checker = UpdateChecker()
check_result = checker.check(__name__, __version__)

if check_result:
    print(check_result)
    check_update_log()
