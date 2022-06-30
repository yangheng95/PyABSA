# -*- coding: utf-8 -*-
# file: __init__.py.py
# time: 2021/8/9
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


from pyabsa.functional.trainer import APCTrainer, ATEPCTrainer, TCTrainer, TADTrainer, Trainer
from pyabsa.core.apc.models import (APCModelList,
                                    BERTBaselineAPCModelList,
                                    GloVeAPCModelList)
from pyabsa.core.tc.models import (GloVeTCModelList,
                                   BERTTCModelList)
from pyabsa.core.tad.models import (GloVeTADModelList,
                                    BERTTADModelList)
from pyabsa.core.atepc.models import ATEPCModelList

from pyabsa.functional.checkpoint.checkpoint_manager import (APCCheckpointManager,
                                                             ATEPCCheckpointManager,
                                                             TCCheckpointManager,
                                                             TADCheckpointManager,
                                                             available_checkpoints)
from pyabsa.functional.dataset import ABSADatasetList, TCDatasetList, AdvTCDatasetList
from pyabsa.functional.config import APCConfigManager
from pyabsa.functional.config import ATEPCConfigManager
from pyabsa.functional.config import TCConfigManager
from pyabsa.functional.config import TADConfigManager
from pyabsa.utils.file_utils import check_update_log, validate_datasets_version
from pyabsa.utils.pyabsa_utils import validate_pyabsa_version

# compatible for v1.14.3 and earlier versions
ClassificationDatasetList = TCDatasetList
TextClassifierCheckpointManager = TCCheckpointManager
GloVeClassificationModelList = GloVeTCModelList
BERTClassificationModelList = BERTTCModelList
ClassificationConfigManager = TCConfigManager
TextClassificationTrainer = TCTrainer
