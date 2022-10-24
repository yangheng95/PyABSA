# -*- coding: utf-8 -*-
# file: __init__.py
# time: 2021/4/22 0022
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95

# Copyright (C) 2021. All Rights Reserved.


__version__ = '1.16.20'

__name__ = 'pyabsa'

from update_checker import UpdateChecker

from pyabsa.functional.trainer import APCTrainer, ATEPCTrainer, TCTrainer, TADTrainer
from pyabsa.core.apc.models import (APCModelList,
                                    BERTBaselineAPCModelList,
                                    GloVeAPCModelList)
from pyabsa.core.tc.models import (GloVeTCModelList,
                                   BERTTCModelList)
from pyabsa.core.tad.models import (GloVeTADModelList,
                                    BERTTADModelList)
from pyabsa.core.atepc.models import ATEPCModelList

from pyabsa.functional import (TCCheckpointManager,
                               TADCheckpointManager,
                               APCCheckpointManager,
                               ATEPCCheckpointManager,
                               )
from pyabsa.functional.checkpoint.checkpoint_manager import (APCCheckpointManager,
                                                             ATEPCCheckpointManager,
                                                             available_checkpoints)
from pyabsa.functional.dataset import ABSADatasetList, TCDatasetList, AdvTCDatasetList
from pyabsa.functional.config import APCConfigManager
from pyabsa.functional.config import ATEPCConfigManager
from pyabsa.functional.config import TCConfigManager
from pyabsa.functional.config import TADConfigManager
from pyabsa.utils.file_utils import check_update_log
from pyabsa.utils.pyabsa_utils import validate_pyabsa_version

from pyabsa.utils.make_dataset import make_ABSA_dataset

validate_pyabsa_version()

checker = UpdateChecker()
check_result = checker.check(__name__, __version__)

if check_result:
    print(check_result)
    check_update_log()
