# -*- coding: utf-8 -*-
# file: __init__.py
# time: 2021/4/22 0022
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95

# Copyright (C) 2021. All Rights Reserved.


__version__ = '2.0.0.dev0'

__name__ = 'pyabsa'

from pyabsa.framework.flag_class import *

from pyabsa.utils.check_utils.package_version_check import validate_pyabsa_version, query_release_notes, check_pyabsa_update

from pyabsa.utils.data_utils.dataset_item import DatasetItem
from pyabsa.utils.absa_utils.make_absa_dataset import make_ABSA_dataset
from pyabsa.utils.absa_utils.absa_utils import generate_inference_set_for_apc, convert_apc_set_to_atepc_set
from pyabsa.utils.data_utils.dataset_manager import download_all_available_datasets, download_dataset_by_name
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file

from pyabsa.utils.check_utils.package_version_check import validate_pyabsa_version, query_release_notes, check_pyabsa_update, check_package_version
from pyabsa.framework.checkpoint_class.checkpoint_utils import available_checkpoints, download_checkpoint
from pyabsa.framework.dataset_class.dataset_dict_class import DatasetDict
from pyabsa.tasks import (
    AspectPolarityClassification,
    AspectTermExtraction,
    TextClassification,
    TextAdversarialDefense,
    RNAClassification,
    RNARegression
)

import pyabsa.tasks.AspectPolarityClassification as APC
import pyabsa.tasks.AspectTermExtraction as ATEPC
import pyabsa.tasks.TextClassification as TC
import pyabsa.tasks.TextAdversarialDefense as TAD
import pyabsa.tasks.RNAClassification as RNAC
import pyabsa.tasks.RNARegression as RNAR

validate_pyabsa_version()


from pyabsa.utils.notification_utils.notification_utils import check_emergency_notification
check_emergency_notification()
