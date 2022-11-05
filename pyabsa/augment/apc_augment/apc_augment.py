# -*- coding: utf-8 -*-
# file: apc_augment.py
# time: 02/11/2022 19:51
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

def __perform_apc_augmentation(dataset, **kwargs):
    print(colored('No augment datasets found, performing APC augment. This may take a long time...', 'yellow'))
    print(colored('The augment tool is available at: {}'.format('https://github.com/yangheng95/BoostTextAugmentation'), 'yellow'))
    from boost_aug import ABSCBoostAug

    config = APCConfigManager.get_apc_config_english()
    config.model = APCModelList.FAST_LCF_BERT

    BoostingAugmenter = ABSCBoostAug(ROOT=os.getcwd(),
                                     CLASSIFIER_TRAINING_NUM=1,
                                     AUGMENT_NUM_PER_CASE=10,
                                     WINNER_NUM_PER_CASE=8,
                                     device=autocuda.auto_cuda())

    # auto-trainer after augment
    BoostingAugmenter.apc_boost_augment(config,  # BOOSTAUG
                                        dataset,
                                        train_after_aug=True,
                                        rewrite_cache=True,
                                        )
    sys.exit(0)
