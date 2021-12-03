# -*- coding: utf-8 -*-
# file: extract_aspects.py
# time: 2021/5/27 0027
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
from pyabsa import ABSADatasetList, available_checkpoints
from pyabsa import ATEPCCheckpointManager

checkpoint_map = available_checkpoints(from_local=False)

examples = ['But the staff was so nice to us .',
            'But the staff was so horrible to us .',
            r'Not only was the food outstanding , but the little ` perks \' were great .',
            'It took half an hour to get our check , which was perfect since we could sit , have drinks and talk !',
            'It was pleasantly uncrowded , the service was delightful , the garden adorable , '
            'the food -LRB- from appetizers to entrees -RRB- was delectable .',
            'How pretentious and inappropriate for MJ Grill to claim that it provides power lunch and dinners !'
            ]

aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='english',
                                                               auto_device=False  # False means load model on CPU
                                                               )

inference_source = ABSADatasetList.SemEval
# inference_source = r'E:\PyABSA-Workspace\latest\PyABSA\examples\aspect_polarity_classification\integrated_datasets\datasets\apc_datasets\TShirt'
# inference_source = 'integrated_datasets/datasets/apc_datasets/TShirt'
# inference_source = ABSADatasetList.TShirt
# inference_source = examples
# inference_source = ABSADatasetList.Laptop14
atepc_result = aspect_extractor.extract_aspect(inference_source=inference_source,  #
                                               save_result=True,
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )

# print(atepc_result)
