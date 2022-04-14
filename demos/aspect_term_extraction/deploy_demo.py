# -*- coding: utf-8 -*-
# file: deploy_demo.py
# time: 2021/10/10
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import gradio as gr
import pandas as pd

from pyabsa import ATEPCCheckpointManager

aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='multilingual2',
                                                               auto_device=True  # False means load model on CPU
                                                               )


def inference(text):
    result = aspect_extractor.extract_aspect(inference_source=[text],
                                             pred_sentiment=True)

    result = pd.DataFrame({
        'aspect': result[0]['aspect'],
        'sentiment': result[0]['sentiment']
    })

    return result


if __name__ == '__main__':
    iface = gr.Interface(
        fn=inference,
        inputs=["text"],
        examples=[['The wine list is incredible and extensive and diverse , the food is all incredible and the staff was all very nice ,'
                   ' good at their jobs and cultured .'],
                  ['Though the menu includes some unorthodox offerings (a peanut butter roll, for instance), the classics are pure and '
                   'great--we have never had better sushi anywhere, including Japan.'],
                  ['Everything, from the soft bread, soggy salad, and 50 minute wait time, with an incredibly rude service to deliver'
                   ' below average food .'],
                  ['Even though it is running Snow Leopard, 2.4 GHz C2D is a bit of an antiquated CPU and thus the occasional spinning '
                   'wheel would appear when running Office Mac applications such as Word or Excel .'],
                  ['This demo is trained on the laptop and restaurant and other review datasets from ABSADatasets (https://github.com/yangheng95/ABSADatasets)'],
                  ['To fit on your data, please train the model on your own data, see the PyABSA (https://github.com/yangheng95/PyABSA)'],
                  ],
        outputs="dataframe",
        title='Aspect Term Extraction for Short Texts (powered by PyABSA)'
    )

    iface.launch()
