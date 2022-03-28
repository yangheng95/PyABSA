# -*- coding: utf-8 -*-
# file: deploy_demo.py
# time: 2021/10/10
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import gradio as gr
import pandas as pd

from pyabsa import ATEPCCheckpointManager


class AspectExtractor:
    def __init__(self, checkpoint_name='english'):
        self.inference_model = ATEPCCheckpointManager.get_aspect_extractor(checkpoint=checkpoint_name,
                                                                           auto_device=True  # False means load model on CPU
                                                                           )

    def inference(self, text):
        result = self.inference_model.extract_aspect(inference_source=[text],
                                                     pred_sentiment=True)

        result = pd.DataFrame({
            'aspect': result[0]['aspect'],
            'sentiment': result[0]['sentiment']
        })

        return result


if __name__ == '__main__':
    aspect_extractor = AspectExtractor('english')
    iface = gr.Interface(
        fn=aspect_extractor.inference,
        inputs=["text"],
        examples=[
            ['the wine list is incredible and extensive and diverse , the food is all incredible and the staff was all very nice , good at their jobs and cultured .'],
            ['Though the menu includes some unorthodox offerings (a peanut butter roll, for instance), the classics are pure and great--we have never had better sushi anywhere, including Japan.'],
            ['Everything , from the soft bread , soggy salad , and 50 minute wait time , with an incredibly rude service to deliver below average food .'],
            ['Even though it is running Snow Leopard , 2.4 GHz C2D is a bit of an antiquated CPU and thus the occasional spinning wheel would appear when running Office Mac applications such as Word or Excel .'],
            # ['中药学这门课程总体设计非常好简洁实用重在功效主治病症用法但涉及的药物数量少只介绍了不同类中有代表性典型的数味药有些常用药还得课外去进一步了解'],
        ],
        live=True,
        outputs="dataframe",
        allow_flagging='auto',
        title='Aspect Term Extraction & Polarity Classification (English)'
    )

    iface.launch(share=True)
