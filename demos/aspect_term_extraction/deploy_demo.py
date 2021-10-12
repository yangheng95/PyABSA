# -*- coding: utf-8 -*-
# file: deploy_demo.py
# time: 2021/10/10
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import gradio as gr
import pandas as pd

from pyabsa import ATEPCCheckpointManager

aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='lcf_atepc_cdw_apcacc_80.94_apcf1_75.95_atef1_67.11.zip',
                                                               auto_device=True  # False means load model on CPU
                                                               )


def inference(text):
    result = aspect_extractor.extract_aspect(inference_source=[text],
                                             save_result=False,
                                             print_result=False,
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
        examples=[['the wine list is incredible and extensive and diverse , the food is all incredible and the staff was all very nice , good at their jobs and cultured .'],
                  ['中 药 学 这 门 课 程 总 体 设 计 非 常 好 简 洁 实 用 重 在 功 效 主 治 病 症 用 法 但 涉 及 的 药 物 数 量 少 只 介 绍 了 不 同 类 中 有 代 表 性 典 型 的 数 味 药 有 些 常 用 药 还 得 课 外 去 进 一 步 了 解'],
                  ['Though the menu includes some unorthodox offerings (a peanut butter roll, for instance), the classics are pure and great--we have never had better sushi anywhere, including Japan.'],
                  ['Everything , from the soft bread , soggy salad , and 50 minute wait time , with an incredibly rude service to deliver below average food .'],
                  ['Even though it is running Snow Leopard , 2.4 GHz C2D is a bit of an antiquated CPU and thus the occasional spinning wheel would appear when running Office Mac applications such as Word or Excel .'],
                  ],
        outputs="dataframe",
        theme='huggingface',
        title='Aspect Term Extraction & Polarity Classification'
    )

    iface.launch(share=True, )
