# -*- coding: utf-8 -*-
# file: deploy_demo.py
# time: 2021/10/10
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import gradio as gr
import pandas as pd

from pyabsa import APCCheckpointManager

sentiment_classifier = APCCheckpointManager.get_sentiment_classifier(checkpoint='multilingual',
                                                                     auto_device=True  # False means load model on CPU
                                                                     )


def inference(text):
    result = sentiment_classifier.infer(text=text,
                                        print_result=True,
                                        clear_input_samples=True)

    result = pd.DataFrame({
        'aspect': result[0]['aspect'],
        'sentiment': result[0]['sentiment'],
        'confidence': [round(c, 3) for c in result[0]['confidence']],
        'ref_sentiment': ['' if ref == '-999' else ref for ref in result[0]['ref_sentiment']],
        'is_correct': result[0]['ref_check'],
    })

    return result


if __name__ == '__main__':
    iface = gr.Interface(
        fn=inference,
        inputs=["text"],
        examples=[
            ['Strong build though which really adds to its [ASP]durability[ASP] .'],  # !sent! Positive
            ['Strong [ASP]build[ASP] though which really adds to its durability . !sent! Positive'],
            ['The [ASP]battery life[ASP] is excellent - 6-7 hours without charging . !sent! Positive'],
            ['I have had my [ASP]computer[ASP] for 2 weeks already and it [ASP]works[ASP] perfectly . !sent!  Positive, Positive'],
            ['And I may be the only one but I am really liking [ASP]Windows 8[ASP] . !sent! Positive'],
            ['This demo is trained on the laptop and restaurant and other review datasets from [ASP]ABSADatasets[ASP] (https://github.com/yangheng95/ABSADatasets)'],
            ['To fit on your data, please train the model on your own data, see the [ASP]PyABSA[ASP] (https://github.com/yangheng95/PyABSA)'],
        ],
        outputs="dataframe",
        title='Multilingual Aspect Sentiment Classification for Short Texts (powered by PyABSA)'
    )

    iface.launch(share=True)
