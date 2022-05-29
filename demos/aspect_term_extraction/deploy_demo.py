# -*- coding: utf-8 -*-
# file: deploy_demo.py
# time: 2021/10/10
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import gradio as gr
import pandas as pd

from pyabsa import ATEPCCheckpointManager

aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='multilingual')


def inference(text):
    result = aspect_extractor.extract_aspect(inference_source=[text],
                                             pred_sentiment=True)

    result = pd.DataFrame({
        'aspect': result[0]['aspect'],
        'sentiment': result[0]['sentiment'],
        'position': result[0]['position']
    })

    return result


if __name__ == '__main__':
    iface = gr.Interface(
        fn=inference,
        inputs=["text"],
        examples=[
            ['Even though it is running Snow Leopard, 2.4 GHz C2D is a bit of an antiquated CPU and thus the occasional spinning '
             'wheel would appear when running Office Mac applications such as Word or Excel .'],
            ['从这门课程的内容丰富程度还有老师的授课及讨论区的答疑上来说，我都是很喜欢的。但是吧，我觉得每个章的内容太多了，三个学分的量就分在了上个章节三次小测'],
            ['このレストランのサービスはまあまあで,待ち時間は長かったが,料理はまずまずのものだった'],
            ['Die wartezeit war recht mittelmäßig, aber das Essen war befriedigend'],
            ['O serviço é médio, o tempo de espera é longo, mas os pratos são razoavelmente satisfatórios'],
            ['Dịch vụ của nhà hàng này rất trung bình và thời gian chờ đợi rất dài, nhưng món ăn thì khá là thỏa mãn'],
            ['Pelayanan di restoran biasa dan penantian yang lama, tetapi hasilnya cukup memuaskan'],
            ['Als je geen steak liefhebber bent is er een visalternatief, eend en lam aan aanvaardbare prijzen.'],
            ['سأوصي بالتأكيد بموقع المدينة القديمة إلا إنه عليك الحذر من الأسعار السياحية الأكثر ارتفاعاً'],
            ['Nous avons bien aimé l\'ambiance, sur la promenade principale de Narbonne-Plage, et la qualité du service.'],
            ['По поводу интерьера: место спокойное, шумных компаний нет (не было, по крайней мере, в момент нашего посещения), очень приятная и уютная атмосфера, все в лучших традициях.'],
            ['la calidad del producto, el servicio, el entorno todo fue excelente'],
            ['Yemekler iyi hos, lezzetler iyi ama heyecan verici bi taraflari yok, iyi bir baligi iyi bir sekilde izgara yapmak artik atla deve bi olay degil.'],
        ],
        outputs="dataframe",
        title='Multilingual Aspect Term Extraction for Short Texts (powered by PyABSA)',
        description='This demo is trained on the public and community shared datasets from ABSADatasets (https://github.com/yangheng95/ABSADatasets),'
                    ' please feel free to share your data to improve this work, To fit on your data, please train our ATEPC models on your own data,'
                    ' see the PyABSA (https://github.com/yangheng95/PyABSA/tree/release/demos/aspect_term_extraction)'
    )

    try:
        iface.launch(share=True)
    except:
        iface.launch()
