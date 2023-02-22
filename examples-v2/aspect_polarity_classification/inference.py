# -*- coding: utf-8 -*-
# file: ensemble_classification_inference.py
# time: 05/11/2022 19:48
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

from pyabsa import AspectPolarityClassification as APC

# sent_classifier = APC.SentimentClassifier('fast_lcf_bert_Multilingual_acc_82.66_f1_82.06.zip')
# sent_classifier = APC.SentimentClassifier('multilingual', auto_device=False)
sent_classifier = APC.SentimentClassifier("multilingual")
# sent_classifier = APC.SentimentClassifier('english')
# sent_classifier = APC.SentimentClassifier('chinese')

examples = [
    "The [B-ASP]food[E-ASP] was good, but the [B-ASP]service[E-ASP] was terrible. $LABEL$ Positive, Negative",
    "The [B-ASP]food[E-ASP] was terrible, but the [B-ASP]service[E-ASP] was good. $LABEL$ Negative, Positive",
    "The [B-ASP]food[E-ASP] was so-so, and the [B-ASP]service[E-ASP] was terrible. $LABEL$ Neutral, Negative",
]

for ex in examples:
    sent_classifier.predict(
        text=ex,
        print_result=True,
        ignore_error=True,  # ignore an invalid example, if it is False, invalid examples will raise Exceptions
        eval_batch_size=32,
    )

sent_classifier.predict(examples)

# inference_sets = APC.APCDatasetList.Restaurant14
#
# results = sent_classifier.batch_predict(
#     target_file=inference_sets,
#     print_result=True,
#     save_result=True,
#     ignore_error=False,
#     eval_batch_size=32,
# )
#
# sent_classifier.destroy()
