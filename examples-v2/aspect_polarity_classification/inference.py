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
sent_classifier = APC.SentimentClassifier('multilingual')
# sent_classifier = APC.SentimentClassifier('english')
# sent_classifier = APC.SentimentClassifier('chinese')

sent_classifier.predict('When I got home, there was a message on the machine because the owner realized that our [B-ASP]waitress[E-ASP] forgot to charge us for our wine. $LABEL$ Negative')

sent_classifier.predict(
    ['The [B-ASP]food[E-ASP] was good, but the [B-ASP]service[E-ASP] was terrible. $LABEL$ Positive, Negative',
     'The [B-ASP]food[E-ASP] was terrible, but the [B-ASP]service[E-ASP] was good. $LABEL$ Negative, Positive',]
)


inference_sets = APC.APCDatasetList.Restaurant14

results = sent_classifier.batch_predict(target_file=inference_sets,
                                        print_result=True,
                                        save_result=True,
                                        ignore_error=False,
                                        eval_batch_size=32,
                                        )

sent_classifier.destroy()
