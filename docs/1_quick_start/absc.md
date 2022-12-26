Aspect-based Sentiment Classification
=====================================

### Inference with ABSC checkpoints

from pyabsa import AspectPolarityClassification as APC

```python3
from pyabsa import AspectPolarityClassification as APC
from pyabsa import available_checkpoints
# You can query available checkpoints by:
checkpoint_map = available_checkpoints(show_ckpts=False)

# use the latest checkpoint by checkpoint name
sent_classifier = APC.SentimentClassifier('multilingual')
# or use the latest checkpoint by following code
# sent_classifier = APC.SentimentClassifier('fast_lcf_bert_Multilingual_acc_82.66_f1_82.06.zip') # from a zip file
# sent_classifier = APC.SentimentClassifier('fast_lcf_bert_Multilingual_acc_82.66_f1_82.06') # from a folder
# sent_classifier = APC.SentimentClassifier('fast_lcf_bert_Multilingual') # search a folder by keyword 'fast_lcf_bert_Multilingual

sent_classifier.predict(
    'When I got home, there was a message on the machine because the owner realized that our [B-ASP]waitress[E-ASP] forgot to charge us for our wine. $LABEL$ Negative')

sent_classifier.predict(
    ['The [B-ASP]food[E-ASP] was good, but the [B-ASP]service[E-ASP] was terrible. $LABEL$ Positive, Negative',
     'The [B-ASP]food[E-ASP] was terrible, but the [B-ASP]service[E-ASP] was good. $LABEL$ Negative, Positive', ]
)

inference_sets = APC.APCDatasetList.Restaurant14

results = sent_classifier.batch_predict(target_file=inference_sets,
                                        print_result=True,
                                        save_result=True,
                                        ignore_error=False, # this option will ignore the error of the inference source
                                        eval_batch_size=32,
                                        )

sent_classifier.destroy()

```

### Train your own ABSC model

```python3
import random

from pyabsa.tasks.AspectPolarityClassification import APCDatasetList

from pyabsa import AspectPolarityClassification as APC
from pyabsa import ModelSaveOption, DeviceTypeOption
import warnings

warnings.filterwarnings('ignore')

for dataset in [
    APCDatasetList.Laptop14,
    APCDatasetList.Restaurant14,
    APCDatasetList.Restaurant15,
    APCDatasetList.Restaurant16,
    APCDatasetList.MAMS
]:
    for model in [
        APC.APCModelList.FAST_LSA_T_V2,
        APC.APCModelList.FAST_LSA_S_V2,
        APC.APCModelList.BERT_SPC_V2,
    ]:
        for pretrained_bert in [
            'microsoft/deberta-v3-base',
            # 'roberta-base',
        ]:
            config = APC.APCConfigManager.get_apc_config_english()
            config.model = model
            config.pretrained_bert = pretrained_bert
            config.evaluate_begin = 0
            config.max_seq_len = 80
            config.num_epoch = 30
            config.log_step = -1  # alias of evaluate_step, -1 means evaluate per epoch
            config.dropout = 0.5
            config.eta = -1
            config.eta_lr = 0.001
            # config.lcf = 'fusion'
            config.cache_dataset = False
            config.l2reg = 1e-8
            config.learning_rate = 1e-5
            config.use_amp = True  # use automatic mixed precision training, may cause convergence problem
            config.use_bert_spc = True
            config.lsa = True
            config.use_torch_compile = False
            config.seed = [random.randint(0, 10000) for _ in range(3)]  # the number of seed is equal to the number of trials

            APC.APCTrainer(config=config,
                           dataset=dataset,
                           # from_checkpoint='english',
                           checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
                           # checkpoint_save_mode=ModelSaveOption.DO_NOT_SAVE_MODEL,
                           auto_device=DeviceTypeOption.AUTO,
                           ).destroy()
```