Aspect Term Extraction and Classification
=========================================

### Inference

```python3
from pyabsa import AspectTermExtraction as ATEPC

# checkpoint_map = available_checkpoints(from_local=False)


aspect_extractor = ATEPC.AspectExtractor('multilingual',
                                         auto_device=True,  # False means load model on CPU
                                         cal_perplexity=True,
                                         )

inference_source = ATEPC.ATEPCDatasetList.SemEval
atepc_result = aspect_extractor.batch_predict(target_file=inference_source,  #
                                              save_result=True,
                                              print_result=True,  # print the result
                                              pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                              )

print(atepc_result)


```

### Train a model of aspect term extraction


```python3
import random

from pyabsa import AspectTermExtraction as ATEPC

config = ATEPC.ATEPCConfigManager.get_atepc_config_english()
config.model = ATEPC.ATEPCModelList.FAST_LCF_ATEPC
config.evaluate_begin = 0
config.max_seq_len = 512
config.pretrained_bert = 'yangheng/deberta-v3-base-absa'
config.l2reg = 1e-8
config.seed = random.randint(1, 100)
config.use_bert_spc = True
config.use_amp = False
config.cache_dataset = False

chinese_sets = ATEPC.ATEPCDatasetList.Multilingual

aspect_extractor = ATEPC.ATEPCTrainer(config=config,
                                      dataset=chinese_sets,
                                      checkpoint_save_mode=1,
                                      auto_device=True
                                      ).load_trained_model()

atepc_examples = ['But the staff was so nice to us .',
                  'But the staff was so horrible to us .',
                  r'Not only was the food outstanding , but the little ` perks \' were great .',
                  'It took half an hour to get our check , which was perfect since we could sit , have drinks and talk !',
                  'It was pleasantly uncrowded , the service was delightful , the garden adorable , '
                  'the food -LRB- from appetizers to entrees -RRB- was delectable .',
                  'How pretentious and inappropriate for MJ Grill to claim that it provides power lunch and dinners !'
                  ]
aspect_extractor.batch_predict(target_file=atepc_examples,  #
                               save_result=True,
                               print_result=True,  # print the result
                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                               )



```
