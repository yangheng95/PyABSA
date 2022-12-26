Aspect Term Extraction
======================


### Inference Example of aspect term extraction

```python3
from pyabsa import AspectTermExtraction as ATEPC
from pyabsa import available_checkpoints, TaskCodeOption

checkpoint_map = available_checkpoints(
    task_code=TaskCodeOption.Aspect_Term_Extraction_and_Classification,
    show_ckpts=True
)

aspect_extractor = ATEPC.AspectExtractor('multilingual',
                                         auto_device=True,  # False means load model on CPU
                                         cal_perplexity=True,
                                         )

inference_source = ATEPC.ATEPCDatasetList.SemEval
ate_result = aspect_extractor.batch_predict(target_file=inference_source,  #
                                              save_result=True,
                                              print_result=True,  # print the result
                                              pred_sentiment=False, # Don't predict the sentiment of extracted aspect terms
                                              )

print(ate_result)

```
