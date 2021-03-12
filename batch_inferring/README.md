# Batch inferring for APC task
We introduce the inferring process of BERT-based models and GloVe-based models in this section.

## Inferring Procedures of BERT-based models
* Step 1. Train a BERT-based model and save its state_dict. (You can save the fine-tuned BERT model and load it to infer, and skip Step 2)
* Step 2. Put the state_dict file in the 'batch_inferring' folder. Load the original BERT model which you chose to train your model, then load the state_dict using pytorch. 
* Step 3. Put the model config of inferring in the 'batch_inferring' folder, which should be the same as the config used in training process.
* Step 4. Construct the inferring dataset according to the sample. You should do it by yourself.
* Step 5. Run the batch_inferring script and see what happens.


## Inferring Procedures of GloVe-based models
* Step 1. Train a GloVe-based model and save its state_dict.
* Step 2. Put the state_dict file in the 'batch_inferring' folder, then load the state_dict using pytorch. 
* Step 3. Put the model config of inferring in the 'batch_inferring' folder, which should be the same as the config used in training process.
* Step 4. Put the '''tokenizer.dat''' in the 'batch_inferring' folder, which was used in the training process.
* Step 5. Put the embedding matrix generated during training process in the 'batch_inferring' folder. The embedding matrix is specific for any dataset.
* Step 6. Construct the inferring dataset according to the sample. You should do it by yourself.
* Step 7. Run the batch_inferring script.
 

### tip
You can run the batch_inferring script if all files are prepared in the 'batch_inferring' folder, the code will recognize each file. Only one is needed for all types of mentioned file.
