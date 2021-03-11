# Batch inferring for APC task
We introduce the inferring process of BERT-based models and GloVe-based models in this section.

# Inferring Procedures of BERT-based models
* Step 1. Train a BERT-based model and save its state_dict. (You can save the fine-tuned BERT model and load it to infer, and skip Step 2)
* Step 2. Put the state_dict file in the 'batch_inferring' folder. Load the original BERT model which you chose to train your model, then load the state_dict using pytorch. 
* Step 3. Put the model config of inferring in the 'batch_inferring' folder, which should be the same as the training process.
* Step 4. Construct the inferring dataset according to the sample by your self.
* Step 5. Run the batch_inferring script
### tip
You can run the batch_inferring script if all files are prepared in the 'batch_inferring' folder, the code will recognize the type of the files. Only one is needed of all types of file.

# Inferring Procedures of GloVe-based models
* Step 1. Train a BERT-based model and save its state_dict. (You can save the fine-tuned BERT model and load it to infer, and skip Step 2)
* Step 2. Put the state_dict file in the 'batch_inferring' folder. Load the original BERT model which you chose to train your model, then load the state_dict using pytorch. 
* Step 3. Put the model config of inferring in the 'batch_inferring' folder, which should be the same as the training process.
* Step 4. Put the tokenizer file in the 'batch_inferring' folder, which was used in the training process.
* Step 5. Put the generated embedding in the 'batch_inferring' folder. The embedding matrix is specific for any dataset, and it is the same matrix generated before training.
* Step 6. Construct the inferring dataset according to the sample by your self.
* Step 7. Run the batch_inferring script
### tip
You can run the batch_inferring script if all files are prepared in the 'batch_inferring' folder, the code will recognize the type of the files. Only one is needed of all types of file.

