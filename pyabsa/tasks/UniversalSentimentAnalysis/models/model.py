import os

import autocuda
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from pyabsa.framework.checkpoint_class.checkpoint_template import CheckpointManager
from pyabsa.utils.file_utils.file_utils import save_model


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Customizing the loss function
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Assign higher weight to EOS tokens
        eos_token_id = self.tokenizer.eos_token_id
        eos_weight = 1.0  # Adjust this based on your requirements
        weights = torch.ones_like(labels, dtype=torch.float)
        weights[labels == eos_token_id] = eos_weight

        weighted_loss = (loss * weights.view(-1)).mean()

        return (weighted_loss, outputs) if return_outputs else weighted_loss


class GenerationModel:
    def __init__(self, config):
        self.config = config
        try:
            checkpoint = CheckpointManager().parse_checkpoint(
                self.config.pretrained_bert, "USA"
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        except Exception as e:
            print(e)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.pretrained_bert
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_bert)
        self.data_collator = DataCollatorForSeq2Seq(self.config.tokenizer)
        self.device = autocuda.auto_cuda()
        self.model.to(self.device)

    def train(self, tokenized_datasets, **kwargs):
        """
        Train the generative model.
        """
        # Set training arguments
        kwargs["output_dir"] = os.path.join(
            self.config.model_path_to_save, "transformers_usa_model/"
        )
        # Training arguments
        kwargs.update(
            {
                "evaluation_strategy": "epoch",
                "save_strategy": "epoch",
                "learning_rate": 5e-5,
                "per_device_train_batch_size": 8,
                "per_device_eval_batch_size": 8,
                "num_train_epochs": 3,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1,
                "load_best_model_at_end": True,
                "push_to_hub": False,
                "eval_accumulation_steps": 1,
                "predict_with_generate": True,
                "logging_steps": 1000000000,
                "use_mps_device": False,
                # 'fp16': True,
                "fp16": False,
            }
        )
        args = Seq2SeqTrainingArguments(**kwargs)

        # Define trainer object
        trainer = CustomSeq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"]
            if tokenized_datasets.get("test") is not None
            else None,
            tokenizer=self.config.tokenizer,
            data_collator=self.data_collator,
        )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        print("\nModel training started ....")
        trainer.train()

        if self.config.model_path_to_save:
            if not os.path.exists(self.config.model_path_to_save):
                os.makedirs(self.config.model_path_to_save)
            save_path = os.path.join(
                self.config.model_path_to_save, "pyabsa_usa_model/"
            )
            save_model(self.config, self.model, self.tokenizer, save_path)
        # Save best model
        trainer.save_model()
        return self.config.model_path_to_save

    def evaluate(
        self,
        tokenized_dataset,
        trained_model_path=None,
        predictor=None,
        batch_size=4,
        dataset_type="train",
    ):
        """
        Get the predictions from the trained model.
        """
        if not predictor:
            print("Prediction from checkpoint")

            def collate_fn(batch):
                input_ids = [torch.tensor(example["input_ids"]) for example in batch]
                input_ids = pad_sequence(
                    input_ids,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                )
                return input_ids

            dataloader = DataLoader(
                tokenized_dataset[dataset_type],
                batch_size=batch_size,
                collate_fn=collate_fn,
            )
            predicted_output = []
            self.model.to(self.device)
            print("Model loaded to: ", self.device)

            for batch in tqdm(dataloader):
                batch = batch.to(self.device)
                output_ids = self.model.generate(batch)
                output_texts = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                for output_text in output_texts:
                    predicted_output.append(output_text)
        else:
            print("Prediction from trainer")
            output_ids = predictor.predict(
                test_dataset=tokenized_dataset[dataset_type]
            ).predictions
            predicted_output = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
        return predicted_output

    def get_aspect_metrics(self, true_aspects, pred_aspects):
        aspect_p = precision_score(true_aspects, pred_aspects, average="macro")
        aspect_r = recall_score(true_aspects, pred_aspects, average="macro")
        aspect_f1 = f1_score(true_aspects, pred_aspects, average="macro")
        return aspect_p, aspect_r, aspect_f1

    def get_classic_metrics(self, y_true, y_pred):
        for i in range(len(y_true)):
            y_true[i] = y_true[i].replace(" ", "")
            y_pred[i] = y_pred[i].replace(" ", "")
            print(y_true[i])
            print(y_pred[i])
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro"),
            "recall": recall_score(y_true, y_pred, average="macro"),
            "f1": f1_score(y_true, y_pred, average="macro"),
        }
