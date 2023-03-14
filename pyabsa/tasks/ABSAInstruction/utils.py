import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
from transformers import (
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Trainer,
    Seq2SeqTrainer,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class T5Generator:
    def __init__(self, model_checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        self.device = "cuda" if torch.has_cuda else ("mps" if torch.has_mps else "cpu")
        self.model.to(self.device)

    def tokenize_function_inputs(self, sample):
        """
        Udf to tokenize the input dataset.
        """
        model_inputs = self.tokenizer(sample["text"], max_length=512, truncation=True)
        labels = self.tokenizer(sample["labels"], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train(self, tokenized_datasets, **kwargs):
        """
        Train the generative model.
        """
        # Set training arguments
        args = Seq2SeqTrainingArguments(**kwargs)

        # Define trainer object
        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"]
            if tokenized_datasets.get("test") is not None
            else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        print("\nModel training started ....")
        trainer.train()

        # Save best model
        trainer.save_model()
        return trainer

    def predict(self, text, **kwargs):
        """
        Predict the output from the model.
        """
        inputs = self.tokenizer(
            text, max_length=512, truncation=True, return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        outputs = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def get_labels(
        self,
        tokenized_dataset,
        trained_model_path=None,
        predictor=None,
        batch_size=4,
        sample_set="train",
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
                tokenized_dataset[sample_set],
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
                test_dataset=tokenized_dataset[sample_set]
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
        total_pred = 0
        total_gt = 0
        tp = 1e-6
        for gt, pred in zip(y_true, y_pred):
            gt_list = gt.split(", ")
            pred_list = pred.split(", ")
            total_pred += len(pred_list)
            total_gt += len(gt_list)
            for gt_val in gt_list:
                for pred_val in pred_list:
                    if pred_val.strip().lower() == gt_val.strip().lower():
                        tp += 1
        p = tp / total_pred
        r = tp / total_gt
        return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r)}

    def get_metrics(self, y_true, y_pred):
        total_pred = 1e-6
        total_gt = 1e-6
        true_aspects = []
        pred_aspects = []
        true_opinions = []
        pred_opinions = []
        true_sentiments = []
        pred_sentiments = []
        true_categories = []
        pred_categories = []
        for gt, pred in zip(y_true, y_pred):
            gt_list = gt.split(", ")
            pred_list = pred.split(", ")
            total_pred += len(pred_list)
            total_gt += len(gt_list)

            for gt_val in gt_list:
                for pred_val in pred_list:
                    try:
                        assert len(gt_val.split("|")) == 4
                        assert len(pred_val.split("|")) == 4
                        gt_val = gt_val.strip().lower()
                        pred_val = pred_val.strip().lower()
                        true_aspects.append(gt_val.split("|")[0])
                        pred_aspects.append(pred_val.split("|")[0])
                        true_opinions.append(gt_val.split("|")[1])
                        pred_opinions.append(pred_val.split("|")[1])
                        true_sentiments.append(gt_val.split("|")[2])
                        pred_sentiments.append(pred_val.split("|")[2])
                        true_categories.append(gt_val.split("|")[3])
                        pred_categories.append(pred_val.split("|")[3])
                    except Exception as e:
                        pass

        aspect_p, aspect_r, aspect_f1 = self.get_aspect_metrics(
            true_aspects, pred_aspects
        )
        opinion_p, opinion_r, opinion_f1 = self.get_aspect_metrics(
            true_opinions, pred_opinions
        )
        sentiment_p, sentiment_r, sentiment_f1 = self.get_aspect_metrics(
            true_sentiments, pred_sentiments
        )
        category_p, category_r, category_f1 = self.get_aspect_metrics(
            true_categories, pred_categories
        )

        return {
            "avg_precision": aspect_p + opinion_p + sentiment_p + category_p / 4,
            "avg_recall": aspect_r + opinion_r + sentiment_r + category_r / 4,
            "avg_f1": aspect_f1 + opinion_f1 + sentiment_f1 + category_f1 / 4,
            "aspect_p": aspect_p,
            "aspect_r": aspect_r,
            "aspect_f1": aspect_f1,
            "opinion_p": opinion_p,
            "opinion_r": opinion_r,
            "opinion_f1": opinion_f1,
            "sentiment_p": sentiment_p,
            "sentiment_r": sentiment_r,
            "sentiment_f1": sentiment_f1,
            "category_p": category_p,
            "category_r": category_r,
            "category_f1": category_f1,
        }


class T5Classifier:
    def __init__(self, model_checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint, force_download=True
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_checkpoint, force_download=True
        )
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)

    def tokenize_function_inputs(self, sample):
        """
        Udf to tokenize the input dataset.
        """
        sample["input_ids"] = self.tokenizer(
            sample["text"], max_length=512, truncation=True
        ).input_ids
        sample["labels"] = self.tokenizer(
            sample["labels"], max_length=128, truncation=True
        ).input_ids
        return sample

    def train(self, tokenized_datasets, **kwargs):
        """
        Train the generative model.
        """

        # Set training arguments
        args = Seq2SeqTrainingArguments(**kwargs)

        # Define trainer object
        trainer = Trainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"]
            if tokenized_datasets.get("test") is not None
            else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        print("\nModel training started ....")
        trainer.train()

        # Save best model
        trainer.save_model()
        return trainer

    def get_labels(
        self, tokenized_dataset, predictor=None, batch_size=4, sample_set="train"
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
                tokenized_dataset[sample_set],
                batch_size=batch_size,
                collate_fn=collate_fn,
            )
            predicted_output = []
            self.model.to(self.device)
            print("Model loaded to: ", self.device)

            for batch in tqdm(dataloader):
                batch = batch.to(self.device)
                output_ids = self.model.to.generate(batch)
                output_texts = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                for output_text in output_texts:
                    predicted_output.append(output_text)
        else:
            print("Prediction from trainer")
            pred_proba = predictor.predict(
                test_dataset=tokenized_dataset[sample_set]
            ).predictions[0]
            output_ids = np.argmax(pred_proba, axis=2)
            predicted_output = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
        return predicted_output

    def get_metrics(self, y_true, y_pred):
        cnt = 0
        for gt, pred in y_true, y_pred:
            if gt == pred:
                cnt += 1
        return cnt / len(y_true)
