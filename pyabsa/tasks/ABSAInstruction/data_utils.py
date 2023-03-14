import json

import findfile
from datasets import Dataset
from datasets.dataset_dict import DatasetDict


class DatasetLoader:
    def __init__(
        self,
        train_df_id,
        test_df_id,
        train_df_ood=None,
        test_df_ood=None,
        sample_size=1,
    ):
        self.train_df_id = train_df_id.sample(frac=sample_size, random_state=1999)
        self.test_df_id = test_df_id
        if train_df_ood is not None:
            self.train_df_ood = train_df_ood.sample(frac=sample_size, random_state=1999)
        else:
            self.train_df_ood = train_df_ood
        self.test_df_ood = test_df_ood

    def reconstruct_strings(self, df, col):
        """
        Reconstruct strings to dictionaries when loading csv/xlsx files.
        """
        reconstructed_col = []
        for text in df[col]:
            if text != "[]" and isinstance(text, str):
                text = (
                    text.replace("[", "")
                    .replace("]", "")
                    .replace("{", "")
                    .replace("}", "")
                    .split(", '")
                )
                req_list = []
                for idx, pair in enumerate(text):
                    if idx % 2 == 0:
                        reconstructed_dict = {}
                        reconstructed_dict[
                            pair.split(":")[0].replace("'", "")
                        ] = pair.split(":")[1].replace("'", "")
                    else:
                        reconstructed_dict[
                            pair.split(":")[0].replace("'", "")
                        ] = pair.split(":")[1].replace("'", "")
                        req_list.append(reconstructed_dict)
            else:
                req_list = text
            reconstructed_col.append(req_list)
        df[col] = reconstructed_col
        return df

    def create_data_for_multitask(self, df, bos_instruction="", eos_instruction=""):
        """
        Prepare the data in the input format required.
        """

        if df is None:
            return

        for text, data in df.iterrows():
            _labels = []
            for label in data["labels"]:
                _label = []
                for key in label:  # Path: data_prep.py
                    # if key != 'category':
                    #     _label.append(key + ':' + label[key])
                    _label.append(key + ":" + label[key])
                    # _label.append(label[key])
                _label = "|".join(_label)
                _labels.append(_label)
            df.at[text, "labels"] = ", ".join(_labels)

        df["text"] = df["text"].apply(lambda x: bos_instruction + x + eos_instruction)
        print(df.head(5))
        return df

    def set_data_for_training_semeval(self, tokenize_function):
        """
        Create the training and test dataset as huggingface datasets format.
        """
        # Define train and test sets
        if self.test_df_id is None:
            indomain_dataset = DatasetDict(
                {"train": Dataset.from_pandas(self.train_df_id)}
            )
        else:
            indomain_dataset = DatasetDict(
                {
                    "train": Dataset.from_pandas(self.train_df_id),
                    "test": Dataset.from_pandas(self.test_df_id),
                }
            )
        indomain_tokenized_datasets = indomain_dataset.map(
            tokenize_function, batched=True
        )

        if (self.train_df_ood is not None) and (self.test_df_ood is None):
            other_domain_dataset = DatasetDict(
                {"train": Dataset.from_pandas(self.train_df_id)}
            )
            other_domain_tokenized_dataset = other_domain_dataset.map(
                tokenize_function, batched=True
            )
        elif (self.train_df_ood is None) and (self.test_df_ood is not None):
            other_domain_dataset = DatasetDict(
                {"test": Dataset.from_pandas(self.train_df_id)}
            )
            other_domain_tokenized_dataset = other_domain_dataset.map(
                tokenize_function, batched=True
            )
        elif (self.train_df_ood is not None) and (self.test_df_ood is not None):
            other_domain_dataset = DatasetDict(
                {
                    "train": Dataset.from_pandas(self.train_df_ood),
                    "test": Dataset.from_pandas(self.test_df_ood),
                }
            )
            other_domain_tokenized_dataset = other_domain_dataset.map(
                tokenize_function, batched=True
            )
        else:
            other_domain_dataset = None
            other_domain_tokenized_dataset = None

        return (
            indomain_dataset,
            indomain_tokenized_datasets,
            other_domain_dataset,
            other_domain_tokenized_dataset,
        )


def read_json(data_path, data_type="train"):
    data = []

    files = findfile.find_files(data_path, [data_type, ".jsonl"])
    for f in files:
        with open(f, "r", encoding="utf8") as fin:
            for line in fin:
                data.append(json.loads(line))
    return data
