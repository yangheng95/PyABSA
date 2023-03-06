# -*- coding: utf-8 -*-
# file: dataset_template.py
# time: 02/11/2022 15:44
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import torch
from torch.utils.data import Dataset


class PyABSADataset(Dataset):
    """
    Attributes
        data: a list of the loaded and preprocessed data samples.

    Methods
        __init__(self, config, tokenizer, dataset_type, **kwargs): constructs a new PyABSADataset object by loading and preprocessing a dataset based on the given configuration and dataset type. config is a configuration object containing the settings for loading and preprocessing the dataset, tokenizer is a pre-trained tokenizer object to tokenize the text data, and dataset_type is the type of the dataset to load (e.g., "train", "dev", "test"). Additional keyword arguments can be passed to customize the loading and preprocessing behavior.
        covert_to_tensor(data): a static method that converts the preprocessed data samples to PyTorch tensors.
        load_data_from_dict(self, dataset_dict, dataset_type, **kwargs): loads the dataset from a dictionary object containing the preprocessed data. dataset_dict is the dictionary object, dataset_type is the type of the dataset to load, and additional keyword arguments can be passed to customize the loading behavior.
        load_data_from_file(self, dataset_file, dataset_type, **kwargs): loads the dataset from a file containing the preprocessed data. dataset_file is the file path, dataset_type is the type of the dataset to load, and additional keyword arguments can be passed to customize the loading behavior.
        get_labels(self): returns a list of the labels for each data sample in the dataset.
        __len__(self): returns the number of data samples in the dataset.
        __str__(self): returns a string representation of the dataset.
        __repr__(self): returns a string representation of the dataset.

    """

    data = []

    def __init__(self, config, tokenizer, dataset_type, **kwargs):
        """
        PyABSADataset is a PyTorch Dataset class used for loading datasets for aspect-based sentiment analysis tasks.
        :param config: A configuration dict containing various settings for the dataset and the model.
        :param tokenizer: A tokenizer used to tokenize the texts in the dataset.
        :param dataset_type: The type of the dataset, which can be "train", "dev", or "test".
        :param kwargs: Additional arguments for loading the dataset, such as "text_column", "aspect_column", "label_column", "separator", and "data_path".
        """
        super(PyABSADataset, self).__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type

        if (
            self.config.get("dataset_dict")
            and dataset_type in self.config.dataset_dict
            and self.config.dataset_dict[dataset_type]
        ):
            self.load_data_from_dict(
                config.dataset_dict, dataset_type=dataset_type, **kwargs
            )
            self.data = self.covert_to_tensor(self.data)

        elif (
            self.config.get("dataset_file")
            and dataset_type in self.config.dataset_file
            and self.config.dataset_file[dataset_type]
        ):
            self.load_data_from_file(
                self.config.dataset_file, dataset_type=dataset_type, **kwargs
            )
            self.data = self.covert_to_tensor(self.data)
        self.data = self.data[
            : self.config.get("data_num", None)
            if self.config.get("data_num", None)
            else None
        ]
        if self.config.get("verbose", True):
            self.config.logger.info(
                "{} data examples:\n {}".format(dataset_type, self.data[:2])
            )

    @staticmethod
    def covert_to_tensor(data):
        """
        Convert the data in the dataset to PyTorch tensors.
        :param data: A list of dictionaries, where each dictionary represents a data sample.
        :return: The data in the dataset as PyTorch tensors.
        """
        for d in data:
            if isinstance(d, dict):
                for key, value in d.items():
                    try:
                        d[key] = torch.tensor(value)
                    except Exception as e:
                        pass
            elif isinstance(d, list):
                for value in d:
                    PyABSADataset.covert_to_tensor(value)
                PyABSADataset.covert_to_tensor(d)
        return data

    def load_data_from_dict(self, dataset_dict, dataset_type, **kwargs):
        """
        Load the dataset from a dictionary.
        :param dataset_dict: A dictionary containing the dataset.
        :param dataset_type: The type of the dataset, which can be "train", "dev", or "test".
        :param kwargs: Additional arguments for loading the dataset, such as "text_column", "aspect_column", "label_column", "separator", and "data_path".
        """
        data = []
        for text, aspect, label in zip(
            dataset_dict[dataset_type][kwargs["text_column"]],
            dataset_dict[dataset_type][kwargs["aspect_column"]],
            dataset_dict[dataset_type][kwargs["label_column"]],
        ):
            data.append(
                {
                    "text": text,
                    "aspect": aspect,
                    "label": label,
                }
            )
        self.data = data

    def load_data_from_file(self, dataset_file, dataset_type, **kwargs):
        """
        Load data from a file.

        :param dataset_file: The file to load data from.
        :param dataset_type: The type of dataset to load, e.g. "train", "test", "dev".
        :param kwargs: Optional additional arguments for loading data.
        """
        if dataset_type in dataset_file:
            self.data = dataset_file[dataset_type](
                self.config, self.tokenizer, **kwargs
            )

    def __getitem__(self, index):
        """
        Get a data sample from the dataset at a specific index.
        :param index: The index of the data sample to retrieve.
        :return: A dictionary representing a data sample, with keys "text", "aspect", and "label".
        """
        return self.data[index]

    def get_labels(self):
        """
        Get the labels of the data samples in the dataset.
        :return: A list of labels.
        """
        return [data["label"] for data in self.data]

    def __len__(self):
        """
        Get the number of data samples in the dataset.
        :return: The number of data samples in the dataset.
        """
        return len(self.data)

    def __str__(self):
        """
        Get a string representation of the dataset.
        :return: A string representing the dataset.
        """
        return f"PyABASDataset: {len(self.data)} samples"

    def __repr__(self):
        """
        Get a string representation of the dataset.
        :return: A string representing the dataset.
        """
        return self.__str__()
