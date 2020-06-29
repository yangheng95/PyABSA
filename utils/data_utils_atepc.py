# -*- coding: utf-8 -*-
# file: data_utils_atepc.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.


import os


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, sentence_label=None, aspect_label=None, polarity=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.sentence_label = sentence_label
        self.aspect_label = aspect_label
        self.polarity = polarity


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids_spc, input_mask, segment_ids, label_id, polarities=None, valid_ids=None,
                 label_mask=None):
        self.input_ids_spc = input_ids_spc
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.polarities = polarities


def readfile(filename):
    '''
    read file
    '''
    f = open(filename, encoding='utf8')
    data = []
    sentence = []
    tag = []
    polarity = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, tag, polarity))
                sentence = []
                tag = []
                polarity = []
            continue
        splits = line.split(' ')
        if len(splits) != 3:
            print('warning! detected error line(s) in input file:{}'.format(line))
        sentence.append(splits[0])
        tag.append(splits[-2])
        polarity.append(int(splits[-1][:-1]))

    if len(sentence) > 0:
        data.append((sentence, tag, polarity))
    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class ATEPCProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        if 'laptop' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "Laptops.atepc.train.log.dat")), "train")
        elif 'rest' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "Restaurants.atepc.train.log.dat")), "train")
        elif 'twitter' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "twitter.atepc.train.log.dat")), "train")
        elif 'car' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "car.atepc.train.log.dat")), "train")
        elif 'phone' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "phone.atepc.train.log.dat")), "train")
        elif 'camera' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "camera.atepc.train.log.dat")), "train")
        elif 'notebook' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "notebook.atepc.train.log.dat")), "train")
        elif 'mixed' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "mixed.atepc.train.log.dat")), "train")

    # def get_dev_examples(self, data_dir):
    #     """See base class."""
    #     if 'laptop' in data_dir:
    #         return self._create_examples(
    #             self._read_tsv(os.experiments.join(data_dir, "Laptops.atepc.valid.dat")), "valid")
    #     elif 'rest' in data_dir:
    #         return self._create_examples(
    #             self._read_tsv(os.experiments.join(data_dir, "Restaurants.atepc.valid.dat")), "valid")
    #     else:
    #         return self._create_examples(
    #             self._read_tsv(os.experiments.join(data_dir, "twitter.atepc.valid.dat")), "valid")

    def get_test_examples(self, data_dir):
        """See base class."""
        if 'laptop' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "Laptops.atepc.test.dat")), "test")
        elif 'rest' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "Restaurants.atepc.test.dat")), "test")
        elif 'twitter' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "twitter.atepc.test.dat")), "test")
        elif 'car' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "car.atepc.test.dat")), "test")
        elif 'phone' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "phone.atepc.test.dat")), "test")
        elif 'camera' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "camera.atepc.test.dat")), "test")
        elif 'notebook' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "notebook.atepc.test.dat")), "test")
        elif 'mixed' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "mixed.atepc.test.dat")), "test")

    def get_labels(self):
        return ["O", "B-ASP", "I-ASP", "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, tag, polarity) in enumerate(lines):
            # aspect = ['[SEP]']
            # aspect_tag = ['O']
            aspect = []
            aspect_tag = []
            aspect_polarity = [-1]
            for w, t, p in zip(sentence, tag, polarity):
                if p != -1:
                    aspect.append(w)
                    aspect_tag.append(t)
                    aspect_polarity.append(-1)
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            text_b = aspect

            polarity.extend(aspect_polarity)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, sentence_label=tag,
                                         aspect_label=aspect_tag, polarity=polarity))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    for (ex_index, example) in enumerate(examples):
        text_spc_tokens = example.text_a
        aspect_tokens = example.text_b
        sentence_label = example.sentence_label
        aspect_label = example.aspect_label
        polaritiylist = example.polarity
        tokens = []
        labels = []
        polarities = []
        valid = []
        label_mask = []
        text_spc_tokens.extend(['[SEP]'])
        text_spc_tokens.extend(aspect_tokens)
        enum_tokens = text_spc_tokens
        sentence_label.extend(['[SEP]'])
        # sentence_label.extend(['O'])
        sentence_label.extend(aspect_label)
        label_lists = sentence_label
        # if len(enum_tokens) != len(label_lists):
        #     print(enum_tokens)
        #     print(label_lists)
        for i, word in enumerate(enum_tokens):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = label_lists[i]
            polarity_1 = polaritiylist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    polarities.append(polarity_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            polarities = polarities[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map["[CLS]"])
        # label_ids.append(label_map["O"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        # label_ids.append(label_map["O"])
        input_ids_spc = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids_spc)
        label_mask = [1] * len(label_ids)
        # import numpy as np
        while len(input_ids_spc) < max_seq_length:
            input_ids_spc.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        while len(polarities) < max_seq_length:
            polarities.append(-1)
        assert len(input_ids_spc) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        # if ex_index < 5:
        #     print("*** Example ***")
        #     print("guid: %s" % (example.guid))
        #     print("tokens: %s" % " ".join(
        #             [str(x) for x in ntokens]))
        #     print("input_ids: %s" % " ".join([str(x) for x in input_ids_spc]))
        #     print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     # print("label: %s (id = %d)" % (example.label, label_ids))
        #
        # input_ids_spc = np.array(input_ids_spc)
        # label_ids = np.array(label_ids)
        # labels = np.array(labels)
        # valid = np.array(valid)

        features.append(
            InputFeatures(input_ids_spc=input_ids_spc,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          polarities=polarities,
                          valid_ids=valid,
                          label_mask=label_mask))
    return features
