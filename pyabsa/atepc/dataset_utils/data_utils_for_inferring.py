# -*- coding: utf-8 -*-
# file: data_utils_for_inferring.py
# time: 2021/5/27 0027
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

SENTIMENT_PADDING = -999


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, IOB_label=None, aspect_label=None, polarity=None):
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
        self.IOB_label = IOB_label
        self.aspect_label = aspect_label
        self.polarity = polarity


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids_spc, input_mask, segment_ids, label_id, polarities=None, valid_ids=None,
                 label_mask=None, tokens=None):
        self.input_ids_spc = input_ids_spc
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.polarities = polarities
        self.tokens = tokens


def parse_example(example):
    tokens = []
    for token in example.split():
        tokens.append(token)
    return [(tokens, ['$NA$'] * len(tokens), [SENTIMENT_PADDING] * len(tokens))]


def parse_examples(examples):
    data = []
    for example in examples:
        tokens = []
        for token in example.split():
            tokens.append(token)
        data.append((tokens, ['$NA$'] * len(tokens), [SENTIMENT_PADDING] * len(tokens)))
    return data


class ATEPCProcessor:
    """Processor for the CoNLL-2003 data set."""

    def get_examples_for_aspect_extraction(self, examples):
        """See base class."""
        return self._create_examples(parse_examples(examples)
                                     if isinstance(examples, list)
                                     else parse_example(examples))

    def get_examples_for_sentiment_classification(self, extraction_result):
        """See base class."""
        return self._create_examples(extraction_result)

    def get_labels(self):
        return ["O", "B-ASP", "I-ASP", "[CLS]", "[SEP]"]

    def _create_examples(self, lines):
        examples = []
        for i, (sentence, tag, polarity) in enumerate(lines):
            examples.append(InputExample(guid=str(i), text_a=sentence, text_b=[], IOB_label=tag,
                                         aspect_label=[], polarity=polarity))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_len, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}
    features = []
    for (ex_index, example) in enumerate(examples):
        text_tokens = example.text_a
        IOB_label = example.IOB_label
        polaritiylist = example.polarity
        tokens = []
        labels = []
        polarities = []
        valid = []
        label_mask = []
        enum_tokens = text_tokens
        label_lists = IOB_label
        for i, word in enumerate(enum_tokens):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = label_lists[i]
            polarity_1 = polaritiylist[i]
            for m in range(len(token)):
                if m == 0:
                    label_mask.append(1)
                    labels.append(label_1)
                    valid.append(1)
                    polarities.append(polarity_1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_len - 1:
            tokens = tokens[0:(max_seq_len - 2)]
            polarities = polarities[0:(max_seq_len - 2)]
            labels = labels[0:(max_seq_len - 2)]
            valid = valid[0:(max_seq_len - 2)]
            label_mask = label_mask[0:(max_seq_len - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map["[CLS]"])
        polarities.insert(0, SENTIMENT_PADDING)
        # label_ids.append(label_map["O"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(0)
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        polarities.append(SENTIMENT_PADDING)
        # label_ids.append(label_map["O"])
        input_ids_spc = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids_spc)
        label_mask = [1] * len(label_ids)
        while len(input_ids_spc) < max_seq_len:
            input_ids_spc.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            label_mask.append(0)
            # while len(valid) < max_seq_len:
            valid.append(1)
        while len(label_ids) < max_seq_len:
            label_ids.append(0)
            label_mask.append(0)
        while len(polarities) < max_seq_len:
            polarities.append(SENTIMENT_PADDING)
        assert len(input_ids_spc) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len
        assert len(label_ids) == max_seq_len
        assert len(valid) == max_seq_len
        assert len(label_mask) == max_seq_len

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
                          label_mask=label_mask,
                          tokens=text_tokens))
    return features
