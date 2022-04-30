# -*- coding: utf-8 -*-
# file: data_utils_for_training.py
# time: 2021/5/27 0027
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import warnings

import tqdm

from pyabsa.core.apc.dataset_utils.apc_utils import configure_spacy_model
from pyabsa.core.atepc.dataset_utils.atepc_utils import prepare_input_for_atepc
from pyabsa.utils.pyabsa_utils import check_and_fix_labels, SENTIMENT_PADDING, validate_example, check_and_fix_IOB_labels

Labels = set()


class InputExample(object):
    """A single training_tutorials/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, IOB_label=None, aspect_label=None, polarity=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence core, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair core.
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

    def __init__(self, input_ids_spc,
                 input_mask,
                 segment_ids,
                 label_id,
                 polarity=None,
                 valid_ids=None,
                 label_mask=None,
                 tokens=None,
                 lcf_cdm_vec=None,
                 lcf_cdw_vec=None
                 ):
        self.input_ids_spc = input_ids_spc
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.polarity = polarity
        self.tokens = tokens
        self.lcf_cdm_vec = lcf_cdm_vec
        self.lcf_cdw_vec = lcf_cdw_vec


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
        splits = line.strip().split(' ')
        if len(splits) != 3:
            print('warning! ignore detected error line(s) in input file:{}'.format(line))
            break
        sentence.append(splits[0])
        tag.append(splits[-2])
        polarity.append(splits[-1])
        Labels.add(splits[-2])

    prepared_data = []
    for s, t, p in data:

        if len(s) > 0:
            # prepare the atepc dataset, refer to https://github.com/yangheng95/PyABSA/issues/78
            polarity_padding = [str(SENTIMENT_PADDING)] * len(t)

            if len(Labels) > 3:
                # for more IOB labels support, but can not split cases in some praticular conditions, e.g., (B,I,E,O)
                for p_idx in range(len(p) - 1):
                    if (p[p_idx] != p[p_idx + 1] and p[p_idx] != str(SENTIMENT_PADDING) and p[p_idx + 1] != str(SENTIMENT_PADDING)) \
                        or (p[p_idx] != str(SENTIMENT_PADDING) and p[p_idx + 1] == str(SENTIMENT_PADDING)):
                        _p = p[:p_idx + 1] + polarity_padding[p_idx + 1:]
                        p = polarity_padding[:p_idx + 1] + p[p_idx + 1:]
                        prepared_data.append((s, t, _p))
            else:
                for t_idx in range(1, len(t)):
                    # for 3 IOB label (B, I, O)
                    if p[t_idx - 1] != str(SENTIMENT_PADDING) and split_aspect(t[t_idx - 1], t[t_idx]):
                        _p = p[:t_idx] + polarity_padding[t_idx:]
                        p = polarity_padding[:t_idx] + p[t_idx:]
                        prepared_data.append((s, t, _p))

                    if p[t_idx] != str(SENTIMENT_PADDING) and t_idx == len(t) - 1 and split_aspect(t[t_idx]):
                        _p = p[:t_idx + 1] + polarity_padding[t_idx + 1:]
                        p = polarity_padding[:t_idx + 1] + p[t_idx + 1:]
                        prepared_data.append((s, t, _p))

    return prepared_data


def split_aspect(tag1, tag2=None):
    if tag1 == 'B-ASP' and tag2 == 'B-ASP':
        return True
    if tag1 == 'B-ASP' and tag2 == 'O':
        return True
    elif tag1 == 'I-ASP' and tag2 == 'O':
        return True
    elif tag1 == 'I-ASP' and tag2 == 'B-ASP':
        return True
    elif (tag1 == 'B-ASP' or tag1 == 'I-ASP') and not tag2:
        return True
    elif tag1 == 'O' and tag2 == 'I-ASP':
        # warnings.warn('Invalid annotation! Found I-ASP without B-ASP')
        return False
    elif tag1 == 'O' and tag2 == 'O':
        return False
    elif tag1 == 'O' and tag2 == 'B-ASP':
        return False
    elif tag1 == 'O' and not tag2:
        return False
    elif tag1 == 'B-ASP' and tag2 == 'I-ASP':
        return False
    elif tag1 == 'I-ASP' and tag2 == 'I-ASP':
        return False
    else:
        return True
        # raise ValueError('Invalid IOB tag combination: {}, {}'.format(tag1, tag2))


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
        data = []
        for file in input_file:
            data += readfile(file)
        return data


class ATEPCProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.bos_token = tokenizer.bos_token if tokenizer.bos_token else '[CLS]'
        self.tokenizer.eos_token = tokenizer.eos_token if tokenizer.eos_token else '[SEP]'

    def get_train_examples(self, data_dir, set_tag):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_dir), set_tag)

    def get_test_examples(self, data_dir, set_tag):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_dir), set_tag)

    def get_labels(self):
        return sorted(list(Labels) + [self.tokenizer.bos_token, self.tokenizer.eos_token])

    def _create_examples(self, lines, set_type):
        examples = []

        for i, (sentence, tag, polarity) in enumerate(lines):
            aspect = []
            aspect_tag = []
            aspect_polarity = SENTIMENT_PADDING
            for w, t, p in zip(sentence, tag, polarity):
                if str(p) != str(SENTIMENT_PADDING):
                    aspect.append(w)
                    aspect_tag.append(t)
                    aspect_polarity = p

            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            text_b = aspect

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, IOB_label=tag,
                                         aspect_label=aspect_tag, polarity=aspect_polarity))

        return examples


def convert_examples_to_features(examples, max_seq_len, tokenizer, opt=None):
    """Loads a data file into a list of `InputBatch`s."""

    configure_spacy_model(opt)

    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    label_map = {label: i for i, label in enumerate(sorted(list(Labels) + [tokenizer.bos_token, tokenizer.eos_token]), 1)}
    opt.IOB_label_to_index = label_map
    features = []
    polarities_set = set()
    for (ex_index, example) in enumerate(tqdm.tqdm(examples, postfix='convert examples to features')):
        text_tokens = example.text_a[:]
        aspect_tokens = example.text_b[:]
        IOB_label = example.IOB_label
        aspect_label = example.aspect_label
        polarity = example.polarity
        if polarity != SENTIMENT_PADDING or int(polarity) != SENTIMENT_PADDING:  # bad case handle in Chinese atepc_datasets
            polarities_set.add(polarity)  # ignore samples without polarities
        tokens = []
        labels = []
        valid = []
        label_mask = []
        enum_tokens = [bos_token] + text_tokens + [eos_token] + aspect_tokens + [eos_token]
        IOB_label = [bos_token] + IOB_label + [eos_token] + aspect_label + [eos_token]

        aspect = ' '.join(example.text_b)
        try:
            text_left, _, text_right = [s.strip() for s in ' '.join(example.text_a).partition(aspect)]
        except:
            continue
        text_raw = text_left + ' ' + aspect + ' ' + text_right

        if validate_example(text_raw, aspect, polarity):
            continue

        prepared_inputs = prepare_input_for_atepc(opt, tokenizer, text_left, text_right, aspect)
        lcf_cdm_vec = prepared_inputs['lcf_cdm_vec']
        lcf_cdw_vec = prepared_inputs['lcf_cdw_vec']

        for i, word in enumerate(enum_tokens):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            cur_iob = IOB_label[i]
            for m in range(len(token)):
                if m == 0:
                    label_mask.append(1)
                    labels.append(cur_iob)
                    valid.append(1)
                else:
                    valid.append(0)
        tokens = tokens[0:min(len(tokens), max_seq_len - 2)]
        labels = labels[0:min(len(labels), max_seq_len - 2)]
        valid = valid[0:min(len(valid), max_seq_len - 2)]
        # segment_ids = [0] * len(example.text_a[:]) + [1] * (max_seq_len - len([0] * len(example.text_a[:])))
        # segment_ids = segment_ids[:max_seq_len]

        segment_ids = [0] * max_seq_len  # simply set segment_ids to all zeros
        label_ids = []

        for i, token in enumerate(tokens):
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])

        input_ids_spc = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids_spc)
        label_mask = [1] * len(label_ids)
        while len(input_ids_spc) < max_seq_len:
            input_ids_spc.append(0)
            input_mask.append(0)
            label_ids.append(0)
            label_mask.append(0)
            while len(valid) < max_seq_len:
                valid.append(1)
        while len(label_ids) < max_seq_len:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids_spc) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len
        assert len(label_ids) == max_seq_len
        assert len(valid) == max_seq_len
        assert len(label_mask) == max_seq_len

        features.append(
            InputFeatures(input_ids_spc=input_ids_spc,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          polarity=polarity,
                          valid_ids=valid,
                          label_mask=label_mask,
                          tokens=example.text_a,
                          lcf_cdm_vec=lcf_cdm_vec,
                          lcf_cdw_vec=lcf_cdw_vec)
        )
    check_and_fix_labels(polarities_set, 'polarity', features, opt)
    check_and_fix_IOB_labels(label_map, opt)
    opt.polarities_dim = len(polarities_set)

    return features
