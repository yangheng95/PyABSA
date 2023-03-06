# -*- coding: utf-8 -*-
# file: aste_utils.py
# time: 05/03/2023
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
import json

import math
import pickle

import torch
from collections import OrderedDict, defaultdict

from sklearn import metrics
from transformers import BertTokenizer

label = ["N", "B-A", "I-A", "A", "B-O", "I-O", "O", "Negative", "Neutral", "Positive"]

label2id, id2label = OrderedDict(), OrderedDict()
for i, v in enumerate(label):
    label2id[v] = i
    id2label[i] = v


def get_spans(tags):
    """for BIO tag"""
    tags = tags.strip().split()
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith("B"):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith("O"):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


def get_evaluate_spans(tags, length, token_range):
    """for BIO tag"""
    spans = []
    start = -1
    for i in range(length):
        l, r = token_range[i]
        if tags[l] == -1:
            continue
        elif tags[l] == 1:
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[l] == 0:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


class Instance(object):
    def __init__(
        self,
        tokenizer,
        sentence_pack,
        post_vocab,
        deprel_vocab,
        postag_vocab,
        synpost_vocab,
        config,
    ):
        self.id = sentence_pack["id"]
        self.sentence = sentence_pack["sentence"]
        self.tokens = self.sentence.strip().split()
        self.postag = sentence_pack["postag"]
        self.head = sentence_pack["head"]
        self.deprel = sentence_pack["deprel"]
        self.sen_length = len(self.tokens)
        self.token_range = []
        self.text_ids = tokenizer.encode(
            self.sentence,
            padding="do_not_pad",
            max_length=config.max_seq_len,
            truncation=True,
        )

        self.length = len(self.text_ids)
        self.bert_tokens_padding = torch.zeros(config.max_seq_len).long()
        self.aspect_tags = torch.zeros(config.max_seq_len).long()
        self.opinion_tags = torch.zeros(config.max_seq_len).long()
        self.tags = torch.zeros(config.max_seq_len, config.max_seq_len).long()
        self.tags_symmetry = torch.zeros(config.max_seq_len, config.max_seq_len).long()
        self.mask = torch.zeros(config.max_seq_len)

        for i in range(self.length):
            self.bert_tokens_padding[i] = self.text_ids[i]
        self.mask[: self.length] = 1

        token_start = 1
        for i, w in enumerate(self.tokens):
            token_end = token_start + len(
                tokenizer.encode(
                    w,
                    padding="do_not_pad",
                    max_length=config.max_seq_len,
                    truncation=True,
                    add_special_tokens=False,
                )
            )
            self.token_range.append([token_start, token_end - 1])
            token_start = token_end
        assert self.length == self.token_range[-1][-1] + 2

        self.aspect_tags[self.length :] = -1
        self.aspect_tags[0] = -1
        self.aspect_tags[self.length - 1] = -1

        self.opinion_tags[self.length :] = -1
        self.opinion_tags[0] = -1
        self.opinion_tags[self.length - 1] = -1

        self.tags[:, :] = -1
        self.tags_symmetry[:, :] = -1
        for i in range(1, self.length - 1):
            for j in range(i, self.length - 1):
                self.tags[i][j] = 0

        for triple in sentence_pack["triples"]:
            aspect = triple["target_tags"]
            opinion = triple["opinion_tags"]
            aspect_span = get_spans(aspect)
            opinion_span = get_spans(opinion)

            """set tag for aspect"""
            for l, r in aspect_span:
                start = self.token_range[l][0]
                end = self.token_range[r][1]
                for i in range(start, end + 1):
                    for j in range(i, end + 1):
                        if j == start:
                            self.tags[i][j] = label2id["B-A"]
                        elif j == i:
                            self.tags[i][j] = label2id["I-A"]
                        else:
                            self.tags[i][j] = label2id["A"]

                for i in range(l, r + 1):
                    set_tag = 1 if i == l else 2
                    al, ar = self.token_range[i]
                    self.aspect_tags[al] = set_tag
                    self.aspect_tags[al + 1 : ar + 1] = -1
                    """mask positions of sub words"""
                    self.tags[al + 1 : ar + 1, :] = -1
                    self.tags[:, al + 1 : ar + 1] = -1

            """set tag for opinion"""
            for l, r in opinion_span:
                start = self.token_range[l][0]
                end = self.token_range[r][1]
                for i in range(start, end + 1):
                    for j in range(i, end + 1):
                        if j == start:
                            self.tags[i][j] = label2id["B-O"]
                        elif j == i:
                            self.tags[i][j] = label2id["I-O"]
                        else:
                            self.tags[i][j] = label2id["O"]

                for i in range(l, r + 1):
                    set_tag = 1 if i == l else 2
                    pl, pr = self.token_range[i]
                    self.opinion_tags[pl] = set_tag
                    self.opinion_tags[pl + 1 : pr + 1] = -1
                    self.tags[pl + 1 : pr + 1, :] = -1
                    self.tags[:, pl + 1 : pr + 1] = -1

            for al, ar in aspect_span:
                for pl, pr in opinion_span:
                    for i in range(al, ar + 1):
                        for j in range(pl, pr + 1):
                            sal, sar = self.token_range[i]
                            spl, spr = self.token_range[j]
                            self.tags[sal : sar + 1, spl : spr + 1] = -1
                            if config.task == "pair":
                                if i > j:
                                    self.tags[spl][sal] = 7
                                else:
                                    self.tags[sal][spl] = 7
                            elif config.task == "triplet":
                                if i > j:
                                    self.tags[spl][sal] = label2id[triple["sentiment"]]
                                else:
                                    self.tags[sal][spl] = label2id[triple["sentiment"]]

        for i in range(1, self.length - 1):
            for j in range(i, self.length - 1):
                self.tags_symmetry[i][j] = self.tags[i][j]
                self.tags_symmetry[j][i] = self.tags_symmetry[i][j]

        """1. generate position index of the word pair"""
        self.word_pair_position = torch.zeros(
            config.max_seq_len, config.max_seq_len
        ).long()
        for i in range(len(self.tokens)):
            start, end = self.token_range[i][0], self.token_range[i][1]
            for j in range(len(self.tokens)):
                s, e = self.token_range[j][0], self.token_range[j][1]
                for row in range(start, end + 1):
                    for col in range(s, e + 1):
                        self.word_pair_position[row][col] = post_vocab.stoi.get(
                            abs(row - col), post_vocab.unk_index
                        )

        """2. generate deprel index of the word pair"""
        self.word_pair_deprel = torch.zeros(
            config.max_seq_len, config.max_seq_len
        ).long()
        for i in range(len(self.tokens)):
            start = self.token_range[i][0]
            end = self.token_range[i][1]
            for j in range(start, end + 1):
                s, e = (
                    self.token_range[self.head[i] - 1] if self.head[i] != 0 else (0, 0)
                )
                for k in range(s, e + 1):
                    self.word_pair_deprel[j][k] = deprel_vocab.stoi.get(self.deprel[i])
                    self.word_pair_deprel[k][j] = deprel_vocab.stoi.get(self.deprel[i])
                    self.word_pair_deprel[j][j] = deprel_vocab.stoi.get("self")

        """3. generate POS tag index of the word pair"""
        self.word_pair_pos = torch.zeros(config.max_seq_len, config.max_seq_len).long()
        for i in range(len(self.tokens)):
            start, end = self.token_range[i][0], self.token_range[i][1]
            for j in range(len(self.tokens)):
                s, e = self.token_range[j][0], self.token_range[j][1]
                for row in range(start, end + 1):
                    for col in range(s, e + 1):
                        self.word_pair_pos[row][col] = postag_vocab.stoi.get(
                            tuple(sorted([self.postag[i], self.postag[j]]))
                        )

        """4. generate synpost index of the word pair"""
        self.word_pair_synpost = torch.zeros(
            config.max_seq_len, config.max_seq_len
        ).long()
        tmp = [[0] * len(self.tokens) for _ in range(len(self.tokens))]
        for i in range(len(self.tokens)):
            j = self.head[i]
            if j == 0:
                continue
            tmp[i][j - 1] = 1
            tmp[j - 1][i] = 1

        tmp_dict = defaultdict(list)
        for i in range(len(self.tokens)):
            for j in range(len(self.tokens)):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)

        word_level_degree = [[4] * len(self.tokens) for _ in range(len(self.tokens))]

        for i in range(len(self.tokens)):
            node_set = set()
            word_level_degree[i][i] = 0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    word_level_degree[i][j] = 1
                    node_set.add(j)
                for k in tmp_dict[j]:
                    if k not in node_set:
                        word_level_degree[i][k] = 2
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                word_level_degree[i][g] = 3
                                node_set.add(g)

        for i in range(len(self.tokens)):
            start, end = self.token_range[i][0], self.token_range[i][1]
            for j in range(len(self.tokens)):
                s, e = self.token_range[j][0], self.token_range[j][1]
                for row in range(start, end + 1):
                    for col in range(s, e + 1):
                        self.word_pair_synpost[row][col] = synpost_vocab.stoi.get(
                            word_level_degree[i][j], synpost_vocab.unk_index
                        )

    def get_data(self):
        return {
            "id": self.id,
            "sentence": self.sentence,
            "sen_length": self.sen_length,
            "token_range": self.token_range,
            "bert_tokens_padding": self.bert_tokens_padding,
            "length": self.length,
            "mask": self.mask,
            "aspect_tags": self.aspect_tags,
            "opinion_tags": self.opinion_tags,
            "tags": self.tags,
            "tags_symmetry": self.tags_symmetry,
            "word_pair_position": self.word_pair_position,
            "word_pair_deprel": self.word_pair_deprel,
            "word_pair_pos": self.word_pair_pos,
            "word_pair_synpost": self.word_pair_synpost,
        }


def load_data_instances(
    sentence_packs, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, config
):
    instances = list()
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_bert)
    for sentence_pack in sentence_packs:
        instances.append(
            Instance(
                tokenizer,
                sentence_pack,
                post_vocab,
                deprel_vocab,
                postag_vocab,
                synpost_vocab,
                config,
            )
        )
    return instances


def load_tokens(filename):
    with open(filename, encoding="utf-8") as infile:
        data = json.load(infile)
        tokens = []
        deprel = []
        postag = []
        postag_ca = []

        max_len = 0
        for d in data:
            sentence = d["sentence"].split()
            tokens.extend(sentence)
            deprel.extend(d["deprel"])
            postag_ca.extend(d["postag"])
            # postag.extend(d['postag'])
            n = len(d["postag"])
            tmp_pos = []
            for i in range(n):
                for j in range(n):
                    tup = tuple(sorted([d["postag"][i], d["postag"][j]]))
                    tmp_pos.append(tup)
            postag.extend(tmp_pos)

            max_len = max(len(sentence), max_len)
    print(
        "{} tokens from {} examples loaded from {}.".format(
            len(tokens), len(data), filename
        )
    )
    return tokens, deprel, postag, postag_ca, max_len


class Metric:
    def __init__(
        self,
        config,
        predictions,
        goldens,
        bert_lengths,
        sen_lengths,
        tokens_ranges,
    ):
        self.config = config
        self.predictions = predictions
        self.goldens = goldens
        self.bert_lengths = bert_lengths
        self.sen_lengths = sen_lengths
        self.tokens_ranges = tokens_ranges
        self.ignore_index = -1
        self.data_num = len(self.predictions)

    def get_spans(self, tags, length, token_range, type):
        spans = []
        start = -1
        for i in range(length):
            l, r = token_range[i]
            if tags[l][l] == self.ignore_index:
                continue
            elif tags[l][l] == type:
                if start == -1:
                    start = i
            elif tags[l][l] != type:
                if start != -1:
                    spans.append([start, i - 1])
                    start = -1
        if start != -1:
            spans.append([start, length - 1])
        return spans

    def find_pair(self, tags, aspect_spans, opinion_spans, token_ranges):
        pairs = []
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                tag_num = [0] * 4
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        a_start = token_ranges[i][0]
                        o_start = token_ranges[j][0]
                        if al < pl:
                            tag_num[int(tags[a_start][o_start])] += 1
                        else:
                            tag_num[int(tags[o_start][a_start])] += 1
                if tag_num[3] == 0:
                    continue
                sentiment = -1
                pairs.append([al, ar, pl, pr, sentiment])
        return pairs

    def find_triplet(self, tags, aspect_spans, opinion_spans, token_ranges):
        # label2id = {'N': 0, 'B-A': 1, 'I-A': 2, 'A': 3, 'B-O': 4, 'I-O': 5, 'O': 6, 'negative': 7, 'neutral': 8, 'positive': 9}
        triplets_utm = []
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                tag_num = [0] * len(self.config.label_to_index)
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        a_start = token_ranges[i][0]
                        o_start = token_ranges[j][0]
                        if al < pl:
                            tag_num[int(tags[a_start][o_start])] += 1
                        else:
                            tag_num[int(tags[o_start][a_start])] += 1

                if sum(tag_num[7:]) == 0:
                    continue
                sentiment = -1
                if tag_num[9] >= tag_num[8] and tag_num[9] >= tag_num[7]:
                    sentiment = 9
                elif tag_num[8] >= tag_num[7] and tag_num[8] >= tag_num[9]:
                    sentiment = 8
                elif tag_num[7] >= tag_num[9] and tag_num[7] >= tag_num[8]:
                    sentiment = 7
                if sentiment == -1:
                    print("wrong!!!!!!!!!!!!!!!!!!!!")
                    exit()
                triplets_utm.append([al, ar, pl, pr, sentiment])

        return triplets_utm

    def score_aspect(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_aspect_spans = get_aspects(
                self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], self.config
            )
            for spans in golden_aspect_spans:
                golden_set.add(str(i) + "-" + "-".join(map(str, spans)))

            predicted_aspect_spans = get_aspects(
                self.predictions[i],
                self.sen_lengths[i],
                self.tokens_ranges[i],
                self.config,
            )
            for spans in predicted_aspect_spans:
                predicted_set.add(str(i) + "-" + "-".join(map(str, spans)))

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        return precision, recall, f1

    def score_opinion(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_opinion_spans = get_opinions(
                self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], self.config
            )
            for spans in golden_opinion_spans:
                golden_set.add(str(i) + "-" + "-".join(map(str, spans)))

            predicted_opinion_spans = get_opinions(
                self.predictions[i],
                self.sen_lengths[i],
                self.tokens_ranges[i],
                self.config,
            )
            for spans in predicted_opinion_spans:
                predicted_set.add(str(i) + "-" + "-".join(map(str, spans)))

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        return precision, recall, f1

    def score_uniontags(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_aspect_spans = get_aspects(
                self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], self.config
            )
            golden_opinion_spans = get_opinions(
                self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], self.config
            )
            if self.config.task == "pair":
                golden_tuples = self.find_pair(
                    self.goldens[i],
                    golden_aspect_spans,
                    golden_opinion_spans,
                    self.tokens_ranges[i],
                )
            elif self.config.task == "triplet":
                golden_tuples = self.find_triplet(
                    self.goldens[i],
                    golden_aspect_spans,
                    golden_opinion_spans,
                    self.tokens_ranges[i],
                )
            for pair in golden_tuples:
                golden_set.add(str(i) + "-" + "-".join(map(str, pair)))

            predicted_aspect_spans = get_aspects(
                self.predictions[i],
                self.sen_lengths[i],
                self.tokens_ranges[i],
                self.config,
            )
            predicted_opinion_spans = get_opinions(
                self.predictions[i],
                self.sen_lengths[i],
                self.tokens_ranges[i],
                self.config,
            )
            if self.config.task == "pair":
                predicted_tuples = self.find_pair(
                    self.predictions[i],
                    predicted_aspect_spans,
                    predicted_opinion_spans,
                    self.tokens_ranges[i],
                )
            elif self.config.task == "triplet":
                predicted_tuples = self.find_triplet(
                    self.predictions[i],
                    predicted_aspect_spans,
                    predicted_opinion_spans,
                    self.tokens_ranges[i],
                )
            for pair in predicted_tuples:
                predicted_set.add(str(i) + "-" + "-".join(map(str, pair)))

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        return precision, recall, f1

    def parse_triplet(self, golden=True):
        all_golden_tuples = []
        all_predicted_tuples = []
        for i in range(self.data_num):
            if golden:
                assert len(self.predictions) == len(self.goldens)
                golden_aspect_spans = get_aspects(
                    self.goldens[i],
                    self.sen_lengths[i],
                    self.tokens_ranges[i],
                    self.config,
                )
                golden_opinion_spans = get_opinions(
                    self.goldens[i],
                    self.sen_lengths[i],
                    self.tokens_ranges[i],
                    self.config,
                )
                if self.config.task == "pair":
                    golden_tuples = self.find_pair(
                        self.goldens[i],
                        golden_aspect_spans,
                        golden_opinion_spans,
                        self.tokens_ranges[i],
                    )
                elif self.config.task == "triplet":
                    golden_tuples = self.find_triplet(
                        self.goldens[i],
                        golden_aspect_spans,
                        golden_opinion_spans,
                        self.tokens_ranges[i],
                    )
                else:
                    raise ValueError("Unknown task type: {}".format(self.config.task))
                all_golden_tuples.append(golden_tuples)

            predicted_aspect_spans = get_aspects(
                self.predictions[i],
                self.sen_lengths[i],
                self.tokens_ranges[i],
                self.config,
            )
            predicted_opinion_spans = get_opinions(
                self.predictions[i],
                self.sen_lengths[i],
                self.tokens_ranges[i],
                self.config,
            )
            if self.config.task == "pair":
                predicted_tuples = self.find_pair(
                    self.predictions[i],
                    predicted_aspect_spans,
                    predicted_opinion_spans,
                    self.tokens_ranges[i],
                )
            elif self.config.task == "triplet":
                predicted_tuples = self.find_triplet(
                    self.predictions[i],
                    predicted_aspect_spans,
                    predicted_opinion_spans,
                    self.tokens_ranges[i],
                )
            else:
                raise ValueError("task must be pair or triplet")
            all_predicted_tuples.append(predicted_tuples)
        return all_golden_tuples, all_predicted_tuples

    def score_uniontags_print(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        all_golden_triplets = []
        all_predicted_triplets = []
        for i in range(self.data_num):
            golden_aspect_spans = get_aspects(
                self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], self.config
            )
            golden_opinion_spans = get_opinions(
                self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], self.config
            )
            if self.config.task == "pair":
                golden_tuples = self.find_pair(
                    self.goldens[i],
                    golden_aspect_spans,
                    golden_opinion_spans,
                    self.tokens_ranges[i],
                )
            elif self.config.task == "triplet":
                golden_tuples = self.find_triplet(
                    self.goldens[i],
                    golden_aspect_spans,
                    golden_opinion_spans,
                    self.tokens_ranges[i],
                )
            for pair in golden_tuples:
                golden_set.add(str(i) + "-" + "-".join(map(str, pair)))

            predicted_aspect_spans = get_aspects(
                self.predictions[i],
                self.sen_lengths[i],
                self.tokens_ranges[i],
                self.config,
            )
            predicted_opinion_spans = get_opinions(
                self.predictions[i],
                self.sen_lengths[i],
                self.tokens_ranges[i],
                self.config,
            )
            if self.config.task == "pair":
                predicted_tuples = self.find_pair(
                    self.predictions[i],
                    predicted_aspect_spans,
                    predicted_opinion_spans,
                    self.tokens_ranges[i],
                )
            elif self.config.task == "triplet":
                predicted_tuples = self.find_triplet(
                    self.predictions[i],
                    predicted_aspect_spans,
                    predicted_opinion_spans,
                    self.tokens_ranges[i],
                )
            for pair in predicted_tuples:
                predicted_set.add(str(i) + "-" + "-".join(map(str, pair)))

            all_golden_triplets.append(golden_tuples)
            all_predicted_triplets.append(predicted_tuples)

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        return precision, recall, f1, all_golden_triplets, all_predicted_triplets

    def tagReport(self):
        print(len(self.predictions))
        print(len(self.goldens))

        golden_tags = []
        predict_tags = []
        for i in range(self.data_num):
            for r in range(102):
                for c in range(r, 102):
                    if self.goldens[i][r][c] == -1:
                        continue
                    golden_tags.append(self.goldens[i][r][c])
                    predict_tags.append(self.predictions[i][r][c])

        print(len(golden_tags))
        print(len(predict_tags))
        target_names = [
            "N",
            "B-A",
            "I-A",
            "A",
            "B-O",
            "I-O",
            "O",
            "negative",
            "neutral",
            "positive",
        ]
        print(
            metrics.classification_report(
                golden_tags, predict_tags, target_names=target_names, digits=4
            )
        )


def get_aspects(tags, length, token_range, config=None):
    ignore_index = -1
    spans = []
    start, end = -1, -1
    for i in range(length):
        l, r = token_range[i]
        if tags[l][l] == ignore_index:
            continue
        label = config.index_to_label[int(tags[l][l])]
        if label == "B-A":
            if start != -1:
                spans.append([start, end])
            start, end = i, i
        elif label == "I-A":
            end = i
        else:
            if start != -1:
                spans.append([start, end])
                start, end = -1, -1
    if start != -1:
        spans.append([start, length - 1])

    return spans


def get_opinions(tags, length, token_range, config=None):
    ignore_index = -1

    spans = []
    start, end = -1, -1
    for i in range(length):
        l, r = token_range[i]
        if tags[l][l] == ignore_index:
            continue
        label = config.index_to_label[int(tags[l][l])]
        if label == "B-O":
            if start != -1:
                spans.append([start, end])
            start, end = i, i
        elif label == "I-O":
            end = i
        else:
            if start != -1:
                spans.append([start, end])
                start, end = -1, -1
    if start != -1:
        spans.append([start, length - 1])

    return spans


class DataIterator(object):
    def __init__(self, instances, config):
        self.instances = instances
        self.config = config
        self.batch_count = math.ceil(len(instances) / config.batch_size)

    def get_batch(self, index):
        sentence_ids = []
        sentences = []
        sens_lens = []
        token_ranges = []
        bert_tokens = []
        lengths = []
        masks = []
        aspect_tags = []
        opinion_tags = []
        tags = []
        tags_symmetry = []
        word_pair_position = []
        word_pair_deprel = []
        word_pair_pos = []
        word_pair_synpost = []

        for i in range(
            index * self.config.batch_size,
            min((index + 1) * self.config.batch_size, len(self.instances)),
        ):
            sentence_ids.append(self.instances[i].id)
            sentences.append(self.instances[i].sentence)
            sens_lens.append(self.instances[i].sen_length)
            token_ranges.append(self.instances[i].token_range)
            bert_tokens.append(self.instances[i].bert_tokens_padding)
            lengths.append(self.instances[i].length)
            masks.append(self.instances[i].mask)
            aspect_tags.append(self.instances[i].aspect_tags)
            opinion_tags.append(self.instances[i].opinion_tags)
            tags.append(self.instances[i].tags)
            tags_symmetry.append(self.instances[i].tags_symmetry)

            word_pair_position.append(self.instances[i].word_pair_position)
            word_pair_deprel.append(self.instances[i].word_pair_deprel)
            word_pair_pos.append(self.instances[i].word_pair_pos)
            word_pair_synpost.append(self.instances[i].word_pair_synpost)

        bert_tokens = torch.stack(bert_tokens).to(self.config.device)
        lengths = torch.tensor(lengths).to(self.config.device)
        masks = torch.stack(masks).to(self.config.device)
        aspect_tags = torch.stack(aspect_tags).to(self.config.device)
        opinion_tags = torch.stack(opinion_tags).to(self.config.device)
        tags = torch.stack(tags).to(self.config.device)
        tags_symmetry = torch.stack(tags_symmetry).to(self.config.device)

        word_pair_position = torch.stack(word_pair_position).to(self.config.device)
        word_pair_deprel = torch.stack(word_pair_deprel).to(self.config.device)
        word_pair_pos = torch.stack(word_pair_pos).to(self.config.device)
        word_pair_synpost = torch.stack(word_pair_synpost).to(self.config.device)

        return (
            sentence_ids,
            sentences,
            bert_tokens,
            lengths,
            masks,
            sens_lens,
            token_ranges,
            aspect_tags,
            tags,
            word_pair_position,
            word_pair_deprel,
            word_pair_pos,
            word_pair_synpost,
            tags_symmetry,
        )

    def __len__(self):
        return self.batch_count

    def __iter__(self):
        for i in range(self.batch_count):
            yield self.get_batch(i)


class VocabHelp(object):
    def __init__(self, counter, specials=["<pad>", "<unk>"]):
        self.pad_index = 0
        self.unk_index = 1
        counter = counter.copy()
        self.itos = list(specials)
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(
            key=lambda tup: tup[1], reverse=True
        )  # words_and_frequencies is a tuple

        for word, freq in words_and_frequencies:
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __eq__(self, other):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v):
        words = v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
        return self

    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)
