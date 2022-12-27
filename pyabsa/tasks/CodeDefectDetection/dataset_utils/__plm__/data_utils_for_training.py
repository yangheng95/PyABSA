# -*- coding: utf-8 -*-
# file: data_utils_for_training.py
# time: 02/11/2022 15:39
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import tqdm

from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from ..cdd_utils import _prepare_corruptted_code_src, read_defect_examples
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from pyabsa.utils.pyabsa_utils import check_and_fix_labels, fprint


class BERTCDDDataset(PyABSADataset):
    def load_data_from_dict(self, dataset_dict, **kwargs):
        pass

    def load_data_from_file(self, dataset_file, **kwargs):
        lines = load_dataset_from_file(self.config.dataset_file[self.dataset_type], config=self.config)
        lines = read_defect_examples(lines, self.config.get('data_num', -1), self.config.get('remove_comments', True))
        all_data = []

        label_set = set()
        c_label_set = set()

        for ex_id, line in enumerate(tqdm.tqdm(lines, description='preparing dataloader...')):
            code_src, label = line.strip().split('$LABEL$')
            # source_str = "{}: {}".format(args.task, example.source)

            code_ids = self.tokenizer.text_to_sequence(code_src, max_length=self.config.max_seq_len,
                                                       padding='max_length', truncation=True)
            data = {
                'ex_id': ex_id,
                'source_ids': code_ids,
                'label': label,
                'corrupt_label': 0
            }

            label_set.add(label)
            c_label_set.add(0)
            all_data.append(data)

            for _ in range(self.config.corrupt_instance_num):
                corrupt_code_src = _prepare_corruptted_code_src(code_src)
                corrupt_code_ids = self.tokenizer.text_to_sequence(corrupt_code_src,
                                                                   max_length=self.config.max_seq_len,
                                                                   padding='max_length', truncation=True)
                data = {
                    'ex_id': ex_id,
                    'source_ids': corrupt_code_ids,
                    'label': label,
                    'corrupt_label': 1
                }
                c_label_set.add(1)
                all_data.append(data)

        check_and_fix_labels(label_set, 'label', all_data, self.config)
        self.config.output_dim = len(label_set)

        self.data = all_data

    def __init__(self, config, tokenizer, dataset_type='train', **kwargs):
        super().__init__(config, tokenizer, dataset_type, **kwargs)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
