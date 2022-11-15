# -*- coding: utf-8 -*-
# file: data_utils_for_training.py
# time: 02/11/2022 15:39
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import tqdm

from pyabsa import LabelPaddingOption
from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from pyabsa.framework.tokenizer_class.tokenizer_class import pad_and_truncate


class RNACDataset(PyABSADataset):
    def load_data_from_dict(self, dataset_dict, **kwargs):
        label_set = set()
        all_data = []

        for ex_id, data in enumerate(tqdm.tqdm(dataset_dict[self.dataset_type], postfix='preparing dataloader...')):
            if self.config.dataset_name.lower() in 'degrad':
                rna, label = data['text'], data['label']
                rna_indices = self.tokenizer.text_to_sequence(rna)
                rna_indices = pad_and_truncate(rna_indices, self.config.max_seq_len)
                data = {
                    'ex_id': ex_id,
                    'text_indices': rna_indices,
                    'label': label,
                }
                label_set.add(label)
                all_data.append(data)
            elif self.config.dataset_name.lower() in 'sfe':
                exon1, intron, exon2, label = data['exon1'], data['intron'], data['exon2'], data['label']
                exon1_ids = self.tokenizer.text_to_sequence(exon1, padding='do_not_pad')
                intron_ids = self.tokenizer.text_to_sequence(intron, padding='do_not_pad')
                exon2_ids = self.tokenizer.text_to_sequence(exon2, padding='do_not_pad')

                rna_indices = exon1_ids + intron_ids + exon2_ids
                rna_indices = pad_and_truncate(rna_indices, self.config.max_seq_len)
                data = {
                    'ex_id': ex_id,
                    'text_indices': rna_indices,
                    'label': label,
                }
                label_set.add(label)
                all_data.append(data)

        check_and_fix_labels(label_set, 'label', all_data, self.config)
        self.config.output_dim = len(label_set)
        self.data = all_data

    def load_data_from_file(self, dataset_file, **kwargs):
        lines = load_dataset_from_file(dataset_file[self.dataset_type])

        all_data = []

        label_set1 = set()
        label_set2 = set()

        for ex_id, i in enumerate(tqdm.tqdm(range(len(lines)), postfix='preparing dataloader...')):
            if self.config.dataset_name.lower() in 'degrad-v2':
                text, _, label = lines[i].strip().partition('$LABEL$')
                rna = text.strip()
                labels = label.strip().split(',')

                rna_indices = self.tokenizer.text_to_sequence(rna, padding='do_not_pad')

                import numpy as np
                if self.dataset_type != 'test' and self.dataset_type != 'valid':
                    new_rna_indices = np.array(rna_indices[:])
                    new_rna_indices[np.random.randint(0, np.count_nonzero(new_rna_indices) - 1, size=np.count_nonzero(new_rna_indices) // 10, dtype=int)] = self.tokenizer.pad_token_id
                    new_rna_indices = new_rna_indices.tolist()
                    new_rna_indices = pad_and_truncate(new_rna_indices, self.config.max_seq_len)
                    if np.count_nonzero(new_rna_indices) != 0:
                        data = {
                            'ex_id': ex_id,
                            'text': rna,
                            'text_indices': new_rna_indices,
                            'label1': labels[0],
                            'label2': labels[1],
                        }
                        label_set1.add(labels[0])
                        label_set2.add(labels[1])
                        all_data.append(data)

                rna_indices = pad_and_truncate(rna_indices, self.config.max_seq_len)

                data = {
                    'ex_id': ex_id,
                    'text': rna,
                    'text_indices': rna_indices,
                    'label1': labels[0],
                    'label2': labels[1],
                }
                label_set1.add(labels[0])
                label_set2.add(labels[1])
                all_data.append(data)

            elif self.config.dataset_name.lower() in 'sfe':
                sequence, _, label = lines[i].strip().partition('$LABEL$')
                label = label.strip() if label else LabelPaddingOption.LABEL_PADDING
                exon1, intron, exon2 = sequence.strip().split(',')
                exon1 = exon1.strip()
                intron = intron.strip()
                exon2 = exon2.strip()
                exon1_ids = self.tokenizer.text_to_sequence(exon1, padding='do_not_pad')
                intron_ids = self.tokenizer.text_to_sequence(intron, padding='do_not_pad')
                exon2_ids = self.tokenizer.text_to_sequence(exon2, padding='do_not_pad')
                rna_indices = [self.tokenizer.tokenizer.cls_token_id] + exon1_ids + intron_ids + exon2_ids + [self.tokenizer.tokenizer.sep_token_id]
                rna_indices = pad_and_truncate(rna_indices, self.config.max_seq_len, value=self.tokenizer.pad_token_id)

                intron_indices = self.tokenizer.text_to_sequence(intron)

                data = {
                    'ex_id': ex_id,
                    'text_indices': rna_indices,
                    'intron_indices': intron_indices,
                    'label': label,
                }
                label_set1.add(label)
                all_data.append(data)

        check_and_fix_labels(label_set1, 'label1', all_data, self.config)
        check_and_fix_labels(label_set2, 'label2', all_data, self.config)
        self.config.output_dim1 = len(label_set1)
        self.config.output_dim2 = len(label_set2)
        self.data = all_data

    def __init__(self, config, tokenizer, dataset_type='train', **kwargs):
        super().__init__(config, tokenizer, dataset_type=dataset_type, **kwargs)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class GloVeRNACDataset(RNACDataset):
    pass


class BERTRNACDataset(RNACDataset):
    pass


def check_and_fix_labels(label_set: set, label_name, all_data, config):
    # update output_dim, init model behind execution of this function!
    config[f'{label_name}_to_index'] = {origin_label: int(idx) if origin_label != '-100' else -100 for origin_label, idx
                                        in zip(sorted(label_set), range(len(label_set)))}
    config[f'index_to_{label_name}'] = {int(idx) if origin_label != '-100' else -100: origin_label for origin_label, idx
                                        in zip(sorted(label_set), range(len(label_set)))}
    num_label = {l: 0 for l in label_set}
    num_label['Sum'] = len(all_data)
    for item in all_data:
        try:
            num_label[item[label_name]] += 1
            item[label_name] = config[f'{label_name}_to_index'][item[label_name]]
        except Exception as e:
            # print(e)
            num_label[item.polarity] += 1
            item[label_name] = config[f'{label_name}_to_index']['-100']
    config.logger.info('Dataset Label Details: {}'.format(num_label))
    print('Dataset Label Details: {}'.format(num_label))
