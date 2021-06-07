import os
import copy

from pyabsa.pyabsa_utils import find_target_file


def is_similar(s1, s2):
    count = 0.0
    for token in s1.split(' '):
        if token in s2:
            count += 1
    if count / len(s1.split(' ')) >= 0.8 and count / len(s2.split(' ')) >= 0.8:
        return True
    else:
        return False


def assemble_aspects(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('$ t $', '$T$').strip()

    def unify_same_samples(same_samples):
        text = same_samples[0][0].replace('$T$', same_samples[0][1])
        polarities = [-999] * len(text.split())
        tags = ['O'] * len(text.split())
        samples = []
        for sample in same_samples:
            # print(sample)
            polarities_tmp = copy.deepcopy(polarities)

            try:
                asp_begin = (sample[0].split().index('$T$'))
                asp_end = sample[0].split().index('$T$') + len(sample[1].split())
                for i in range(asp_begin, asp_end):
                    polarities_tmp[i] = int(sample[2])
                    if i - sample[0].split().index('$T$') < 1:
                        tags[i] = 'B-ASP'
                    else:
                        tags[i] = 'I-ASP'
                samples.append([text, tags, polarities_tmp])
            except:
                print('Ignore Error:', sample[0])

        return samples

    samples = []
    aspects_in_one_sentence = []
    for i in range(0, len(lines), 3):

        lines[i] = lines[i].replace('$T$', ' $T$ ').replace('  ', ' ')

        if len(aspects_in_one_sentence) == 0:
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])
            continue
        if is_similar(aspects_in_one_sentence[-1][0], lines[i]):
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])
        else:
            samples.extend(unify_same_samples(aspects_in_one_sentence))
            aspects_in_one_sentence = []
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])

    return samples


def split_aspects(sentence):
    single_aspect_with_contex = []

    aspect_num = len(sentence[1].split("|"))
    aspects = sentence[1].split("|")
    polarity = sentence[2].split("|")
    pre_position = 0
    aspect_contex = sentence[0]
    for i in range(aspect_num):
        aspect_contex = aspect_contex.replace("$A$", aspects[i], 1)
        single_aspect_with_contex.append(
            (aspect_contex[pre_position:aspect_contex.find("$A$")], aspects[i], polarity[i]))
        pre_position = aspect_contex.find(aspects[i]) + len(aspects[i]) + 1

    return single_aspect_with_contex


def convert(fname):
    dist_fname = fname + '.atepc'
    lines = []
    samples = assemble_aspects(fname)

    for sample in samples:
        for token_index in range(len(sample[1])):
            token, label, polarity = sample[0].split()[token_index], sample[1][token_index], sample[2][token_index]
            lines.append(token + " " + label + " " + str(polarity))
        lines.append('\n')

    # 写之前，先检验文件是否存在，存在就删掉
    if os.path.exists(dist_fname):
        os.remove(dist_fname)
    fout = open(dist_fname, 'w', encoding='utf8')
    for line in lines:
        fout.writelines((line + '\n').replace('\n\n', '\n'))
    fout.close()


# 将数据集中的aspect切割出来
def convert_apc_set_to_atepc_set(path):
    for target_file in find_target_file(path, file_type='', exclude_key='atepc', find_all=True):
        try:
            convert(target_file)
        except:
            print('failed to process"{}'.format(target_file))
    print('finished')


# 将数据集中的aspect切割出来
def refactor_chinese_dataset(fname, train_fname, test_fname):
    lines = []
    samples = assemble_aspects(fname)
    positive = 0
    negative = 0
    sum = 0
    # refactor testset
    for sample in samples[:int(len(samples) / 5)]:
        for token_index in range(len(sample[1])):
            token, label, polarty = sample[0].split()[token_index], sample[1][token_index], sample[2][token_index]
            lines.append(token + " " + label + " " + str(polarty))
        lines.append('\n')
        if 1 in sample[2]:
            positive += 1
        else:
            negative += 1
        sum += 1
    print(train_fname + f"sum={sum} positive={positive} negative={negative}")
    if os.path.exists(test_fname):
        os.remove(test_fname)
    fout = open(test_fname, 'w', encoding='utf8')
    for line in lines:
        fout.writelines((line + '\n').replace('\n\n', '\n'))
    fout.close()

    positive = 0
    negative = 0
    sum = 0
    # refactor trainset
    for sample in samples[int(len(samples) / 5):]:
        for token_index in range(len(sample[1])):
            tokens = sample[0].split()
            token, label, polarty = sample[0].split()[token_index], sample[1][token_index], sample[2][token_index]
            lines.append(token + " " + label + " " + str(polarty))
        lines.append('\n')
        if 1 in sample[2]:
            positive += 1
        else:
            negative += 1
        sum += 1
    print(train_fname + f"sum={sum} positive={positive} negative={negative}")
    if os.path.exists(train_fname):
        os.remove(train_fname)
    fout = open(train_fname, 'w', encoding='utf8')
    for line in lines:
        fout.writelines((line + '\n').replace('\n\n', '\n'))
    fout.close()


def detect_error_in_dataset(dataset):
    f = open(dataset, 'r', encoding='utf8')
    lines = f.readlines()
    for i in range(0, len(lines), 3):
        # print(lines[i].replace('$T$', lines[i + 1].replace('\n', '')))
        if i + 3 < len(lines):
            if is_similar(lines[i], lines[i + 3]) and len((lines[i] + " " + lines[i + 1]).split()) != len(
                    (lines[i + 3] + " " + lines[i + 4]).split()):
                print(lines[i].replace('$T$', lines[i + 1].replace('\n', '')))
                print(lines[i + 3].replace('$T$', lines[i + 4].replace('\n', '')))
