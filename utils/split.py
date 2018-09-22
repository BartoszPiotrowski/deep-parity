#!/usr/bin/env python3
import argparse
import random
'''
usage: python3 utils/dedup_and_split.py file --train 0.5 --valid 0.2 --test 0.3
'''


SEP = ' '  # TODO make use of it


def parse_and_dedup_examples(filename, sep=' '):
    '''
    It is assumed that each line in the file constitutes one example, and if the
    separator = ' ', the lines are of the form of strings 'label example'.
    '''
    labels = []
    examples = []
    with open(filename, 'r') as f:
        examples_lines = f.read().splitlines()
    examples_lines = [e for e in examples_lines if e]  # remove empty lines
    examples_lines = list(set(examples_lines))  # remove duplicates
    random.shuffle(examples_lines)
    for e in examples_lines:
        e_sep = e.split(sep)
        labels.append(e_sep[0])
        examples.append(e_sep[1])
    return labels, examples


def partition_list(l, partition_sizes):
    assert sum(partition_sizes) == 1.
    partitioned = []
    length = len(l)
    for s in partition_sizes:
        if l:
            index = round(s * length)
            partitioned.append(l[:index])
            l = l[index:]
    return partitioned


def split_to_train_valid_test(labels, examples, train=.5, valid=.3, test=.2):
    assert train + valid + test == 1.
    classes_examples = {c: [] for c in set(labels)}
    for i in range(len(labels)):
        classes_examples[labels[i]].append(examples[i])
    train_labels, train_examples = [], []
    valid_labels, valid_examples = [], []
    test_labels, test_examples = [], []
    for c in classes_examples:
        train_part, valid_part, test_part = \
            partition_list(classes_examples[c], (train, valid, test))
        train_examples.extend(train_part)
        valid_examples.extend(valid_part)
        test_examples.extend(test_part)
        train_labels.extend([c for i in train_part])
        valid_labels.extend([c for i in valid_part])
        test_labels.extend([c for i in test_part])
    return (train_labels, valid_labels, test_labels), \
           (train_examples, valid_examples, test_examples)


def write_examples(labels, examples, filename, sep=' '):
    tuples = list(zip(labels, examples))
    lines = [sep.join(t) for t in tuples]
    random.shuffle(lines)
    with open(filename, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_vocab(examples, filename):
    lines = list(set(''.join(examples)))
    with open(filename, 'w') as f:
        f.write('\n'.join(lines) + '\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split file with training examples into three balanced.")
    parser.add_argument(
        'filename',
        type=str,
        help="Path to file with examples.")
    parser.add_argument(
        '--dirname',
        type=str,
        help="Path to directory to save.")
    parser.add_argument('--train', type=float, help="Train set size.")
    parser.add_argument('--valid', type=float, help="Validation set size.")
    parser.add_argument('--test', type=float, help="Test set size.")
    args = parser.parse_args()
    assert args.train + args.valid + args.test == 1.

labels, examples = parse_and_dedup_examples(args.filename)
(train_labels, valid_labels, test_labels), \
    (train_examples, valid_examples, test_examples) = \
    split_to_train_valid_test(labels, examples, args.train, args.valid, args.test)
write_examples(train_labels, train_examples, args.dirname + '/train')
write_examples(valid_labels, valid_examples, args.dirname + '/valid')
write_examples(test_labels, test_examples, args.dirname + '/test')
write_vocab(examples, args.dirname + '/vocab')
write_vocab(labels, args.dirname + '/labels')
