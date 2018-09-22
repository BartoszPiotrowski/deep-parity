#!/usr/bin/env python3
import numpy as np


class Dataset:
    def __init__(
            self,
            terms_filename,
            vocab_filename,
            shuffle_batches=True,
            test=False):
        self.terms = []
        self.labels = []
        with open(terms_filename, 'r') as terms:
            if test:
                for line in terms:
                    self.terms.append(line.strip('\n'))
            else:
                for line in terms:
                    l, t = line.strip('\n').split(' ')
                    self.terms.append(t)
                    self.labels.append(l)
        with open(vocab_filename, 'r') as vocab:
            self.vocab = vocab.read().splitlines()
        self.num_tokens = len(self.vocab) + 1  # + 1 because of padding with 0s
        self.num_labels = len(set(self.labels))
        self.vocab_map = {self.vocab[i]: i + 1 for i in range(len(self.vocab))}
        self.seqs = [[self.vocab_map[t] for t in f] for f in self.terms]
        self.terms_lens = [len(f) for f in self.terms]
        self.shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self)) \
            if self.shuffle_batches else np.arange(len(self))

    def __len__(self):
        return len(self.terms)

    def pad(self, sequences, length, pad_symbol=0):
        padded_sequences = []
        for s in sequences:
            assert len(s) <= length
            padded_sequences.append(s + [pad_symbol] * (length - len(s)))
        return padded_sequences

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = \
            self._permutation[:batch_size], self._permutation[batch_size:]
        lens = [self.terms_lens[i] for i in batch_perm]
        max_len = max(lens)
        seqs = np.array(self.pad([self.seqs[i] for i in batch_perm], max_len))
        labels = [self.labels[i] for i in batch_perm]
        return seqs, lens, labels

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self)) \
                if self.shuffle_batches else np.arange(len(self))
            return True
        return False

    def test(self):
        max_len = max(self.terms_lens)
        seqs = np.array(self.pad(self.seqs, max_len))
        return seqs, self.terms_lens
