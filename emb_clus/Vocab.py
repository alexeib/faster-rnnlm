import numpy as np

class VocabEntry:
    def __init__(self, index, word, freq, embeddings):
        self.index = index
        self.word = word
        self.freq = freq
        self.embeddings = np.array(list(map(float, embeddings)))


class Vocab:
    entries = {}

    def __init__(self, filename):
        idx = 0
        freq_sum = 0;
        for line in open(filename):
            items = line.rstrip().split('\t')
            entry = VocabEntry(idx, items[0], int(items[1]), items[2:])
            self.entries[items[0]] = entry
            idx += 1
            freq_sum += entry.freq
        for e in self.entries.values():
            e.freq = e.freq / freq_sum
