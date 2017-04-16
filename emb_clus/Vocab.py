class VocabEntry:
    def __init__(self, index, word, embeddings):
        self.index = index
        self.word = word
        self.embeddings = list(map(float, embeddings))


class Vocab:
    entries = {}

    def __init__(self, filename):
        idx = 0
        for line in open(filename):
            items = line.rstrip().split('\t')
            entry = VocabEntry(idx, items[0], items[1:])
            self.entries[items[0]] = entry
            idx += 1
