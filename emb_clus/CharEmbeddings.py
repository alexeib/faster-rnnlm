import numpy as np


class CharEmbeddings:
    embeddings = []

    def __init__(self, filename):
        for line in open(filename):
            items = line.rstrip().split(',')
            self.embeddings.append(np.array(list(map(float, items))))

    def char_embedding(self, char):
        return self.embeddings[ord(char)]

    def word_embedding(self, word):
        emb = np.zeros(len(self.embeddings[0]))
        for c in word:
            emb += self.char_embedding(c)
        return emb / len(word)
