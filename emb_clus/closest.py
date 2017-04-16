import argparse
import operator
from scipy import spatial

from Vocab import Vocab


# best performing types = cosine, braycurtis
def entries_by_dist(entries, target_entry, distance_type):
    res = {}
    dist_f = getattr(spatial.distance, distance_type)
    for e in entries:
        if e == target_entry:
            continue
        dist = dist_f(target_entry.embeddings, e.embeddings)
        res[e] = dist
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-f', '--file', help='embeddings file name (tsv)', required=True)
    parser.add_argument('target_word', help='target word to find closest items to')
    parser.add_argument('-n', '--num', type=int, default=10, help='number of closest words to find')
    parser.add_argument('-t', '--distance_type', default='cosine',
                        choices=['cosine', 'euclidean', 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
                                 'correlation', 'sqeuclidean'],
                        help='Distance type to use to find closest')
    args = parser.parse_args()

    vocab = Vocab(args.file)
    target_entry = vocab.entries[args.target_word]

    measured = entries_by_dist(vocab.entries.values(), target_entry, args.distance_type)
    sorted_measured = sorted(measured.items(), key=operator.itemgetter(1))
    for e, d in sorted_measured[:args.num]:
        print("{}:\t{}".format(e.word, d))
