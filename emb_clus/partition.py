import argparse
from sklearn.mixture import GaussianMixture
from collections import deque

from Vocab import Vocab
from CharEmbeddings import CharEmbeddings


def emb_retriever(char_embeddings):
    def embedding(entry):
        if char_embeddings:
            return char_embeddings.word_embedding(entry.word)
        return entry.embeddings

    return embedding


def partition(entries, covariance_type, max_iter, get_embedding, max_cluster_weight_perc):
    mixture = GaussianMixture(n_components=2, covariance_type=covariance_type, max_iter=max_iter, n_init=1)
    data = [get_embedding(x) for x in entries]
    mixture.fit(data)
    total_weight = sum([x.freq for x in entries])
    scored = []
    for x in entries:
        inp = get_embedding(x).reshape(1, -1)
        probs = mixture.predict_proba(inp)
        scored.append((x, probs[0][0], probs[0][1]))

    scored.sort(key=lambda x: x[1])
    p1 = []
    p1_weight = 0
    while len(scored) > 1:
        prob_1 = scored[-1][1]
        prob_2 = scored[-1][2]
        w_1 = (p1_weight + scored[-1][0].freq) / total_weight
        w_2 = 1 - w_1 + (scored[-1][0].freq / total_weight)
        if len(p1) > 0 and (
                    (w_1 > max_cluster_weight_perc > w_2) or (prob_2 > prob_1 and w_2 < max_cluster_weight_perc)):
            break
        p1.append(scored.pop()[0])
        p1_weight += p1[-1].freq
    p2 = [x[0] for x in scored]

    return [p1, p2]


print_idx = 0


def rec_partition(children, node_start, entries, covariance_type, max_iter, get_embedding, max_cluster_weight_perc,
                  level):
    if len(entries) == 0:
        raise Exception('must have more than 0 entries')
    if len(entries) == 1:
        return entries[0].index
    elif len(entries) == 2:
        global print_idx
        print_idx += 1
        if print_idx % 100 == 0:
            print([e.word for e in entries])
        children += [e.index for e in entries]
    else:
        ps = partition(entries, covariance_type, max_iter, get_embedding, max_cluster_weight_perc)
        if level < 4:
            len1 = len(ps[0])
            freq1 = sum([w.freq for w in ps[0]])
            freq2 = sum([w.freq for w in ps[1]])
            freq_sum = freq1 + freq2
            freq1 = freq1 / freq_sum
            freq2 = freq2 / freq_sum
            len2 = len(ps[1])
            total = len1 + len2
            len1_ratio = len1 / total
            len2_ratio = len2 / total
            print(
                "Partitioned at level {level}: {len1} ({len1_ratio:.2%}, freq: {freq1:.4%}) / {len2} ({len2_ratio:.2%}, freq: {freq2:.4%})".format(
                    **locals()))
        c1 = rec_partition(children, node_start, ps[0], covariance_type, max_iter, get_embedding,
                           max_cluster_weight_perc, level + 1)
        c2 = rec_partition(children, node_start, ps[1], covariance_type, max_iter, get_embedding,
                           max_cluster_weight_perc, level + 1)
        children += [c1, c2]
    node_id = node_start + (len(children) // 2) - 1
    c1 = (node_id - node_start) * 2
    c2 = (node_id - node_start) * 2 + 1
    child1 = children[c1] if c1 >= 0 else None
    child2 = children[c2] if c2 >= 0 else None
    if node_id == child1 or node_id == child2:
        print("wtf ", node_id, child1, child2)
    return node_id


def write(arr, output):
    with open(output, "w") as f:
        for x in arr:
            print(x, file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-i', '--input', help='embeddings file name (tsv)', required=True)
    requiredNamed.add_argument('-o', '--output', help='output file (children array for binary tree)', required=True)
    parser.add_argument('-t', '--covariance_type', default='spherical', choices=['spherical', 'tied', 'diag', 'full'],
                        help='Type of covariance to use')
    parser.add_argument('-m', '--max_iter', type=int, default=10, help="number of max iterations for EM")
    parser.add_argument('-w', '--max_cluster_weight_perc', type=float, default=1,
                        help="maximum weight percentage that a cluster may take. should be > 0 and <= 1")
    parser.add_argument('-c', '--char_embedding_file', default=None,
                        help="path to char embeddings. if this is provided, partitioning is done only by char embeddings (input is used as vocab)")
    args = parser.parse_args()

    vocab = Vocab(args.input)
    char_embeddings = CharEmbeddings(args.char_embedding_file) if args.char_embedding_file else None
    get_embedding = emb_retriever(char_embeddings)
    children = []
    root = rec_partition(children, len(vocab.entries), vocab.entries.values(), args.covariance_type, args.max_iter,
                         get_embedding, args.max_cluster_weight_perc, 0)
    print("root is ", root)
    print("children len is ", len(children))
    write(children, args.output)
