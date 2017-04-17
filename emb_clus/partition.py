import argparse
from sklearn.mixture import GaussianMixture

from Vocab import Vocab
from CharEmbeddings import CharEmbeddings


def emb_retriever(char_embeddings):
    def embedding(entry):
        if char_embeddings:
            return char_embeddings.word_embedding(entry.word)
        return entry.embeddings

    return embedding


def partition(entries, covariance_type, max_iter, get_embedding, n_comp=2):
    mixture = GaussianMixture(n_components=n_comp, covariance_type=covariance_type, max_iter=max_iter, n_init=1)
    data = [get_embedding(x) for x in entries]
    mixture.fit(data)
    partitions = [[] for _ in range(n_comp)]
    for x in entries:
        inp = get_embedding(x).reshape(1, -1)
        partitions[mixture.predict(inp)[0]].append(x)

    empty = next((p for p in partitions if len(p) == 0), None)
    if empty is not None:
        source = next((p for p in partitions if len(p) > 1), None)
        if source is not None:
            empty.append(source.pop())

    return partitions


print_idx = 0


def rec_partition(children, node_start, entries, covariance_type, max_iter, get_embedding, level, n_comp=2):
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
        ps = partition(entries, covariance_type, max_iter, get_embedding, n_comp)
        if level < 4:
            len1 = len(ps[0])
            len2 = len(ps[1])
            total = len1 + len2
            print("Partitioned at level {}: {} ({:.2%}) / {} ({:.2%})".format(level, len1, len1 / total, len2,
                                                                            len2 / total))
        c1 = rec_partition(children, node_start, ps[0], covariance_type, max_iter, get_embedding, level + 1, n_comp)
        c2 = rec_partition(children, node_start, ps[1], covariance_type, max_iter, get_embedding, level + 1, n_comp)
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
    parser.add_argument('-c', '--char_embedding_file', default=None,
                        help="path to char embeddings. if this is provided, partitioning is done only by char embeddings (input is used as vocab)")
    args = parser.parse_args()

    vocab = Vocab(args.input)
    char_embeddings = CharEmbeddings(args.char_embedding_file) if args.char_embedding_file else None
    get_embedding = emb_retriever(char_embeddings)
    children = []
    root = rec_partition(children, len(vocab.entries), vocab.entries.values(), args.covariance_type, args.max_iter,
                         get_embedding, 0)
    print("root is ", root)
    print("children len is ", len(children))
    write(children, args.output)
