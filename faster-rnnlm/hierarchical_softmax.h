#ifndef FASTER_RNNLM_HIERARCHICAL_SOFTMAX_H_
#define FASTER_RNNLM_HIERARCHICAL_SOFTMAX_H_

#include <math.h>
#include <stdio.h>

#include <vector>

#include "faster-rnnlm/settings.h"
#include "faster-rnnlm/util.h"
#include "faster-rnnlm/words.h"

class MaxEnt;

class Tree {
public:
    // children array defines for each inner node defines its children
    Tree(int leaf_count, const std::vector<int>& children, int arity);

    int GetRootNode() const { return root_node_; }

    int GetTreeHeight() const { return tree_height_; }

    int GetArity() const { return arity_; }

    bool IsLeaf(int node) const { return (node<leaf_count_); }

    // get path_lengths for the word, i.e. the length of the path
    // from the root to the word
    int GetPathLength(WordIndex word) const { return path_lengths_[word]; }

    int& GetPathLength(WordIndex word) { return path_lengths_[word]; }

    // get array of path length node ids that lead from root to the word
    const int* GetPathToLeaf(WordIndex word) const
    {
        return points_.data()+word*(MAX_HSTREE_DEPTH+1);
    }

    int* GetPathToLeaf(WordIndex word)
    {
        return points_.data()+word*(MAX_HSTREE_DEPTH+1);
    }

    // get array of (path_lengths - 1) indices of branch ids that lead from root to the word
    const int* GetBranchPathToLead(WordIndex word) const
    {
        return branches_.data()+word*MAX_HSTREE_DEPTH;
    }

    int* GetBranchPathToLead(WordIndex word)
    {
        return branches_.data()+word*MAX_HSTREE_DEPTH;
    }

    // get array of size arity_ that contains indices of the node's children
    // 'node' must be inner node (node >= vocab_size)
    const int* GetChildren(int node) const
    {
        return children_.data()+(node-leaf_count_)*arity_;
    }

    // Get offset that corresponds to the child node in weights matrix
    //
    // Child node is the child number 'branch' of the node 'node_id'
    //
    // 'branch' belongs to {0, ..., arity_ - 2}
    // 'node' must be inner node (node >= vocab_size)
    int GetChildOffset(int node, int branch) const
    {
        return (node-leaf_count_)*(arity_-1)+branch;
    }

    // Get offset that corresponds to the child node in weights matrix
    //
    // Child node is the child number 'branch' of the node on depth 'depth'
    // on the path from the root to the word
    //
    // 'branch' belongs to {0, ..., arity_ - 2}
    int GetChildOffsetByDepth(WordIndex word, int depth, int branch) const
    {
        return GetChildOffset(GetPathToLeaf(word)[depth], branch);
    }

protected:
    const int leaf_count_;
    const int arity_;
    int root_node_;
    int tree_height_;

    std::vector<int> children_;
    std::vector<int> path_lengths_;
    std::vector<int> points_;
    std::vector<int> branches_;
};

class HSTree {
public:
    // Factory function to build k-ary Huffman tree_ using the word counts
    // Frequent words will have short unique k-nary codes
    static HSTree* CreateHuffmanTree(const Vocabulary&, int layer_size, int arity);

    static HSTree* CreateRandomTree(const Vocabulary&, int layer_size, int arity, uint64_t seed);

    ~HSTree();

    void Dump(FILE* fo) const;

    void Load(FILE* fo);

    // Make ME hash act like Bloom filter: if the weight is zero, it was probably
    // never touched by training and this (an higher) ngrams
    // should not be considered for the target_word
    //
    // returns truncated maxent_order
    int DetectEffectiveMaxentOrder(
            WordIndex target_word, const MaxEnt* maxent,
            const uint64_t* feature_hashes, int maxent_order) const;

    // Propagate softmax forward and backward
    // given maxent and hidden layers
    //
    // Updates
    //   tree_->weights_, hidden_grad, maxent
    //
    // Returns
    //   log10 probability of the words, if calculate_probability is true
    //   0, otherwise
    //
    // feature_hashes is an array of offsets for the target_word
    // feature_hashes must contain at least maxent_order elements
    Real PropagateForwardAndBackward(
            bool calculate_probability, WordIndex target_word,
            const uint64_t* feature_hashes, int maxent_order,
            Real lrate, Real maxent_lrate, Real l2reg, Real maxent_l2reg, Real gradient_clipping,
            const Real* hidden,
            Real* hidden_grad, MaxEnt* maxent);

    // Propagate softmax forward and calculate probability
    // given maxent and hidden layers
    //
    //
    // Returns
    //   log10 probability of the word
    //
    // feature_hashes is an array of offsets for the target_word
    // feature_hashes must contain at least maxent_order elements
    Real CalculateLog10Probability(
            WordIndex target_word,
            const uint64_t* feature_hashes, int maxent_order,
            bool dynamic_maxent_prunning,
            const Real* hidden, const MaxEnt* maxent) const;

    // Sample a word given maxent and hidden layers
    //
    //
    // Returns
    //   log10 probability of the word
    //   sampled word
    //
    // feature_hashes is an array of offsets for the target_word
    // feature_hashes must contain at least maxent_order elements
    void SampleWord(
            const uint64_t* feature_hashes, int maxent_order,
            const Real* hidden, const MaxEnt* maxent,
            Real* logprob, WordIndex* sampled_word) const;

    std::vector<double> ChildProbs(int node, const uint64_t* feature_hashes, int maxent_order,
            bool dynamic_maxent_prunning,
            const Real* hidden, const MaxEnt* maxent) const;

    const Tree* GetTree() { return tree_; }

    const int layer_size;
    const size_t syn_size;
    RowMatrix weights_;
    Tree* tree_;

protected:
    HSTree(int vocab_size, int layer_size, int arity, const std::vector<int>& children);
private:
    HSTree(const HSTree&);
    HSTree& operator=(const HSTree&);
};

#endif  // FASTER_RNNLM_HIERARCHICAL_SOFTMAX_H_
