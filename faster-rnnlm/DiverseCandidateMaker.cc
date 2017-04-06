//
// Created by abaev on 4/5/2017.
//

#include "DiverseCandidateMaker.h"

DirData createDirData(NNet* net, const std::vector<WordIndex>& wids, int pos) {
    uint64_t ngram_hashes[MAX_NGRAM_ORDER];
    auto updater= net->rec_layer->CreateUpdater();
    const auto& output = updater->GetOutputMatrix();
    auto maxent_present =CalculateMaxentHashIndices(net, wids.data(), pos, ngram_hashes);
    return { ngram_hashes, maxent_present, net, output.row(pos-1).data(),updater};
}

std::vector<WordIndex>
DiverseCandidateMaker::DiverseCandidates(const std::vector<WordIndex>& wids, int pos, int target_number,
        bool dynamic_maxent_prunning) const
{
    auto tree = mn_.GetHSTree();
    if (tree->GetArity()!=2) throw "expected tree with arity = 2";
    int target_depth = ceil(log2(target_number))-1;

    auto forw = createDirData(mn_.GetForwardNet(), wids, pos);
    const std::vector<WordIndex> revWids(wids.rbegin(), wids.rend());
    auto rev = createDirData(mn_.GetReverseNet(), revWids, wids.size() - 1 - pos);

    return DiverseCandidates(tree->GetRootNode(), 0, target_depth, dynamic_maxent_prunning,
            wids.size(), pos, forw, rev);
}

std::vector<double>
combinedProbs(const std::vector<double>& forward_probs, const std::vector<double>& reverse_prob, int sentence_length,
        int word_pos)
{
    std::vector<double> ret;
    double forw_weight, rev_weight;
    if (sentence_length<=1) {
        forw_weight = rev_weight = 0;
    }
    else {
        forw_weight = (double) word_pos/(sentence_length-1);
        rev_weight = (double) (sentence_length-1-word_pos)/(sentence_length-1);
    }
    for (int i = 0; i<forward_probs.size() && i<reverse_prob.size(); i++) {
        ret.emplace_back((forward_probs[i]*forw_weight)+(reverse_prob[i]*rev_weight));
    }
    return ret;
}

std::vector<WordIndex>
DiverseCandidateMaker::DiverseCandidates(int node, int curr_depth, int target_depth, bool dynamic_maxent_prunning,
        int sentence_length, int word_pos, const DirData& forward, const DirData& reverse) const
{
    const Real kThreshold = 0.2;
    auto tree = forward.nnet->softmax_layer->GetTree();

    std::vector<WordIndex> res;
    if (tree->IsLeaf(node)) {
        res.push_back(node);
        return res;
    }
    auto children = tree->GetChildren(node);
    auto forward_child_probs = forward.nnet->softmax_layer->ChildProbs(node, forward.feature_hashes,
            forward.maxent_present,
            dynamic_maxent_prunning, forward.hidden, &forward.nnet->maxent_layer);
    auto reverse_child_probs = reverse.nnet->softmax_layer->ChildProbs(node, reverse.feature_hashes,
            reverse.maxent_present,
            dynamic_maxent_prunning, reverse.hidden, &reverse.nnet->maxent_layer);
    auto child_probs = combinedProbs(forward_child_probs, reverse_child_probs, sentence_length, word_pos);
    if (curr_depth<=target_depth) {
        for (int i = 0; i<tree->GetArity(); ++i) {
            if (child_probs[i]>=kThreshold) {
                auto v = DiverseCandidates(children[i], curr_depth+(1-child_probs[i]<kThreshold ? 0 : 1), target_depth,
                        dynamic_maxent_prunning, sentence_length, word_pos, forward, reverse);
                res.insert(res.end(), v.begin(), v.end());
            }
        }
    }
    else {
        auto max_branch = std::distance(child_probs.begin(),
                std::max_element(child_probs.begin(), child_probs.end()));
        auto selected_node = children[max_branch];
        if (tree->IsLeaf(selected_node)) {
            res.push_back(selected_node);
        }
        else {
            auto v = DiverseCandidates(selected_node, curr_depth+1, target_depth, dynamic_maxent_prunning,
                    sentence_length, word_pos, forward, reverse);
            res.insert(res.end(), v.begin(), v.end());
        }
    }
    return res;
}

