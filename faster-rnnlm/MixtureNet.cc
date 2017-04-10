//
// Created by abaev on 4/5/2017.
//

#include "MixtureNet.h"

MixtureNet::MixtureNet(NNet* forward, NNet* reverse, bool dynamicPruning)
        :forward_(forward), reverse_(reverse), kDynamicMaxentPruning(dynamicPruning)
{
}

std::vector<WordIndex> MixtureNet::GetWids(const std::string& sentence) const
{
    std::vector<std::string> words;
    split(sentence, ' ', std::back_inserter(words));
    std::vector<WordIndex> wids;
    for (auto& t : words) {
        WordIndex wid = forward_.nnet->vocab.GetIndexByWord(t.c_str());
        if (wid==0) {
            break;
        }
        if (wid==Vocabulary::kWordOOV) {
            wid = forward_.nnet->vocab.GetIndexByWord("<unk>");
            if (wid==Vocabulary::kWordOOV) {
                fprintf(stderr, "ERROR Word '%s' is not found in vocabulary;"
                        " moreover, <unk> is not found as well\n", t);
                exit(1);
            }
        }
        wids.push_back(wid);
    }
    return wids;
}

Real getWeightedProb(const NetData& nd, const std::vector<WordIndex> wids, int pos, bool dynamicMaxentPruning,
        const std::string& curr_chars)
{
    if (wids.size()<=1 || pos==0) {
        return 0;
    }
    auto weight = (Real) pos/(wids.size()-1);
    const auto& output = nd.updater->GetOutputMatrix();
    PropagateForward(nd.nnet, wids.data(), wids.size(), nd.updater);

    uint64_t ngram_hashes[MAX_NGRAM_ORDER];
    int maxent_present = CalculateMaxentHashIndices(nd.nnet, wids.data(), pos, ngram_hashes);
    auto prob = nd.nnet->softmax_layer->CalculateLog10Probability(
            wids[pos], ngram_hashes, maxent_present, dynamicMaxentPruning,
            output.row(pos-1).data(), &nd.nnet->maxent_layer, curr_chars);
    return prob*weight;
}

Real MixtureNet::Log10WordProbability(const std::string& sentence, int wordPos)
{
    auto wids = GetWids(sentence);
    return Log10WordProbability(wids, wordPos);
}

Real MixtureNet::Log10WordProbability(const std::vector<WordIndex>& wids, int wordPos)
{
    std::vector<WordIndex> paddedWids;
    paddedWids.push_back(0);
    paddedWids.insert(paddedWids.end(), wids.begin(), wids.end());
    paddedWids.push_back(0);

    auto word = std::string(forward_.nnet->vocab.GetWordByIndex(wids[wordPos]));

    std::vector<WordIndex> reverseWids(paddedWids.rbegin(), paddedWids.rend());
    auto forwardProb = getWeightedProb(forward_, paddedWids, wordPos+1, kDynamicMaxentPruning, word);
    auto reverseProb = getWeightedProb(reverse_, reverseWids, wids.size()-wordPos, kDynamicMaxentPruning, word);
    return reverseProb+forwardProb;
}
