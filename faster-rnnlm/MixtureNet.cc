//
// Created by abaev on 4/5/2017.
//

#include "MixtureNet.h"

MixtureNet::MixtureNet(NNet* forward, NNet* reverse, bool dynamicPruning)
        :forward_(forward), reverse_(reverse), kDynamicMaxentPruning(dynamicPruning)
{
}

template<typename Out>
void split(const std::string& s, char delim, Out result)
{
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty())
            *(result++) = item;
    }
}

std::vector<WordIndex> MixtureNet::GetWids(std::string& sentence) const
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

Real getWeightedProb(const NetData& nd, const std::vector<WordIndex> wids, int pos, bool dynamicMaxentPruning)
{
    if (wids.size()<=1 || pos==0) {
        return 0;
    }
    auto weight = (Real) pos/(wids.size()-1);
    const auto& output = nd.updater->GetOutputMatrix();
    PropagateForward(nd.nnet, wids.data(), wids.size(), nd.updater);

    uint64_t ngram_hashes[MAX_NGRAM_ORDER];
    int maxent_present = CalculateMaxentHashIndices(nd.nnet, wids.data(), pos, ngram_hashes);
    return pow(10., nd.nnet->softmax_layer->CalculateLog10Probability(
            wids[pos], ngram_hashes, maxent_present, dynamicMaxentPruning,
            output.row(pos-1).data(), &nd.nnet->maxent_layer));
}

Real MixtureNet::Log10WordProbability(std::string& sentence, int wordPos)
{
    auto wids = GetWids(sentence);
    return Log10WordProbability(wids, wordPos);
}

Real MixtureNet::Log10WordProbability(std::vector<WordIndex>& wids, int wordPos)
{
    std::vector<WordIndex> reverseWids(wids.rbegin(), wids.rend());
    auto forwardProb = getWeightedProb(forward_, wids, wordPos, kDynamicMaxentPruning);
    auto reverseProb = getWeightedProb(reverse_, reverseWids, wids.size()-1-wordPos, kDynamicMaxentPruning);
    return reverseProb+forwardProb;
}
