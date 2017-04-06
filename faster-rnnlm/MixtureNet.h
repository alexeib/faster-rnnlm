//
// Created by abaev on 4/5/2017.
//

#ifndef FASTER_RNNLM_MIXTURENET_H
#define FASTER_RNNLM_MIXTURENET_H

#include <faster-rnnlm/layers/interface.h>
#include "nnet.h"

struct NetData {
    NNet* nnet;
    IRecUpdater* updater;

    NetData(NNet* net)
            :nnet(net), updater(net->rec_layer->CreateUpdater()) { }

    ~NetData() { delete updater; }
};

class MixtureNet {
public:
    explicit MixtureNet(NNet* forward, NNet* reverse, bool dynamicPruning);

    std::vector<WordIndex> GetWids(std::string& sentence) const;

    Real Log10WordProbability(std::string& sentence, int wordPos);

    Real Log10WordProbability(std::vector<WordIndex>& wids, int wordPos);

    const Tree* GetHSTree() const { return forward_.nnet->softmax_layer->GetTree(); }

    NNet* GetForwardNet() const { return forward_.nnet; }

    NNet* GetReverseNet() const { return forward_.nnet; }

    std::string GetWordByIndex(WordIndex idx) { return forward_.nnet->vocab.GetWordByIndex(idx); }

private:
    const NetData forward_;
    const NetData reverse_;
    const bool kDynamicMaxentPruning;
};

#endif //FASTER_RNNLM_MIXTURENET_H
