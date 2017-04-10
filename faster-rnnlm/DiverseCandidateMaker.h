#ifndef FASTER_RNNLM_DIVERSECANDIDATEMAKER_H
#define FASTER_RNNLM_DIVERSECANDIDATEMAKER_H

#include "maxent.h"
#include "hierarchical_softmax.h"
#include "MixtureNet.h"

struct DirData {
    const uint64_t* feature_hashes;
    int maxent_present;
    const NNet* nnet;
    const Real* hidden;
    IRecUpdater* updater;

    ~DirData()
    {
        delete updater;
        delete[] feature_hashes;
    }
};

class DiverseCandidateMaker {
public:
    explicit DiverseCandidateMaker(MixtureNet& mixtureNet)
            :mn_(mixtureNet) { }

    std::vector<WordIndex>
    DiverseCandidates(const std::vector<WordIndex>& wids, int pos, int target_number,
            bool dynamic_maxent_prunning) const;

private:
    const MixtureNet& mn_;

    std::vector<WordIndex>
    DiverseCandidates(int node, int curr_depth, int target_depth, bool dynamic_maxent_prunning,
            int sentence_length, int word_pos, const DirData& forward, const DirData& reverse,
            const std::string& curr_word) const;
};

#endif //FASTER_RNNLM_DIVERSECANDIDATEMAKER_H
