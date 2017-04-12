//
// Created by abaev on 4/5/2017.
//

#ifndef FASTER_RNNLM_MIXTURENET_H
#define FASTER_RNNLM_MIXTURENET_H

#include <faster-rnnlm/layers/interface.h>
#include "nnet.h"

struct NetData {
  NNet *nnet;
  IRecUpdater *updater;

  NetData(NNet *net)
      : nnet(net), updater(net ? net->rec_layer->CreateUpdater() : nullptr) {}

  ~NetData() { if (updater) delete updater; }
};

class MixtureNet {
 public:
  explicit MixtureNet(NNet *forward, NNet *reverse, bool dynamicPruning);

  std::vector<WordIndex> GetWids(const std::string &sentence) const;

  Real Log10WordProbability(const std::string &sentence, int wordPos);

  Real Log10WordProbability(const std::vector<WordIndex> &wids, int wordPos, std::string *currWord = nullptr);

  const Tree *GetHSTree() const { return forward_.nnet->softmax_layer->GetTree(); }

  NNet *GetForwardNet() const { return forward_.nnet; }

  NNet *GetReverseNet() const { return reverse_.nnet; }

  std::string GetWordByIndex(WordIndex idx) const { return forward_.nnet->vocab.GetWordByIndex(idx); }

  const Vocabulary &GetVocabulary() const { return forward_.nnet->vocab; }

 private:
  const NetData forward_;
  const NetData reverse_;
  const WordIndex sen_start_wid_;
  const WordIndex sen_end_wid_;
  const bool kDynamicMaxentPruning;
};

#endif //FASTER_RNNLM_MIXTURENET_H
