//
// Created by abaev on 4/9/2017.
//

#include "CharEmbedding.h"
#include "util.h"
#include <fstream>

PaddingLm1bCharEmbedding::PaddingLm1bCharEmbedding() {
  std::ifstream infile("char_embeddings.csv");
  std::string line;
  while (std::getline(infile, line)) {
    std::vector<std::string> v;
    split(line, ',', std::back_inserter(v));
    std::vector<double> emb(emb_dims_);
    std::transform(v.begin(), v.end(), emb.begin(),
                   [](auto &s) { return atof(s.c_str()); });
    embeddings_by_char_.emplace_back(std::move(emb));
  }
}

std::vector<double>
PaddingLm1bCharEmbedding::get_embedding(const std::string &word) const {
  std::vector<double> embs;
  for (int i = 0; i < emb_size_; ++i) {
    if (i >= word.length()) {
      add_emb(pad_, embs);
    } else {
      add_emb(word[i], embs);
    }
  }
  return embs;
}

void PaddingLm1bCharEmbedding::add_emb(char c,
                                       std::vector<double> &embs) const {
  std::copy(embeddings_by_char_[c].begin(), embeddings_by_char_[c].end(),
            std::back_inserter(embs));
}

SquashedLm1bCharEmbedding::SquashedLm1bCharEmbedding() {
  std::ifstream infile("char_embeddings.csv");
  std::string line;
  while (std::getline(infile, line)) {
    std::vector<std::string> v;
    split(line, ',', std::back_inserter(v));
    std::vector<double> emb(emb_dims_);
    std::transform(v.begin(), v.end(), emb.begin(),
                   [](auto &s) { return atof(s.c_str()); });
    embeddings_by_char_.emplace_back(std::move(emb));
  }
}

std::vector<double>
SquashedLm1bCharEmbedding::get_embedding(const std::string &word) const {
  std::vector<double> embs(emb_dims_);
  int i =0;
  for (unsigned char c : word) {
    if (c >= 256)
      continue;
    auto &ec = embeddings_by_char_[c];
    for (int i = 0; i < ec.size(); ++i) {
      embs[i] += ec[i];
    }
  }
  //    for (int i = 0; i<emb_dims_; ++i) {
  //        embs[i] /
  //    }
  return embs;
}
