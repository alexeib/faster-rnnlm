//
// Created by abaev on 4/9/2017.
//

#ifndef FASTER_RNNLM_CHAREMBEDDING_H
#define FASTER_RNNLM_CHAREMBEDDING_H

#include <string>
#include <vector>
#include <memory>
#include "settings.h"

class CharEmbedding {
 public:
  virtual int size() const = 0;

  virtual std::vector<double> get_embedding(const std::string &word) const = 0;
};

class NoopCharEmbedding : public CharEmbedding {
  int size() const override {
    return 0;
  }

  std::vector<double> get_embedding(const std::string &word) const override {
    return {};
  }
};

class PaddingLm1bCharEmbedding : public CharEmbedding {
 public:
  explicit PaddingLm1bCharEmbedding();

  int size() const override {
    return size_;
  }

  std::vector<double> get_embedding(const std::string &word) const override;

 private:
  static constexpr int emb_size_ = 50;
  static constexpr int emb_dims_ = 16;
  static constexpr int size_ = emb_dims_*emb_size_;
  static constexpr int pad_ = 4;
  std::vector<std::vector<double>> embeddings_by_char_;

  void add_emb(char c, std::vector<double> &embs) const;
};

class SquashedLm1bCharEmbedding : public CharEmbedding {
 public:
  explicit SquashedLm1bCharEmbedding();

  int size() const override {
    return size_;
  }

  std::vector<double> get_embedding(const std::string &word) const override;

 private:
  static constexpr int emb_dims_ = 16;
  static constexpr int size_ = emb_dims_;
  std::vector<std::vector<double>> embeddings_by_char_;
};

class CharEmbeddingFactory {
 public:
  static std::shared_ptr<CharEmbedding> create() {
    return std::shared_ptr<CharEmbedding>(new SquashedLm1bCharEmbedding());
  }
};

#endif //FASTER_RNNLM_CHAREMBEDDING_H
