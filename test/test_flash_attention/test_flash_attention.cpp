#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using std::vector;

// Q, K, V: [N, D]
// O:       [N, D]
void attention_v3_row_scores(const vector<float>& Q, const vector<float>& K,
                             const vector<float>& V, vector<float>& O, int N,
                             int D) {
  float scale = 1.0f / std::sqrt(static_cast<float>(D));

  vector<float> scores(N, 0.0f);
  O.assign(N * D, 0.0f);

  for (int i = 0; i < N; ++i) {
    float row_max = -std::numeric_limits<float>::infinity();
    // 1. Compute one row of scores
    for (int j = 0; j < N; ++j) {
      float score = 0.0f;
      for (int d = 0; d < D; ++d) {
        score += Q[i * D + d] * K[j * D + d];
      }

      score *= scale;
      scores[j] = score;
      row_max = std::max(row_max, score);
    }

    // 2. Compute denominator
    float row_sum_exp = 0.0f;

    for (int j = 0; j < N; ++j) {
      scores[j] = std::exp(scores[j] - row_max);
      row_sum_exp += scores[j];
    }

    // 3. Multiply softmax row by V
    for (int d = 0; d < D; ++d) {
      float out = 0.0f;
      for (int j = 0; j < N; ++j) {
        float prob = scores[j] / row_sum_exp;
        out += prob * V[j * D + d];
      }
      O[i * D + d] = out;
    }
  }
}