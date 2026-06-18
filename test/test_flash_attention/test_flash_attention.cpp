#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using std::vector;

// Q, K, V: [N, D]
// O:       [N, D]
void attention_v1_full_matrices(const vector<float>& Q, const vector<float>& K,
                                const vector<float>& V, vector<float>& O, int N,
                                int D) {
  float scale = 1.0f / std::sqrt(static_cast<float>(D));

  vector<float> S(N * N, 0.0f);
  vector<float> P(N * N, 0.0f);
  O.assign(N * D, 0.0f);

  // 1. S = QK^T / sqrt(D)
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float score = 0.0f;
      for (int d = 0; d < D; ++d) {
        score += Q[i * D + d] * K[j * D + d];
      }
      S[i * N + j] = score * scale;
    }
  }

  // 2. P = softmax(S), row by row
  for (int i = 0; i < N; ++i) {
    float row_max = -std::numeric_limits<float>::infinity();

    for (int j = 0; j < N; ++j) {
      row_max = std::max(row_max, S[i * N + j]);
    }

    float row_sum_exp = 0.0f;

    for (int j = 0; j < N; ++j) {
      float value = std::exp(S[i * N + j] - row_max);
      P[i * N + j] = value;
      row_sum_exp += value;
    }

    for (int j = 0; j < N; ++j) {
      P[i * N + j] /= row_sum_exp;
    }
  }

  // 3. O = P V
  for (int i = 0; i < N; ++i) {
    for (int d = 0; d < D; ++d) {
      float out = 0.0f;
      for (int j = 0; j < N; ++j) {
        out += P[i * N + j] * V[j * D + d];
      }
      O[i * D + d] = out;
    }
  }
}