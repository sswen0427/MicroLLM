#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using std::vector;

// Q, K, V: [N, D]
// O:       [N, D]
void attention_v4_online_softmax(const vector<float>& Q, const vector<float>& K,
                                 const vector<float>& V, vector<float>& O,
                                 int N, int D) {
  float scale = 1.0f / std::sqrt(static_cast<float>(D));

  O.assign(N * D, 0.0f);

  for (int i = 0; i < N; ++i) {
    float running_max = -std::numeric_limits<float>::infinity();
    float running_sum_exp = 0.0f;
    vector<float> running_weighted_value_sum(D, 0.0f);

    for (int j = 0; j < N; ++j) {
      // 1. Compute current score
      float score = 0.0f;
      for (int d = 0; d < D; ++d) {
        score += Q[i * D + d] * K[j * D + d];
      }

      score *= scale;

      // 2. Online softmax update
      float new_running_max = std::max(running_max, score);

      float old_scale = std::exp(running_max - new_running_max);
      float new_scale = std::exp(score - new_running_max);

      float new_running_sum_exp = running_sum_exp * old_scale + new_scale;

      for (int d = 0; d < D; ++d) {
        running_weighted_value_sum[d] =
            running_weighted_value_sum[d] * old_scale +
            new_scale * V[j * D + d];
      }

      running_max = new_running_max;
      running_sum_exp = new_running_sum_exp;
    }

    // 3. Normalize
    for (int d = 0; d < D; ++d) {
      O[i * D + d] = running_weighted_value_sum[d] / running_sum_exp;
    }
  }
}