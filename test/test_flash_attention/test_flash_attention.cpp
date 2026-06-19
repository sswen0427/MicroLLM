#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using std::vector;

// Q, K, V: [N, D]
// O:       [N, D]
void attention_v5_flash_style_block_online(const vector<float>& Q,
                                           const vector<float>& K,
                                           const vector<float>& V,
                                           vector<float>& O, int N, int D,
                                           int block_size) {
  float scale = 1.0f / std::sqrt(static_cast<float>(D));

  O.assign(N * D, 0.0f);

  for (int i = 0; i < N; ++i) {
    float running_max = -std::numeric_limits<float>::infinity();
    float running_sum_exp = 0.0f;
    vector<float> running_weighted_value_sum(D, 0.0f);

    for (int block_start = 0; block_start < N; block_start += block_size) {
      int block_end = std::min(block_start + block_size, N);
      int current_block_size = block_end - block_start;

      vector<float> block_scores(current_block_size, 0.0f);

      float block_max = -std::numeric_limits<float>::infinity();

      // 1. Compute scores for this K/V block
      for (int j = block_start; j < block_end; ++j) {
        float score = 0.0f;

        for (int d = 0; d < D; ++d) {
          score += Q[i * D + d] * K[j * D + d];
        }

        score *= scale;

        block_scores[j - block_start] = score;
        block_max = std::max(block_max, score);
      }

      // 2. Compute local block softmax state
      float block_sum_exp = 0.0f;
      vector<float> block_weighted_value_sum(D, 0.0f);

      for (int j = block_start; j < block_end; ++j) {
        float weight = std::exp(block_scores[j - block_start] - block_max);

        block_sum_exp += weight;

        for (int d = 0; d < D; ++d) {
          block_weighted_value_sum[d] += weight * V[j * D + d];
        }
      }

      // 3. Merge old running state with current block state
      float new_running_max = std::max(running_max, block_max);

      float old_scale = std::exp(running_max - new_running_max);

      float block_scale = std::exp(block_max - new_running_max);

      float new_running_sum_exp =
          running_sum_exp * old_scale + block_sum_exp * block_scale;

      for (int d = 0; d < D; ++d) {
        running_weighted_value_sum[d] =
            running_weighted_value_sum[d] * old_scale +
            block_weighted_value_sum[d] * block_scale;
      }

      running_max = new_running_max;
      running_sum_exp = new_running_sum_exp;
    }

    // 4. Normalize
    for (int d = 0; d < D; ++d) {
      O[i * D + d] = running_weighted_value_sum[d] / running_sum_exp;
    }
  }
}