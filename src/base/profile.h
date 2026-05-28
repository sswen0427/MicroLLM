#pragma once

#include <chrono>

namespace base {

class ScopedProfile {
 public:
  explicit ScopedProfile(double& elapsed_ms)
      : elapsed_ms_(elapsed_ms), start_(Clock::now()) {}

  ScopedProfile(const ScopedProfile&) = delete;
  ScopedProfile& operator=(const ScopedProfile&) = delete;
  ScopedProfile(ScopedProfile&&) = delete;
  ScopedProfile& operator=(ScopedProfile&&) = delete;

  ~ScopedProfile() {
    elapsed_ms_ +=
        std::chrono::duration<double, std::milli>(Clock::now() - start_)
            .count();
  }

 private:
  using Clock = std::chrono::steady_clock;

  double& elapsed_ms_;
  Clock::time_point start_;
};

}  // namespace base
