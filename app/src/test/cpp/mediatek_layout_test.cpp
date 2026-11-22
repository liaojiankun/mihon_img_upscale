// Host regression test for the MediaTek output-geometry resolver; no Android,
// LiteRT or NCNN runtime is required:
//   c++ -std=c++17 -O2 -Wall -Wextra -Werror mediatek_layout_test.cpp -o mediatek_layout_test
// This pins the C++ rule (app/src/main/cpp/litert_backend.h). The JVM test
// app/src/test/java/eu/kanade/tachiyomi/util/mediatek/MediaTekModelLayoutTest.kt
// pins the same table through the Kotlin view of the rule and additionally
// re-reads the declared shapes out of the 15 shipped .tflite files.
#include "../../main/cpp/litert_backend.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void require(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

using Dims = std::vector<int32_t>;

void expect(const std::string &name, Dims dims, int requested,
            mediatek_layout::Layout layout, int side, int crop, int downsample,
            int target) {
  const auto plan = mediatek_layout::dispatch(dims.data(), static_cast<int>(dims.size()), requested);
  require(plan.ok, name + ": expected success, got " + plan.error);
  if (!plan.ok) return;
  require(plan.layout == layout, name + ": layout");
  require(plan.output_side == side, name + ": output_side");
  require(plan.crop == crop, name + ": crop");
  require(plan.downsample == downsample, name + ": downsample");
  require(plan.target_side == target, name + ": target_side");
}

void expect_rejected(const std::string &name, Dims dims, int requested) {
  const auto plan = mediatek_layout::dispatch(dims.data(), static_cast<int>(dims.size()), requested);
  require(!plan.ok, name + ": expected rejection");
  require(plan.error != nullptr && plan.error[0] != '\0', name + ": empty error");
}

} // namespace

int main() {
  using mediatek_layout::Layout;

  // The distinct shapes the 15 shipped models declare.
  expect("realcugan-se-x2-*", {1, 3, 512, 512}, 2, Layout::Nchw, 512, 0, 1, 512);
  expect("realcugan-pro-x2-*", {1, 3, 512, 512}, 2, Layout::Nchw, 512, 0, 1, 512);
  expect("realcugan-pro-x3-*", {1, 3, 792, 792}, 3, Layout::Nchw, 792, 12, 1, 768);
  expect("realesrgan-animevideov3-x2", {1, 3, 1024, 1024}, 2, Layout::Nchw, 1024, 0, 2, 512);
  expect("realesrgan-general-x4v3-x2", {1, 3, 1024, 1024}, 2, Layout::Nchw, 1024, 0, 2, 512);
  expect("span-nomosuni-x2", {1, 512, 512, 3}, 2, Layout::Nhwc, 512, 0, 1, 512);
  expect("w2xex-photo-small-x2", {1, 3, 512, 512}, 2, Layout::Nchw, 512, 0, 1, 512);

  // Anything unexpected must fail with a message rather than degrade silently.
  expect_rejected("unsupported scale", {1, 3, 512, 512}, 4);
  expect_rejected("scale zero", {1, 3, 512, 512}, 0);
  expect_rejected("batch 2", {2, 3, 512, 512}, 2);
  expect_rejected("rank 3", {3, 512, 512}, 2);
  expect_rejected("channel mismatch", {1, 4, 512, 512}, 2);
  expect_rejected("4 channels", {1, 1024, 1024, 4}, 2);
  expect_rejected("768 declared for 2x", {1, 3, 768, 768}, 2);
  expect_rejected("1024 declared for 3x", {1, 3, 1024, 1024}, 3);
  expect_rejected("too small", {1, 3, 256, 256}, 2);
  expect_rejected("huge 2x border", {1, 3, 1536, 1536}, 2);
  expect_rejected("non-square", {1, 3, 512, 768}, 2);
  expect_rejected("empty dims", {}, 2);
  require(!mediatek_layout::dispatch(nullptr, 4, 2).ok, "nullptr dims must be rejected");

  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "mediatek_layout_test: all checks passed\n";
  return EXIT_SUCCESS;
}
