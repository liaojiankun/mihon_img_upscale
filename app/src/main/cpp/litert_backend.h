#ifndef MIHON_LITERT_BACKEND_H
#define MIHON_LITERT_BACKEND_H
#include "periodic_texture_guard.h"
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

// Pure declaration-driven layout resolution for the MediaTek LiteRT models.
// Header-only and free of Android/LiteRT types so the exact same code the
// device build compiles can be regression tested on a host compiler
// (app/src/test/cpp/mediatek_layout_test.cpp).
//
// The 15 shipped models do not all declare the shape the caller asks for:
//   * 14 models declare NCHW [1,3,S,S]; span-nomosuni declares NHWC [1,S,S,3].
//   * realcugan pro x3 declares 792 = 3*(256+2*4): the graph keeps an 8 pixel
//     input pad, which shows up as a 12 pixel border on each side of the
//     792 pixel result. Trimming that border yields 768 = 3*256.
//   * the two Real-ESRGAN models are 4x native (1024 = 4*256) while the app
//     asks for 2x page geometry. The Vulkan path handles the same weights by
//     resizing the 4x result down (realesrgan-models/v3-anime/x2.param ends
//     with "Interp Resize_40 ... 0.5"), so here the 4x result is reduced by an
//     exact 2:1 box average after the border crop.
namespace mediatek_layout {
enum class Layout { Nhwc, Nchw };

struct Dispatch {
  bool ok = false;
  Layout layout = Layout::Nhwc;
  // Side length the model actually declares.
  int output_side = 0;
  // Border trimmed from each side of the declared result, in output pixels.
  int crop = 0;
  // Integer area reduction applied after the crop (1 = none, 2 = 2:1 box).
  int downsample = 1;
  // 256 * requested scale: the side float_tile_processor indexes at.
  int target_side = 0;
  const char *error = "";
};

constexpr int kInputSide = 256;
constexpr int kMinimumNativeScale = 2;
constexpr int kMaximumNativeScale = 4;
constexpr int kRequiredRank = 4;
// A declared side only counts as a padded graph when the implied pad is small,
// so a 1024 side is read as 4x rather than as 2x with a 256 pixel border.
constexpr int kMaximumPad = 32;

inline Dispatch reject(Dispatch result, const char *error) {
  result.ok = false;
  result.error = error;
  return result;
}

// dims is the declared output dimensions (rank entries). requested_scale is
// the value MediaTekNpu.kt computes (2, or 3 when the name contains "-x3-").
inline Dispatch dispatch(const int32_t *dims, int rank, int requested_scale) {
  Dispatch result;
  if (requested_scale != 2 && requested_scale != 3) {
    return reject(result, "Unsupported model scale");
  }
  if (!dims || rank != kRequiredRank) {
    return reject(result, "Expected a rank-4 output tensor");
  }
  if (dims[0] != 1) {
    return reject(result, "Expected a single-image batch");
  }
  if (dims[3] == 3 && dims[1] > 0 && dims[1] == dims[2]) {
    result.layout = Layout::Nhwc;
    result.output_side = dims[1];
  } else if (dims[1] == 3 && dims[2] > 0 && dims[2] == dims[3]) {
    result.layout = Layout::Nchw;
    result.output_side = dims[2];
  } else {
    return reject(result, "Unsupported output layout: expected NHWC [1,S,S,3] or NCHW [1,3,S,S]");
  }

  const int side = result.output_side;
  int native = 0, crop = 0, matches = 0;
  // Exact graphs first, then padded ones, so a 768-declaring 3x model is not
  // also read as a 2x graph with a 128 pixel border.
  for (int pass = 0; pass < 2 && matches == 0; ++pass) {
    for (int candidate = kMinimumNativeScale; candidate <= kMaximumNativeScale; ++candidate) {
      if (side % candidate != 0) continue;
      const int core = side / candidate;
      if (core < kInputSide) continue;
      const int pad = core - kInputSide;
      if (pad % 2 != 0 || pad / 2 > kMaximumPad) continue;
      if ((pass == 0) != (pad == 0)) continue;
      ++matches;
      native = candidate;
      crop = pad / 2 * candidate;
    }
  }
  if (matches == 0) {
    return reject(result, "Unsupported output side: expected 256*n scaled for 2x, 3x or 4x");
  }
  if (matches > 1) {
    return reject(result, "Ambiguous model output side");
  }

  const int target = kInputSide * requested_scale;
  const int effective = side - 2 * crop;
  if (effective % target != 0) {
    return reject(result, "Model output side does not match the requested scale");
  }
  const int downsample = effective / target;
  if (downsample != 1 && downsample != 2) {
    return reject(result, "Unsupported output reduction factor");
  }
  // No shipped graph has both a padding border and a 4x native factor; treat
  // such a declaration as a doubt rather than guess a geometry for it.
  if (crop != 0 && downsample != 1) {
    return reject(result, "Padded output with an unexpected native scale");
  }
  result.crop = crop;
  result.downsample = downsample;
  result.target_side = target;
  result.ok = true;
  return result;
}
} // namespace mediatek_layout

namespace litert_backend {
bool initialize(const std::string &model, const std::string &cache,
                const std::string &libraries, int scale, int padding,
                bool cpu_test_only = false);
bool is_initialized();
bool is_npu_active();
std::string last_error();
bool run_tile(const std::vector<float> &input, std::vector<float> &output);
int process_rgba(const uint8_t *input, int width, int height, int input_stride,
                 uint8_t *output, int output_stride,
                 periodic_texture_guard::Analysis &analysis,
                 std::atomic<int> *progress, const std::atomic<bool> *abort);
void shutdown();
}
#endif
