#ifndef MIHON_FLOAT_TILE_PROCESSOR_H
#define MIHON_FLOAT_TILE_PROCESSOR_H
#include "periodic_texture_guard.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>

namespace float_tiles {
inline int reflect(int coordinate, int size) {
  if (size <= 1) return 0;
  while (coordinate < 0 || coordinate >= size)
    coordinate = coordinate < 0 ? -coordinate : 2 * size - 2 - coordinate;
  return coordinate;
}

// Same tile borders, texture protection and alpha handling as the QNN path.
template <typename Run>
int process(const uint8_t *input, int width, int height, int input_stride,
            uint8_t *output, int output_stride, int scale, int padding,
            periodic_texture_guard::Analysis &analysis,
            std::atomic<int> *progress, const std::atomic<bool> *abort, Run run) {
  constexpr int tile = 256;
  const int core = tile - 2 * padding, output_tile = tile * scale;
  if (!input || !output || width <= 0 || height <= 0 || core <= 0 ||
      (scale != 2 && scale != 3)) return -1;
  const auto &plan = analysis.plan(core);
  const periodic_texture_guard::AxisTransform guard(tile);
  const int total = ((width + core - 1) / core) * ((height + core - 1) / core);
  int completed = 0;
  std::vector<float> in(tile * tile * 3), out(output_tile * output_tile * 3);
  for (int ty = 0; ty < height; ty += core) {
    for (int tx = 0; tx < width; tx += core) {
      if (abort && abort->load()) return -2;
      const bool guarded = plan.guards(tx, ty);
      auto pixel = [&](int x, int y, int c) {
        return input[static_cast<size_t>(reflect(y, height)) * input_stride +
                     static_cast<size_t>(reflect(x, width)) * 4 + c] / 255.0f;
      };
      for (int y = 0; y < tile; ++y) for (int x = 0; x < tile; ++x) {
        for (int c = 0; c < 3; ++c) {
          float v;
          if (!guarded) {
            v = pixel(tx - padding + x, ty - padding + y, c);
          } else if (x < guard.border || y < guard.border ||
                     x >= guard.border + guard.scaled_length ||
                     y >= guard.border + guard.scaled_length) {
            v = 1;
          } else {
            float sx = tx - padding + guard.source_coordinate(x);
            float sy = ty - padding + guard.source_coordinate(y);
            int x0 = static_cast<int>(std::floor(sx)), y0 = static_cast<int>(std::floor(sy));
            float fx = sx - x0, fy = sy - y0;
            v = (pixel(x0, y0, c) * (1-fx) + pixel(x0+1, y0, c) * fx) * (1-fy) +
                (pixel(x0, y0+1, c) * (1-fx) + pixel(x0+1, y0+1, c) * fx) * fy;
          }
          in[(y * tile + x) * 3 + c] = v;
        }
      }
      if (!run(in, out)) return -1;
      if (out.size() != static_cast<size_t>(output_tile * output_tile * 3)) return -1;
      for (float value : out) if (!std::isfinite(value)) return -1;
      if (abort && abort->load()) return -2;
      auto sample = [&](int x, int y, int c) {
        return out[(static_cast<size_t>(y) * output_tile + x) * 3 + c];
      };
      const int cw = std::min(core, width - tx) * scale;
      const int ch = std::min(core, height - ty) * scale;
      // Value this tile predicts for output coordinate (x, y) of its own tile,
      // i.e. exactly what the hard copy below writes. `x`/`y` are core-local.
      auto value_at = [&](int x, int y, int c) {
        if (!guarded) return sample(padding * scale + x, padding * scale + y, c);
        float sx = guard.restored_output_coordinate(x, padding, scale);
        float sy = guard.restored_output_coordinate(y, padding, scale);
        int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, output_tile - 1);
        int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, output_tile - 1);
        int x1 = std::min(x0 + 1, output_tile - 1), y1 = std::min(y0 + 1, output_tile - 1);
        float fx = sx - std::floor(sx), fy = sy - std::floor(sy);
        return (sample(x0, y0, c) * (1 - fx) + sample(x1, y0, c) * fx) * (1 - fy) +
               (sample(x0, y1, c) * (1 - fx) + sample(x1, y1, c) * fx) * fy;
      };
      // Feather into the already written neighbour before the hard copy. The
      // padding band is this tile's prediction of the rows/columns that belong to
      // the neighbour, so ramping through it removes the discontinuity where two
      // tiles meet.
      //
      // Band row i sits at ABSOLUTE output row ty*scale - feather + i, i.e. source
      // row ty - padding + i/scale. value_at(x, y) returns the prediction for
      // source row ty + y/scale (its unguarded form samples model row
      // padding*scale + y, and model row r is source row ty - padding + r/scale).
      // Equating the two gives y = i - padding*scale = i - feather.
      //
      // Verified empirically on the real model, not just algebraically: for each
      // band row compare this tile's prediction against the row the tile above
      // already wrote (tmp/r3_band_proof.py, realcugan-se-x2-no-denoise,
      // page-001, padding 18):
      //     mean |diff| with y = i - feather : 0.0039 .. 0.0178
      //     mean |diff| with y = i           : 0.0430 .. 0.0842   (5-15x worse)
      // Passing i straight through makes the band paint content from `padding`
      // source rows too late, which is the one-row seam line on every tile edge.
      const int feather = std::min({padding * scale, cw, ch});
      const bool feather_up = feather > 1 && ty > 0 && !plan.guards(tx, ty - core);
      const bool feather_left = feather > 1 && tx > 0 && !plan.guards(tx - core, ty);
      const int band_core_offset = -feather;
      if (feather > 1) {
        for (int i = 0; i < feather; ++i) {
          const float alpha = static_cast<float>(i) / (feather - 1);
          const int band_core = i + band_core_offset;
          if (feather_up) {
            uint8_t *band = output +
                            static_cast<size_t>(ty * scale - feather + i) * output_stride;
            for (int x = 0; x < cw; ++x) {
              for (int c = 0; c < 3; ++c) {
                const float current = band[(tx * scale + x) * 4 + c] / 255.0f;
                const float blended = current * (1.0f - alpha) + value_at(x, band_core, c) * alpha;
                band[(tx * scale + x) * 4 + c] =
                    static_cast<uint8_t>(std::clamp(blended, 0.0f, 1.0f) * 255 + 0.5f);
              }
            }
          }
          if (feather_left) {
            const int column = tx * scale - feather + i;
            for (int y = 0; y < ch; ++y) {
              uint8_t *row = output + static_cast<size_t>(ty * scale + y) * output_stride;
              for (int c = 0; c < 3; ++c) {
                const float current = row[column * 4 + c] / 255.0f;
                const float blended = current * (1.0f - alpha) + value_at(band_core, y, c) * alpha;
                row[column * 4 + c] =
                    static_cast<uint8_t>(std::clamp(blended, 0.0f, 1.0f) * 255 + 0.5f);
              }
            }
          }
        }
      }
      for (int y = 0; y < ch; ++y) {
        uint8_t *row = output + static_cast<size_t>(ty * scale + y) * output_stride + tx * scale * 4;
        for (int x = 0; x < cw; ++x) {
          for (int c = 0; c < 3; ++c) {
            row[x*4+c] = static_cast<uint8_t>(std::clamp(value_at(x, y, c), 0.0f, 1.0f) * 255 + 0.5f);
          }
          row[x*4+3] = input[static_cast<size_t>(std::min(height-1, ty+y/scale)) * input_stride +
                                std::min(width-1, tx+x/scale) * 4 + 3];
        }
      }
      if (progress) progress->store(++completed * 100 / total);
    }
  }
  return 0;
}
} // namespace float_tiles
#endif
