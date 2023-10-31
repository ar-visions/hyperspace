#include <net/net.hpp>
#include <ux/ux.hpp>

namespace ion {

image simplex_equirect_gray(int64_t seed, int width, int height, float *scales, int levels);
image simplex_equirect_normal(int64_t seed, int width, int height, float vary_degrees, float *scales);
image simplex_equirect(int64_t seed, int width, int height, float *scales, int levels, lambda<void(rgba8*, float*)> out);

}