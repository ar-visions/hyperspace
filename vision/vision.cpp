#include <vision/vision.hpp>
#include <FastNoiseLite.h>

namespace ion {

/// run independent simplex across levels at scale[levels]; user outputs what they want with it
image simplex_equirect(int64_t seed, int width, int height, float *scales, int levels, lambda<void(rgba8*, float*)> out) {
    FastNoiseLite **simplex = (FastNoiseLite**)calloc(sizeof(FastNoiseLite*), levels);
    float *values = (float*)calloc(sizeof(float), levels);

    for (int i = 0; i < levels; i++) {
        simplex[i] = new FastNoiseLite(1337 * i + i + seed);
        simplex[i]->SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    }

    image img { size { height, width } };
    rgba8 *pixels = img.pixels();
    rgba8 *px     = pixels;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++, px++) {
            // convert pixel coordinates to equirectangular coordinates
            // (lets use double so we can get a better unit vector for pos)
            double lat = glm::radians(180.0 * (y / double(height-1) - 0.5));
            double lon = glm::radians(360.0 *  x / double(width)    - 180.0);
            
            // convert equirectangular coordinates to 3D cartesian coordinates (normalized)
            vec3f pos(
                cos(lat) * cos(lon),
                sin(lat),
                cos(lat) * sin(lon));

            memset(values, 0, sizeof(float) * levels);
            for (int i = 0; i < levels; i++)
                values[i] = simplex[i]->GetNoise(
                     pos.x * scales[i],
                     pos.y * scales[i],
                     pos.z * scales[i]);

            out(px, values);
        }
    }

    for (int i = 0; i < levels; i++)
        delete simplex[i];
    
    free(simplex);
    free(values);
    return img;
}

image simplex_equirect_gray(int64_t seed, int width, int height, float *scales, int levels) {
    return simplex_equirect(seed, width, height, scales, levels, [&](rgba8* px, float* values) {
        float v = 0;
        for (int i = 0; i < levels; i++)
            v += values[i];
        v /= levels;
        v  = (v + 1.0f) / 2.0f * 255.0f;
        px->r = u8(v);
        px->g = u8(v);
        px->b = u8(v);
        px->a = 255;
    });
}

image simplex_equirect_normal(int64_t seed, int width, int height, float vary_degrees, float *scales) {
    return simplex_equirect(seed, width, height, scales, 3, [&](rgba8* px, float* values) {
        float rx = glm::radians(values[0] * vary_degrees);
        float ry = glm::radians(values[1] * vary_degrees);
        float rz = glm::radians(values[2] * vary_degrees);

        m44f      r = m44f(1.0f);
        r = glm::rotate(r, rx, vec3f(1.0f, 0.0f, 0.0f));
        r = glm::rotate(r, ry, vec3f(0.0f, 1.0f, 0.0f));
        r = glm::rotate(r, rz, vec3f(0.0f, 0.0f, 1.0f));

        /// the purply blue color is 128, 128, 255 or 0.0, 0.0, 1.0
        /// (we reserve half for the other sign with cartesian xyz to rgb)
        vec3f base_normal = vec3f(0.0f, 0.0f, 1.0f);
        vec3f v = r * vec4f(base_normal, 0.0f);

        px->r = u8((v.x + 1.0f) / 2.0f * 255.0f);
        px->g = u8((v.y + 1.0f) / 2.0f * 255.0f);
        px->b = u8((v.z + 1.0f) / 2.0f * 255.0f);
        px->a = 255;
    });
}

}