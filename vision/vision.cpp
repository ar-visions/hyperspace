#include <vision/vision.hpp>
#include <FastNoiseLite.h>

image simplex_equirect(int width, int height, float scale) {
    FastNoiseLite fn = FastNoiseLite(1337 + (int)millis());
    fn.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    image img { size { height, width } };
    rgba8 *pixels = img.pixels();
    rgba8 *px = pixels;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++, px++) {
            
            // Convert pixel coordinates to equirectangular coordinates
            double lat = glm::radians(180.0 * (y / double(height-1) - 0.5));
            double lon = glm::radians(360.0 *  x / double(width)    - 180.0);
            
            // Convert equirectangular coordinates to 3D cartesian coordinates
            glm::vec3 pos(
                cos(lat) * cos(lon) * width * scale,
                sin(lat) * width * scale,
                cos(lat) * sin(lon) * width * scale
            );
            
            const int levels = 1;
            float values[levels];
            float av = 1.0;

            for (int i = 0; i < levels; i++) {
                float n = fn.GetNoise(pos.x / (1 + (i * 4)), pos.y / (1 + (i * 4)), pos.z / (1 + (i * 4)));
                values[i] = (1 + n) / 2;
                av *= values[i];
            }

            av *= levels;
            av  = math::clamp(av, 0.0f, 1.0f);

            // store noise
            u8  v = av * 255.0;
            px->r = v;
            px->g = v;
            px->b = v;
            px->a = 255;
        }
    }
    return img;
}