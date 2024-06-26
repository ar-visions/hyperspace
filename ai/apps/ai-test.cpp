#include <ai/ai.hpp>
#include <ai/gen.hpp>
#include <media/media.hpp>

using namespace ion;

/// simple inference test of model on image (resamples in ai if needed)
int main(int ac, cchar_t* av[]) {
    map def = map {
        {"model", str("")},
        {"image", str("")}
    };
    ///
    mx       ar = map::args(ac, av, def);
    image    im = path_t(ar["image"]);
    AI       ai = path_t(ar["model"]);
    ///
    console.test(img, "no-image");
    console.test(ai,  "no-model")
    ///
    Array<float> inf = ai({ im }); // inf, inference. infinite possibilities kind of
    console.log("inference: {0}", { inf });
    return 0;
}
