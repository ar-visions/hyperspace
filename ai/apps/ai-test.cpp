#include <ai/ai.hpp>
#include <ai/gen.hpp>
#include <media/image.hpp>

using namespace ion;

/// simple inference test of model on image (resamples in ai if needed)
int main(int ac, cchar_t* av[]) {
    map<mx> def = map<mx> {
        {"model", str("")},
        {"image", str("")}
    };
    ///
    mx       ar = map<mx>::args(ac, av, def);
    image    im = path_t(ar["image"]);
    AI       ai = path_t(ar["model"]);
    ///
    console.test(img, "no-image");
    console.test(ai,  "no-model")
    ///
    array<float> inf = ai({ im }); // inf, inference. infinite possibilities kind of
    console.log("inference: {0}", { inf });
    return 0;
}
