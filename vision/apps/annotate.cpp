/// ux apps are both gfx and 3D context.  the 3D background can be reality or a game or a picture of both.
/// all of the scenes are staged differently
/// issue is accessing the members of their View, perhaps not a deal at all?

/// 1. add 3D ops to canvas.  
/// 2. ability to detect when mouse is within range of shapes, which axis to select
/// 3. integration into Element in subtle way

#include <ux/app.hpp>

using namespace ion;

struct View:Element {
    struct props {
        float       angle;
        int         sample;
        int         sample2;
        callback    clicked;
        doubly<prop> meta() {
            return {
                prop { "sample",  sample },
                prop { "sample2", sample2 },
                prop { "clicked", clicked}
            };
        }
        type_register(props);
    };
    component(View, Element, props);

    void draw(Canvas& canvas) {
        static rgbad white = { 1.0, 1.0, 1.0, 1.0 };

        std::array<glm::vec3, 24> cube_edges = {
            // AB
            glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.5f, -0.5f, -0.5f),
            // BC
            glm::vec3(0.5f, -0.5f, -0.5f), glm::vec3(0.5f, 0.5f, -0.5f),
            // CD
            glm::vec3(0.5f, 0.5f, -0.5f), glm::vec3(-0.5f, 0.5f, -0.5f),
            // DA
            glm::vec3(-0.5f, 0.5f, -0.5f), glm::vec3(-0.5f, -0.5f, -0.5f),
            // AE
            glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(-0.5f, -0.5f, 0.5f),
            // BF
            glm::vec3(0.5f, -0.5f, -0.5f), glm::vec3(0.5f, -0.5f, 0.5f),
            // CG
            glm::vec3(0.5f, 0.5f, -0.5f), glm::vec3(0.5f, 0.5f, 0.5f),
            // DH
            glm::vec3(-0.5f, 0.5f, -0.5f), glm::vec3(-0.5f, 0.5f, 0.5f),
            // EF
            glm::vec3(-0.5f, -0.5f, 0.5f), glm::vec3(0.5f, -0.5f, 0.5f),
            // FG
            glm::vec3(0.5f, -0.5f, 0.5f), glm::vec3(0.5f, 0.5f, 0.5f),
            // GH
            glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(-0.5f, 0.5f, 0.5f),
            // HE
            glm::vec3(-0.5f, 0.5f, 0.5f), glm::vec3(-0.5f, -0.5f, 0.5f)
        };

        static float angle;
        angle += 0.01;

        glm::vec2 sz    = { canvas.get_virtual_width(), canvas.get_virtual_height() };
        glm::mat4 proj  = glm::perspective(glm::radians(45.0f), sz.x / sz.y, 0.1f, 100.0f);
        glm::mat4 view  = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 model = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0.0f, 1.0f, 0.0f));

        canvas.save();
        canvas.projection(model, view, proj);
        canvas.color(white);
        canvas.outline_sz(2);
        for (size_t i = 0; i < 12; i++) {
            canvas.line(cube_edges[i * 2 + 0], cube_edges[i * 2 + 1]);
        }
        canvas.restore();
    }
};

int main(int argc, char *argv[]) {
    map<mx> defs  {{ "debug", uri { "ssh://ar-visions.com:1022" } }};
    map<mx> config { args::parse(argc, argv, defs) };
    if    (!config) return args::defaults(defs);
    return App(config, [](App &app) -> node {
        return View {
            { "id", "main" }
        };
    });
}
