#include <vk/vk.hpp>

using namespace ion;

/// a useful tooling experiment to create an annotation cursor out of a rubiks cube model; location, unit scale and orientation.

struct Light {
    alignas(16) glm::vec4 pos;
    alignas(16) glm::vec4 color;
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;

    Vertex() { }
    Vertex(float *v_pos, int p_index, float *v_uv, int uv_index, float *v_normal, int n_index) {
        pos = {
            v_pos[3 * p_index + 0],
            v_pos[3 * p_index + 1],
            v_pos[3 * p_index + 2]
        };
        uv = glm::vec2 {
                   v_uv[2 * uv_index + 0],
            1.0f - v_uv[2 * uv_index + 1]
        };
        normal = { v_normal[3 * n_index  + 0], v_normal[3 * n_index + 1], v_normal[3 * n_index + 2] };
    }

    doubly<prop> meta() const {
        return {
            prop { "pos",      pos      },
            prop { "normal",   normal   },
            prop { "uv",       uv       }
        };
    }

    type_register(Vertex);

    bool operator==(const Vertex& b) const {
        return pos    == b.pos    && 
               normal == b.normal && 
               uv     == b.uv;
    }
};

/// used by uniqueVertices
namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^ 
                    (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^
                    (hash<glm::vec2>()(vertex.uv)     << 1);
        }
    };
}

struct UniformBufferObject;

struct Rubiks:mx {
    struct M {
        Vulkan          vk { 1, 0 };        /// this lazy loads 1.0 when GPU performs that action [singleton data]
        vec2i           sz { 1920, 1080 };  /// store current window size
        Window          gpu;                /// GPU class, responsible for holding onto GPU, Surface and GLFWwindow
        Device          device;             /// Device created with GPU
        Pipeline        pipeline;           /// pipeline for single object scene
        bool            design = true;      /// design mode

        static void resized(vec2i &sz, M* app) {
            app->sz = sz;
            app->device->framebufferResized = true;
        }

        void init() {
            gpu      = Window::select(sz, ResizeFn(resized), this);
            device   = Device::create(gpu);
            pipeline = Pipeline(
                Graphics<UniformBufferObject, Vertex>(device, "rubiks")
            );
        }

        void run() {
            while (!glfwWindowShouldClose(gpu->window)) {
                glfwPollEvents();
                device->mtx.lock();
                array<Pipeline> pipes = { pipeline };
                device->drawFrame(pipes);
                /// i need to do it here <-
                device->mtx.unlock();
            }
            vkDeviceWaitIdle(device);
        }

        register(M);
    };
    
    mx_basic(Rubiks);

    /// return the class in main() to loop and return exit-code
    operator int() {
        try {
            data->pipeline->user = mem->grab();
            data->run();
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
};

/// uniform has an update method with a pipeline arg
struct UniformBufferObject {
    alignas(16) glm::mat4  model;
    alignas(16) glm::mat4  view;
    alignas(16) glm::mat4  proj;
    alignas(16) glm::vec3  eye;
    alignas(16) Light      lights[3];

    void update(Pipeline::impl *pipeline) {
        VkExtent2D &ext = pipeline->device->swapChainExtent;

        Rubiks rubiks = pipeline->user.grab();
        bool   design = rubiks->design;

        eye   = glm::vec3(0.0f, 0.0f, 0.0f);

        image img = path { "textures/rubiks.color2.png" };
        pipeline->textures[Asset::color - 1].update(img); ///

        do {
            float min_z = 0.05f + (0.0575f / 2.0f);
            float x     = design ? 0.0 : rand::uniform(-1.0f, 1.0f);
            float y     = design ? 0.0 : rand::uniform(-1.0f, 1.0f);
            float z     = design ? 0.3 : rand::uniform(min_z, rand::uniform(min_z, rand::uniform(min_z, 2.0f)));

            glm::vec3 cube_center = glm::vec3(x, y, z);
            glm::mat4 pos = glm::translate(glm::mat4(1.0f), cube_center);

            static float sr = 0;
            if (design)
                sr += 0.1;
            static bool set = false;

            /// 
            static int seed_val = 1;
            seed_val++;
            //image img = simplex_equirect_normal(seed_val, 1024, 1024, 15.0f, scales);

            //pipeline->textures[Asset::color].update(img);
            
            float rx = design ? 0.0 : rand::uniform(0.0, 180.0);
            model = glm::rotate(
                pos,
                glm::radians(rx),
                glm::vec3(1.0f, 0.0f, 0.0f)
            );
            float ry = design ? sr : rand::uniform(0.0, 180.0);
            model = glm::rotate(
                model,
                glm::radians(ry),
                glm::vec3(0.0f, 1.0f, 0.0f)
            );
            float rz = design ? sr / 8 : rand::uniform(0.0, 180.0);
            model = glm::rotate(
                model,
                glm::radians(rz),
                glm::vec3(0.0f, 0.0f, 1.0f)
            );
            view  = glm::lookAt(
                eye,
                glm::vec3(0.0f, 0.0f, 1.0f),
                glm::vec3(0.0f, 1.0f, 0.0f)
            );

            float cube_rads = 0.0015f; // double since we want the whole cube in scene (verify this)
            
            /// measuring the cubes appearance in series can give us a pretty good estimate of field of view.
            /// essentially moving it forward back and to the sides
            proj  = glm::perspective(
                glm::radians(70.0f), /// 70 seems avg, but i want to have a range to service if we can get this at runtime, configured or measured
                ext.width / (float) ext.height,
                0.05f, 10.0f); /// 5cm near, 10m far (this is clip only)
            proj[1][1] *= -1;

            glm::mat4 VP     = proj * view;
            glm::vec4 left   = glm::normalize(glm::row(VP, 3) + glm::row(VP, 0));
            glm::vec4 right  = glm::normalize(glm::row(VP, 3) - glm::row(VP, 0));
            glm::vec4 bottom = glm::normalize(glm::row(VP, 3) + glm::row(VP, 1));
            glm::vec4 top    = glm::normalize(glm::row(VP, 3) - glm::row(VP, 1));
            glm::vec4 vnear  = glm::normalize(glm::row(VP, 3) + glm::row(VP, 2));
            glm::vec4 vfar   = glm::normalize(glm::row(VP, 3) - glm::row(VP, 2));

            if (glm::dot(glm::vec3(left),   cube_center) + left.w   + cube_rads < 0) continue;
            if (glm::dot(glm::vec3(right),  cube_center) + right.w  + cube_rads < 0) continue;
            if (glm::dot(glm::vec3(bottom), cube_center) + bottom.w + cube_rads < 0) continue;
            if (glm::dot(glm::vec3(top),    cube_center) + top.w    + cube_rads < 0) continue;
            if (glm::dot(glm::vec3(vnear),  cube_center) + vnear.w  + cube_rads < 0) continue;
            if (glm::dot(glm::vec3(vfar),   cube_center) + vfar.w   + cube_rads < 0) continue;

            lights[0] = {
                glm::vec4(glm::vec3(2.0f, 0.0f, 4.0f), 25.0f),
                glm::vec4(1.0, 1.0, 1.0, 1.0)
            };
            lights[1] = {
                glm::vec4(glm::vec3(0.0f, 0.0f, -5.0f), 100.0f),
                glm::vec4(1.0, 1.0, 1.0, 1.0)
            };
            lights[2] = {
                glm::vec4(glm::vec3(0.0f, 0.0f, -5.0f), 100.0f),
                glm::vec4(1.0, 1.0, 1.0, 1.0)
            };
        } while (0);
    }
};



int main() {
    //float scales[3] = { 64, 64, 64 };
    //image img = simplex_equirect_normal(0, 1024, 512, 15.0f, scales);
    //img.save("simplex-normals.png");
    return Rubiks();
}
