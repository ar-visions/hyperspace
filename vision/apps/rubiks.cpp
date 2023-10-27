#include <vk/vk.hpp>

using namespace ion;

/// a useful tooling experiment to create an annotation cursor out of a rubiks cube model; location and orientation.
/// curious to know how accurate one could get the data from a single shot of vit.
/// we would have composed accuracy on the labeling -- to me its exciting when you have that.
/// use-case is value into any annotation in space, position and rotation in animation or singular plots.

/// the wonderful thing about the rubiks is how easy it is to render different styles and shadings
/// a simple pbr shader would be the ultimate way to do it, but a gloss map is probably fine with 
/// not so sophisticated shading.  thats most of what you would be doing in pbr.
/// probably more in pbr would be different env maps mipmapped for the various diffuse levels of lighting that pbr uses.
/// i am interested in more gaussian distribution if its at all performant.

/// anyway it probably wouldnt even make a huge difference on a simpleton approach, 
/// but it would be the best way to do it for this domain of model.
/// simple is just some OR and ANDs of blend src at varying scale, applied w perlin, 
/// and perlin masked blurring or both the blend material and the mix
/// i guess thats simple enough.  its possible that might even beat 3D but i dont 
/// really want to hack around with that stuff.  just make the scene, and in a
/// general way you can make others.. thats easier not harder

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
            rand::uniform(0.0f, 1.0f),
            rand::uniform(0.0f, 1.0f)
        }; //{     v_uv[2 * uv_index + 0], 1.0f - v_uv[2 * uv_index + 1] };
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

/// uniform has an update method with a pipeline arg
struct UniformBufferObject {
    alignas(16) glm::mat4  model;
    alignas(16) glm::mat4  view;
    alignas(16) glm::mat4  proj;
    alignas(16) glm::vec3  eye;
    alignas(16) Light      lights[3];

    void update(Pipeline::impl *pipeline) {
        VkExtent2D &ext = pipeline->device->swapChainExtent;

        static auto startTime   = std::chrono::high_resolution_clock::now();
        auto        currentTime = std::chrono::high_resolution_clock::now();
        float       time        = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
        
        eye   = glm::vec3(0.0f, 4.0f, 6.0f);
        model = glm::rotate(
            glm::mat4(1.0f),
            time * glm::radians(90.0f) * 0.5f,
            glm::vec3(0.0f, 0.0f, 1.0f)
        );
        model = glm::scale(model, glm::vec3(10.0));
        view  = glm::lookAt(
            eye,
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 1.0f)
        );
        proj  = glm::perspective(
            glm::radians(70.0f), /// 70 seems avg, but i want to have a range to service if we can get this at runtime, configured or measured
            ext.width / (float) ext.height,
            0.05f, 10.0f); /// 5cm near, 10cm far
            /// we should land rockets using CV -- why not build a game to land with the players 
            /// impulse recordings in horrifying conditions.. use for training.  what games should be: recordings for ai.
            /// your recordings become selectable
        proj[1][1] *= -1;

        /// setup some scene lights
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
    }
};

struct Rubiks:mx {
    struct M {
        Vulkan          vk { 1, 0 };        /// this lazy loads 1.0 when GPU performs that action [singleton data]
        vec2i           sz { 1920, 1080 };  /// store current window size
        Window          gpu;                /// GPU class, responsible for holding onto GPU, Surface and GLFWwindow
        Device          device;             /// Device created with GPU
        Pipeline        pipeline;           /// pipeline for single object scene

        static void resized(vec2i &sz, M* app) {
            app->sz = sz;
            app->device->framebufferResized = true;
        }

        void init() {
            printf("init called (war were declared)\n");
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
            data->run();
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
};

int main() {
    return Rubiks();
}
