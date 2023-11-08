#include <vk/vk.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/random.hpp>
//#include <glm/gtx/random.hpp>

using namespace ion;

struct Light {
    alignas(16) glm::vec4 pos;
    alignas(16) glm::vec4 color;
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 tangent;
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
        normal = {
            v_normal[3 * n_index + 0],
            v_normal[3 * n_index + 1],
            v_normal[3 * n_index + 2]
        };
    }

    doubly<prop> meta() const {
        return {
            prop { "POSITION",      pos },
            prop { "NORMAL",        normal },
            prop { "TANGENT",       tangent },
            prop { "TEXCOORD_0",    uv }
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

struct Labels:mx {
    /// data protected by NAN
    struct M {
        float   x = NAN,  y = NAN,  z = NAN;
        float  qx = NAN, qy = NAN, qz = NAN, qw = NAN;
        float fov = NAN;

        doubly<prop> meta() {
            return {
                { "x",     x },
                { "y",     y },
                { "z",     z },
                { "qx",   qx },
                { "qy",   qy },
                { "qz",   qz },
                { "qw",   qw },
                { "fov", fov }
            };
        }
        /// NAN helps to keep it real with a bool operator
        operator bool() {
            return !std::isnan(x)  && !std::isnan(y)  && !std::isnan(z)  &&
                   !std::isnan(qx) && !std::isnan(qy) && !std::isnan(qz) && !std::isnan(qw);
        }
        register(M);
    };

    mx_basic(Labels);
    Labels(null_t):Labels() { }

    operator bool() {
        return *data;
    }
};

struct Rubiks:mx {
    struct M {
        Vulkan          vk { 1, 0 };        /// this lazy loads 1.0 when GPU performs that action [singleton data]
        vec2i           sz { 256, 256 };    /// store current window size
        Window          gpu;                /// GPU class, responsible for holding onto GPU, Surface and GLFWwindow
        Device          device;             /// Device created with GPU
        Pipes           pipes;              /// pipeline for single object scene
        bool            design = true;      /// design mode
        Labels          labels = null;
        path            output_dir { "gen" };

        static void resized(vec2i &sz, M* app) {
            app->sz = sz;
            app->device->framebufferResized = true;
        }

        void init() {
            gpu      = Window::select(sz, ResizeFn(resized), this);
            device   = Device::create(gpu);
            pipes    = Pipes(
                device, "rubiks", array<Graphics> {
                    Graphics { "rubiks", typeof(UniformBufferObject), typeof(Vertex) }
                }
            );
        }

        void run() {
            str odir = output_dir.cs();
            output_dir.make_dir();
            array<Pipes> a_pipes({ pipes });

            while (!glfwWindowShouldClose(gpu->window)) {
                glfwPollEvents();
                device->mtx.lock();
                device->drawFrame(a_pipes);
                vkDeviceWaitIdle(device);
                
                if (labels) {
                    image img      = device->screenshot();
                    assert(img);
                    
                    str  base      = fmt { "rubiks-{0}",   { str::rand(12, 'a', 'z') }};
                    path rel_png   = fmt { "{0}.png",      { base }};
                    path path_png  = fmt { "{1}/{0}.png",  { base, odir }};
                    path path_json = fmt { "{1}/{0}.json", { base, odir }};

                    if (path_png.exists() || path_json.exists())
                        continue;
                    
                    var     annots = map<mx> {
                        { "labels", labels  },
                        { "source", rel_png }
                    };
                    assert(path_json.write(annots));
                    assert(img.save(path_png));
                    labels = null;
                }
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
            for (Pipeline &pipeline: data->pipes->pipelines)
                pipeline->user = mem->grab();
            data->run();
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
};

glm::quat quaternion_rotate(glm::vec3 v, float rads) {
    return glm::angleAxis(rads, glm::normalize(v));
}

glm::quat rand_quaternion() {
    glm::vec3 rv(
        glm::linearRand(-1.0f, 1.0f),
        glm::linearRand(-1.0f, 1.0f),
        glm::linearRand(-1.0f, 1.0f)
    );
    float angle = glm::linearRand(0.0f, glm::pi<float>());
    return glm::angleAxis(angle, glm::normalize(rv));
}

/// uniform has an update method with a pipeline arg
struct UniformBufferObject {
    alignas(16) glm::mat4  model;
    alignas(16) glm::mat4  view;
    alignas(16) glm::mat4  proj;
    alignas(16) glm::vec4  eye;
    alignas(16) Light      lights[3];

    void process(Pipeline pipeline) { /// memory* -> Pipeline conversion implicit from the function in static
        VkExtent2D &ext = pipeline->device->swapChainExtent;
        Rubiks   rubiks = pipeline->user.grab();
        bool     design = rubiks->design;

        eye = glm::vec4(glm::vec3(0.0f, 0.0f, 0.0f), 0.0f); /// these must be padded in general
        
        //image img = path { "textures/rubiks.color2.png" };
        //pipeline->textures[Asset::color].update(img); /// updating in here is possible because the next call is to check for updates to descriptor

        static bool did_perlin = false;
        if (!did_perlin) {
            float scales[3] = { 256, 256, 256 };
            image img = simplex_equirect_normal(1, 1024, 512, 15.0f, scales);
            img.save("noise_map.png");
            did_perlin = true;
        }

        float z_clip = 0.0575f / 2.0f * sin(radians(45.0f));
        float z_far  = 10.0f;
        float z_max_train = 1.0f;
        proj = glm::perspective(
            glm::radians(70.0f),
            ext.width / (float) ext.height,
            z_clip, z_far); /// radius of 45 degrees * object size / 2 = near, 10m far (clip only)
        proj[1][1] *= -1;

        //pipeline->textures[Asset::color].update(img)
        view = glm::lookAt(
            glm::vec3(eye),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec3(0.0f, 1.0f, 0.0f));
        
        float px = design ? 0.00f           : rand::uniform( 0.0f, 0.0f);
        float py = design ? 0.00f           : rand::uniform( 0.0f, 0.0f);
        float pz = design ? (z_clip * 4.0f) : rand::uniform(z_clip / z_far * 2.0f, z_max_train / z_far); /// train to 1 meters out.  thats a very far away cube from the camera

        /// this should place the object as close to the camera as possible without a clip even as it rotates to 45
        /// not sure why we are needing to double this zclip, as it should be zclip*2 intuitively.
        /// perspective does not seem to work as i thought; the labeling should offset by this amount.
        glm::vec3 cube_center = glm::vec3(px, py, pz);
        glm::mat4 position    = glm::translate(glm::mat4(1.0f), cube_center);

        if (design) {
            static float r = 0.0f;
            static const float rads = M_PI * 2.0f;
            glm::vec3 v  = glm::vec3(0.0f, 1.0f, 0.0f);
            glm::quat qt = quaternion_rotate(v, r * rads);
            r += rads * 0.00001;
            if (r > rads)
                r -= rads;
            model = position * glm::toMat4(qt);
        } else {
            /// vary model rotation
            glm::quat qt = rand_quaternion();
            model = position * glm::toMat4(qt);

            /// set all fields in Labels
            rubiks->labels = Labels::M {
                .x   = cube_center.x,
                .y   = cube_center.y,
                .z   = cube_center.z,
                .qx  = qt.x,
                .qy  = qt.y,
                .qz  = qt.z,
                .qw  = qt.w,
                .fov = 70.0f / 90.0f // normalize by 90
            };
        }

        float cube_rads = 0.0575f * 5;
        glm::mat4 VP     = proj * view;
        glm::vec4 left   = glm::normalize(glm::row(VP, 3) + glm::row(VP, 0));
        glm::vec4 right  = glm::normalize(glm::row(VP, 3) - glm::row(VP, 0));
        glm::vec4 bottom = glm::normalize(glm::row(VP, 3) + glm::row(VP, 1));
        glm::vec4 top    = glm::normalize(glm::row(VP, 3) - glm::row(VP, 1));
        glm::vec4 vnear  = glm::normalize(glm::row(VP, 3) + glm::row(VP, 2));
        glm::vec4 vfar   = glm::normalize(glm::row(VP, 3) - glm::row(VP, 2));

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

int main() {
    return Rubiks();
}
