/// ux apps are both gfx and 3D context.  the 3D background can be reality or a game or a picture of both.
/// integration of 3D into Element

#include <ux/app.hpp>
#include <math/math.hpp>
#include <media/video.hpp>
#include <media/camera.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/random.hpp>

using namespace ion;

/// the head visor model is pretty basic to describe:
struct Head {
    float     width    =   0.15f; /// width  in meters -- ear canal to ear canal
    float     height   =   0.15f; /// height in meters
    float     depth    =   0.15f; /// depth  in meters -- can keep this locked to the width of the face
    float     eye_x    =   0.22f; /// distance scale between eyes, relative to width; should range from 0.35 to 0.62 (definitely!)
    float     eye_y    =  -0.10f; /// eye_y ratio, on orientation of head on the front plane; -11% height from median; should range from -30% to -5%
    float     eye_z    =   0.00f; /// z offset from frontal plane in units of face width
    float     eye_w    =   0.20f; /// width of eye segment from edge to edge
    float     ear_x    =   0.00f; /// ear_x offset from centroid of side (ear should be in the middle of the head on side plane)
    float     ear_y    =   0.00f; /// ear_y position (same relative ratio from head plane center on y; this is on the side, not front
    float     nose_y   =   0.00f; /// nose relative from median y of head; no sense of x offset here as if anything
    float     nose_z   =   0.15f; /// nose tip z position; in face width scale 
    glm::vec3 pos      = { 0.0f, 0.0f, 0.5f }; /// center of head in xyz cartesian coords
    glm::quat orient   = { 1.0f, 0.0f, 0.0f, 0.0f }; /// rotation stored in quaternion form
    map<mx>   tags;

    properties meta() {
        return {
            {"width",     width},
            {"height",    height},
            {"depth",     depth},
            {"eye_x",     eye_x},
            {"eye_y",     eye_y},
            {"eye_z",     eye_z},
            {"eye_w",     eye_w},
            {"ear_x",     ear_x},
            {"ear_y",     ear_y},
            {"nose_y",    nose_y},
            {"nose_z",    nose_z},
            {"pos",       pos},
            {"orient",    orient},
            {"tags",      tags}
        };
    }
    register(Head);
};


/// should be self contained and contain all methods for this feature
struct Cursor {
    /// the types we support; a Cursor type of none or !none may change the user interface.
    enums(Type, none, none, rubiks, cubidi);

    /// set centroid and scale on the coordinates, the active range we use;
    /// the model domain should be expanded by 20% when we infer this;
    /// then we can scale it 80% after

    /// it may be a bit useful to make the constraint a bit of a radial scale;
    /// on a cartesian coords as given by model

};

/// JFM
// feature Cursor;
// app     Cursor;

/// buttons inside here
struct Navigator:Element {
    
    enums(Nav, annotate,
        annotate, record, cursor_config); /// cursor selection (if any) 

    struct props {
        array<Nav> buttons;

        properties meta() {
            return {
                prop { "buttons", buttons  }
            };
        }

        type_register(props);
    };

    component(Navigator, Element, props);

    void on_select(event e) {
        printf("selected\n");
    }

    node update() {
        return node::each<Nav>(state->buttons, [&](Nav &button_type) -> node {
            symbol s_type = button_type.symbol();
            return Button {
                { "id",         s_type },
                { "behavior",   Button::Behavior::radio },
                { "on-select",  callback(this, &Navigator::on_select) }
            };
        });
    }
};

/// should stretch the fft when its less columns than the width of the control
/// love.
struct Spectrum {
    float frequencies[128]; /// 128 is likely the max amount of frequencies we would use
};

/// combined video seek with fft display of the audio, computed once, translated afterwards; 
/// it can also display thumbnails which it would generate once;
struct Seekbar:Element {
    struct props {
        bool display_video;
        bool display_audio;
        /// its a good idea to pan around as we zoom, so lets not store this here.
        /// model: displays array of Spectrum
        /// 
        properties meta() {
            return {
                { "display_video", display_video },
                { "display_audio", display_audio }
            };
        }

        type_register(props);
    };
    component(Seekbar, Element, props);

    void mounted() {
        if (state->live) {
            state->cam = camera(
                { StreamType::Audio, StreamType::Video, StreamType::Image },
                { Media::PCM, Media::PCMf32, Media::YUY2, Media::NV12, Media::MJPEG },
                "Logi", "PnP", 640, 360
            );
            state->cam.listen({ this, &VideoViewer::on_frame });
            state->video = Video(640, 360, 30, 48000, "test.mp4");
        } else {
            state->video = Video(ion::path("sample.mp4"));
            state->current_image = state->video.fetch_frame(40);
        }
    }

    void draw(Canvas &canvas) {
        Element::draw(canvas);
        rectd rect { 2, 2, 16, 16 };
        canvas.color(rgbad { 1.0, 1.0, 1.0, 1.0 });
        canvas.fill(rect);
    }
};

struct AudioTrack:mx {
    struct M {
        array<Spectrum> spectrum;
        i64 duration_millis;
    };
    mx_basic(AudioTrack);

};

struct VideoViewer:Element {
    struct props {
        float       angle;
        float       z_near, z_far;
        callback    clicked;
        vec2d       last_xy;
        bool        swirl;
        glm::vec2   sz;
        glm::mat4   model;
        glm::mat4   view;
        glm::mat4   proj;
        glm::vec3   start_cursor;
        glm::vec3   start_pos;
        glm::quat   start_orient;
        float       scroll_scale = 0.005f;
        image       current_image;
        bool        live = false;
        MStream     cam;
        Video       video;
        int         frame_id; /// needs to inter-operate with pts

        properties meta() {
            return {
                prop { "live",    live    },
                prop { "clicked", clicked }
            };
        }

        type_register(props);
    };

    component(VideoViewer, Element, props);

    void mounted() {
        if (state->live) {
            state->cam = camera(
                { StreamType::Audio,
                  StreamType::Video,
                  StreamType::Image }, /// ::Image resolves the Image from the encoded Video data
                { Media::PCM, Media::PCMf32, Media::YUY2, Media::NV12, Media::MJPEG },
                "Logi", "PnP", 640, 360
            );
            state->cam.listen({ this, &VideoViewer::on_frame });
            state->video = Video(640, 360, 30, 48000, "test.mp4");
        } else {
            state->video = Video(ion::path("sample.mp4"));
            state->current_image = state->video.fetch_frame(40);
        }
    }

    void on_frame(Frame &frame) {
        state->current_image = frame.image;
        if (state->video) {
            if (state->frame_id < 30 * 10) {
                state->frame_id++;
                state->video.write_frame(frame);
                if (state->frame_id == 30 * 10) {
                    state->video.stop();
                    state->cam.cancel();
                }
            }
        }
    }

    void down() {
        Head *head = context<Head>("head");
        state->last_xy        = Element::data->cursor;
        state->start_cursor   = glm::vec3(Element::data->cursor.x, Element::data->cursor.y, 0.0);
        state->start_pos      = head->pos;
        state->start_orient   = head->orient;

        // Convert to NDC
        glm::vec2 ndc;
        ndc.x =        (2.0f * state->last_xy.x) / state->sz.x - 1.0f;
        ndc.y = 1.0f - (2.0f * state->last_xy.y) / state->sz.y;

        glm::vec4 rayClip = glm::vec4(ndc.x, ndc.y, -1.0f, 1.0f);
        glm::vec4 rayEye  = glm::inverse(state->proj) * rayClip;
        rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);

        glm::vec3 rayWor = glm::normalize(glm::vec3(glm::inverse(state->view) * rayEye));

        double dist = glm::distance(
            glm::vec3(rayWor.x, rayWor.y, 0.0),
            glm::vec3(head->pos.x, head->pos.y, 0.0)
        );
        
        // A simple way to check if the click is outside the cube
        state->swirl = dist > head->width * 1.0;
    }

    glm::vec3 forward() {
        glm::mat4 &v = state->view;
        return -glm::normalize(glm::vec3(v[0][2], v[1][2], v[2][2]));
    }

    glm::vec3 to_world(float x, float y, float reference_z, const glm::mat4 &viewMatrix, const glm::mat4 &projectionMatrix, float screenWidth, float screenHeight) {
        // Convert to normalized device coordinates
        float xNDC = (2.0f * x) / screenWidth - 1.0f;
        float yNDC = 1.0f - (2.0f * y) / screenHeight;
        float zNDC = 2.0f * reference_z - 1.0f; // Convert the reference_z to NDC

        glm::vec4 clipSpacePos = glm::vec4(xNDC, yNDC, zNDC, 1.0f);

        // Convert from clip space to eye space
        glm::vec4 eyeSpacePos = glm::inverse(projectionMatrix) * clipSpacePos;

        // Convert from eye space to world space
        glm::vec4 worldSpacePos = glm::inverse(viewMatrix) * eyeSpacePos;

        return glm::vec3(worldSpacePos) / worldSpacePos.w;
    }

    void scroll(real x, real y) {
        Head *head = context<Head>("head");
        head->pos.z += y * state->scroll_scale;
    }

    /// mouse move event
    void move() {
        Head *head = context<Head>("head");
        if (!Element::data->active)
            return;
        
        vec2d diff = Element::data->cursor - state->last_xy;
        state->last_xy = Element::data->cursor;

        const float sensitivity = 0.2f; // Sensitivity factor

        // Convert pixel difference to angles (in radians)
        float ax = glm::radians(diff.y * sensitivity); // Vertical   movement for X-axis rotation
        float ay = glm::radians(diff.x * sensitivity); // Horizontal movement for Y-axis rotation

        auto cd = node::data;
        glm::vec3 drag_pos = glm::vec3(Element::data->cursor.x, Element::data->cursor.y, 0.0f);
        glm::vec3 drag_vec = state->start_cursor - drag_pos;
        drag_vec.y = -drag_vec.y;

        if (cd->composer->shift) {
            float z  = head->pos.z;
            float zv = 1.0f - (head->pos.z - state->z_near) / (state->z_far - state->z_near);

            glm::vec3 cursor    = glm::vec3(Element::data->cursor.x, Element::data->cursor.y, 0.0f);
            glm::vec3 p0        = to_world(state->start_cursor.x, state->start_cursor.y, zv, state->view, state->proj, state->sz.x, state->sz.y);
            glm::vec3 p1        = to_world(cursor.x, cursor.y, zv, state->view, state->proj, state->sz.x, state->sz.y);
            
            glm::vec3 pd = p1 - p0;
            printf("pd = %.2f %.2f %.2f\n", pd.x, pd.y, pd.z);
            printf("cursor = %.2f %.2f %.2f\n", cursor.x, cursor.y, cursor.z);

            head->pos     = state->start_pos + (p1 - p0);
            head->pos.z   = z;
        } else {
            if (state->swirl) {
                head->orient = head->orient * glm::angleAxis(-ax, glm::vec3(0.0f, 0.0f, 1.0f));
            } else {
                // Calculate the rotation axis and angle from the mouse drag
                glm::vec3 view_dir = forward();
                glm::vec3 r_axis   = glm::normalize(glm::cross(drag_vec, view_dir));
                float     r_amount = glm::length(drag_vec) / 100.0f; // Adjust sensitivity
                head->orient = state->start_orient * glm::angleAxis(r_amount, r_axis);
            }
        }
    }

    void draw(Canvas& canvas) {
        /// the base method calculates all of the rectangular regions; its done in draw because canvas context is needed for measurement
        Element::draw(canvas);

        Head *head = context<Head>("head");
        float w = head->width  / 2.0f;
        float h = head->height / 2.0f;
        float d = head->depth  / 2.0f;

        // test code:
        //glm::quat additional_rotation = glm::angleAxis(radians(1.0f) / 10.0f, glm::vec3(0.0f, 1.0f, 0.0f));
        //head->orient = head->orient * additional_rotation;

        array<glm::vec3> face_box = {
            glm::vec3(-w, -h, -d), glm::vec3( w, -h, -d), // EF
            glm::vec3( w, -h, -d), glm::vec3( w,  h, -d), // FG
            glm::vec3( w,  h, -d), glm::vec3(-w,  h, -d), // GH
            glm::vec3(-w,  h, -d), glm::vec3(-w, -h, -d)  // HE
        };
 
        glm::vec3 eye = glm::vec3(0.0f, 0.0f, 0.0f);
        
        //image img = path { "textures/rubiks.color2.png" };
        //pipeline->textures[Asset::color].update(img); /// updating in here is possible because the next call is to check for updates to descriptor

        state->z_near = 0.0575f / 2.0f * sin(radians(45.0f));
        state->z_far  = 10.0f;

        double cw = Element::data->bounds.w;
        double ch = Element::data->bounds.h;
        glm::vec2 sz    = { cw, ch };
        glm::mat4 proj  = glm::perspective(glm::radians(70.0f), sz.x / sz.y, state->z_near, state->z_far);
        proj[1][1] *= -1;

        state->sz = sz;

        glm::mat4 view  = glm::lookAt(eye, glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 model = glm::translate(glm::mat4(1.0f), head->pos) * glm::toMat4(head->orient); // glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0.0f, 1.0f, 0.0f));

        static rgbad white = { 1.0, 1.0, 1.0, 1.0 };
        static rgbad red   = { 1.0, 0.0, 0.0, 1.0 };
        static rgbad green = { 0.0, 1.0, 0.0, 1.0 };
        static rgbad blue  = { 0.0, 0.0, 1.0, 1.0 };

        canvas.save();

        /// draw webcam!
        rectd     bounds { 0.0, 0.0, sz.x, sz.y };
        vec2d     offset { 0.0, 0.0 };
        alignment align  { 0.5, 0.5 };

        canvas.color(Element::data->drawings[operation::fill].color);
        canvas.fill(bounds);

        if (state->current_image) {
            canvas.image(state->current_image, bounds, align, offset);  
        }

        canvas.projection(model, view, proj);
        canvas.outline_sz(2);
        for (size_t i = 0; i < 4; i++)
            canvas.line(face_box[i * 2 + 0], face_box[i * 2 + 1]);
        
        state->model = model;
        state->view  = view;
        state->proj  = proj;
       
        /// draw eyes
        float fw    = head->width;
        float fh    = head->height;
        float eye_w = fw * head->eye_w;
        float eye_x = fw * head->eye_x;
        float eye_y = fw * head->eye_y;
        float eye_z = fw * head->eye_z; /// frontal plane is the eye plane as annotated; useful to have a z offset
        
        float nose_x = 0.0f;
        float nose_y = fh * head->nose_y;
        float nose_z = fw * head->nose_z;
        float nose_h = fh * 0.02f;

        float ear_x  = fw * head->ear_x;
        float ear_y  = fh * head->ear_y;
        float ear_h  = fh * 0.02f; /// should be a circle or a square, not a line segment
        canvas.outline_sz(1);

        /// we want to replace this with a silohette on 2 axis
        /// thats far easier to scale and line up
        /// its not a chore to manage these points with a profile view.
        /// its literally a profile that we measure with the model, associated to the subject in annotations
        /// the idea of making planes is not good
        /// top part: middle of forehead (not visible with hair, but measurable by human; basically lower middle of ballcap or something)
        /// bottom: top of the upper lip is probably good
        /// ability to copy and paste profiles is a good feature, from file to file

        array<glm::vec3> features = {
            glm::vec3(-eye_x - eye_w / 2, eye_y, -d + eye_z),
            glm::vec3(-eye_x + eye_w / 2, eye_y, -d + eye_z),

            glm::vec3( eye_x - eye_w / 2, eye_y, -d + eye_z),
            glm::vec3( eye_x + eye_w / 2, eye_y, -d + eye_z),

            glm::vec3( nose_x, nose_y,          -d - nose_z),
            glm::vec3( nose_x, nose_y + nose_h, -d - nose_z),

            glm::vec3(-w, ear_y,         d * ear_x),
            glm::vec3(-w, ear_y + ear_h, d * ear_x), /// just for noticable length

            glm::vec3(+w, ear_y,         d * ear_x),
            glm::vec3(+w, ear_y + ear_h, d * ear_x) /// just for noticable length
        };

        for (size_t i = 0; i < 10; i += 2)
            canvas.line(features[i + 0], features[i + 1]);
        
        canvas.restore();
    }
};

/// its a button and it controls the main menu ops
struct MainMenu:Element {
    struct props {
        bool sample;
        type_register(props);
    };

    component(MainMenu, Element, props);

    void on_click(event e) {
        printf("main menu\n");
    }

    node update() {
        return Button {
            { "id",         "main-menu" },
            { "on-click",    callback(this, &MainMenu::on_click) }
        };
    }
};

/// this should perform operation with alterations to tags, and subsequent style change
/// need syntax for remaining units in coord, and it can be a % of that; i suppose % can always do this?
/// insight: this is probably preferred since it can be reduced to be the same function it was with some heuristics
/// 

/// will be controlled in css; it holds onto 
struct Page:Element {
    struct props {
        bool sample;
    };
    component(Page, Element, props);
};

struct Ribbon:Element {
    struct props {
        map<Element> content; // headers need only an 'id'-header, their selected/unselected state tag, content would have 'id'-content, selected/unselected state
        str          selected; // we set this, its not exposed
        type_register(props);
    };

    component(Ribbon, Element, props);

    /// Elements can be called because we can grab Elements along with type-driven context properties
    void select(str id) {
        state->selected = id;
    }

    void on_click(event e) {
        printf("on_click method called\n");
    }

    node update() {
        return node::each<str, Element>(state->content, [&](str &id, Element &e) -> node {
            str  header_id = fmt { "{0}-header",  id };
            str content_id = fmt { "{0}-content", id };
            str       tags = id == state->selected ? array<str> { "selected" } : {};
            return array<node> {
                Button {
                    { "id",         header_id }, /// css can do the rest
                    { "behavior",   Button::Behavior::radio },
                    { "on-change", []() {
                        // call update
                    }}
                },
                Page {
                    header_id.symbolize(), tags, array<node> { e }
                }
            }
        });

        return Button {
            { "id", "main-menu" },
            { "on-click", callback(this, &Ribbon::on_click) }
        };
    }
};

struct Annotate:Element {
    struct props {
        Head head;
        properties meta() {
            return {
                prop { "head", head }
            };
        }
        type_register(props);
    };
    
    component(Annotate, Element, props);

    node update() {
        Head *head = &state->head;
        return array<node> {
            MainMenu {
                { "id", "main-menu" }
            },
            Navigator {
                { "id", "navigator" },
                { "buttons", array<Navigator::Nav> {
                    Navigator::Nav("annotate"),
                    Navigator::Nav("record"),
                    Navigator::Nav("cursor-config") } }
            },
            VideoViewer {
                { "id", "video-viewer" }
            },
            Seekbar {
                {"id", "seekbar" }
            }
        };
    }
};

int main(int argc, char *argv[]) {
    map<mx> defs  {{ "debug", uri { null }}};
    map<mx> config { args::parse(argc, argv, defs) };
    if    (!config) return args::defaults(defs);
    ///
    return App(config, [](App &app) -> node {
        return Annotate {
            { "id", "main" }
        };
    });
}

/// visualize audio track
/// seek position thumb