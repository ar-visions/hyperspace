#version 450

struct Light {
    vec4 pos;
    vec4 color;
};

layout(binding = 0) uniform UniformBufferObject {
    mat4  model;
    mat4  view;
    mat4  proj;
    vec3  eye;
    Light lights[3];
} ubo;

layout(location = 0) in vec3 v_pos; /// v_pos here is out_pos in .vert
layout(location = 1) in vec2 v_uv;
layout(location = 2) in mat3 v_tbn;

layout(location = 0) out vec4 pixel;

layout(binding = 1) uniform sampler2D tx_color;
layout(binding = 2) uniform sampler2D tx_normal;
layout(binding = 3) uniform sampler2D tx_material;
layout(binding = 4) uniform sampler2D tx_reflect;

void main() {
    vec4 color = texture(tx_color, v_uv).rgba;
    pixel      = vec4(vec3(1.0, 0.0, 1.0), 1.0);
}