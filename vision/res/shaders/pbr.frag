#version 450
#define PI 3.1415926535897932384626433832795

precision mediump int;
precision highp float;

struct Light {
    vec4 pos;
    vec4 color;
};

layout(binding = 0) uniform UniformBufferObject {
    mat4  model;
    mat4  view;
    mat4  proj;
    vec4  eye;
    Light lights[3];
} ubo;

layout(location = 0) in vec3 v_pos;
layout(location = 1) in vec3 v_world_pos; 
layout(location = 2) in vec2 v_uv;
layout(location = 3) in vec3 v_eye_dir;
layout(location = 4) in mat3 v_tbn;

layout(location = 0) out vec4 pixel;

layout(binding = 1) uniform sampler2D tx_color;
layout(binding = 2) uniform sampler2D tx_normal;
layout(binding = 3) uniform sampler2D tx_material;
layout(binding = 4) uniform sampler2D tx_reflect;
layout(binding = 5) uniform sampler2D tx_env;

vec3 _reflect(vec3 I, vec3 N) {
    return I - 2.0 * dot(N, I) * N;
}

void main() {
    vec4 color  = texture(tx_color,  v_uv).rgba;
    vec3 nmap   = texture(tx_normal, v_uv).rgb * 2.0 - 1.0;
    vec3 normal = normalize(v_tbn * nmap); /// perturb the normals

    /// reflection vector
    vec3 R = _reflect(-normalize(v_eye_dir), normal);

    /// convert reflection vector to spherical coordinates
    vec2 uv = vec2(atan(R.z, R.x), asin(R.y));
    uv.x   /= 2.0 * PI;         // Map from -PI to PI to 0 to 1
    uv.y   /= PI;               // Map from -PI/2 to PI/2 to 0 to 1
    uv      = uv * 0.5 + 0.5;   // Map from -1,1 to 0,1

    vec3 env_color = texture(tx_env, uv).rgb;
    
    pixel = vec4(v_uv.x, v_uv.y, 0.0, 1.0);
}
