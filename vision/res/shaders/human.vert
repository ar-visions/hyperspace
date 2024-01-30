#version 450

#define MAX_JOINTS 512
#define MAX_LIGHTS 3

struct Light {
    vec4 pos;
    vec4 color;
};

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 eye; // use .xyz (the extra float is needed for padding)
    mat4 joint_mats[MAX_JOINTS];
    vec4 lights[MAX_LIGHTS];
} ubo;

layout(location = 0) in  vec3  a_pos;
layout(location = 1) in  vec3  a_normal;
layout(location = 2) in  vec2  a_uv0;
layout(location = 3) in  vec2  a_uv1;
layout(location = 4) in  vec4  a_tangent;
layout(location = 5) in  ivec4 a_joints;
layout(location = 6) in  vec4  a_weights;

layout(location = 0) out vec3  v_pos; 
layout(location = 1) out vec3  v_world_pos;
layout(location = 2) out vec2  v_uv0;
layout(location = 3) out vec2  v_uv1;
layout(location = 4) out vec3  v_eye_dir;
layout(location = 5) out mat3  v_tbn;

void main() {
    mat3 m         = mat3(ubo.model);
    mat4 skin      = mat4(0.0);
    for (int i = 0; i < 4; i++)
        skin += ubo.joint_mats[a_joints[i]] * a_weights[i];

    vec4 world_pos = ubo.model * skin * vec4(a_pos, 1.0);
    mat3 nm        = transpose(inverse(mat3(ubo.model))); // transpose inverted model view of normal when making bi/tan/normal m3
    vec3 tangent   = normalize(nm * a_tangent.xyz);
    vec3 bitangent = normalize(nm * (cross(a_normal, a_tangent.xyz) * a_tangent.w));
    vec3 normal    = normalize(nm * a_normal);
    v_pos          = world_pos.xyz;
    v_world_pos    = world_pos.xyz;
    v_uv0          = vec2(a_uv0.x, a_uv0.y);
    v_uv1          = vec2(a_uv1.x, a_uv1.y);
    v_eye_dir      = normalize(world_pos.xyz - ubo.eye.xyz); // direction from eye to vertex
    v_tbn          = mat3(tangent, bitangent, normal);
    gl_Position    = ubo.proj * ubo.view * world_pos;
}