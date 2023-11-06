#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 eye; // use .xyz (the extra float is needed for padding)
} ubo;

layout(location = 0) in  vec3 in_pos;
layout(location = 1) in  vec3 in_normal;
layout(location = 2) in  vec3 in_tangent;
layout(location = 3) in  vec3 in_bitangent;
layout(location = 4) in  vec2 in_uv;

layout(location = 0) out vec3 out_pos; 
layout(location = 1) out vec3 out_world_pos;
layout(location = 2) out vec2 out_uv;
layout(location = 3) out vec3 out_eye_dir;
layout(location = 4) out vec3 out_normal;
layout(location = 5) out mat3 out_tbn;

void main() {
    /// transpose inverted model view of normal when making bi/tan/normal
    mat3  m = mat3(ubo.model);
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(in_pos, 1.0);

    vec4 world_pos = ubo.model * vec4(in_pos, 1.0);

    out_pos     = world_pos.xyz;
    out_uv      = in_uv;
    out_eye_dir = normalize(world_pos.xyz - ubo.eye.xyz); // direction from eye to vertex
    out_normal  = m * in_normal;
    out_tbn     = mat3(m * in_tangent, m * in_bitangent, out_normal);
    out_world_pos = (m * in_tangent).xyz;
}