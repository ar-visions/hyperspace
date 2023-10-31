#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 eye; // use .xyz (the extra float is needed for padding)
} ubo;

layout(location = 0) in  vec3 in_pos;
layout(location = 1) in  vec3 in_normal;
layout(location = 2) in  vec2 in_uv;

layout(location = 0) out vec3 out_pos; 
layout(location = 1) out vec2 out_uv;
layout(location = 2) out vec3 out_eye_dir;
layout(location = 3) out mat3 out_tbn;

void main() {
    /// transpose inverted model view of normal when making bi/tan/normal
    vec3 iN = mat3(transpose(inverse(ubo.model))) * in_normal;
    vec3 iU = mat3(transpose(ubo.model)) * vec3(0.0, 0.0, 1.0);
    vec3  T = cross(iN, iU);
    vec3  B = cross(T, iN);
    
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(in_pos, 1.0);

    vec4 world_pos = ubo.model * vec4(in_pos, 1.0);

    out_pos     = world_pos.xyz;
    out_uv      = in_uv;
    out_eye_dir = (world_pos - ubo.eye).xyz; // direction from eye to vertex
    out_tbn     = mat3(T, B, iN);
}