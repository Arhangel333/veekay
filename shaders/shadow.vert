// shadow.vert - ПРАВИЛЬНО
#version 450

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 light_view_projection;
} pc;

layout(location = 0) in vec3 inPosition;

void main() {
    gl_Position = pc.light_view_projection * pc.model * vec4(inPosition, 1.0);
}