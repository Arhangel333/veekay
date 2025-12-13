#version 460

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    mat4 light_view_projection;
    vec3 view_position;
    float _pad0;
    uint point_light_count;
    uint spot_light_count;
    uint directional_light_count;
    float _pad1[1];
    vec3 ambientColor;
    float ambientIntensity;
} scene;

struct Material {
    vec3 albedo;
    vec3 specular;  
    float shininess;
};

layout(set = 0, binding = 1) uniform ModelUniforms {
    mat4 model;
    mat4 normal_matrix;
    Material material;
};

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragUV;
layout(location = 3) out vec4 fragPosLightSpace;

void main() {
    vec4 worldPos = model * vec4(inPosition, 1.0);
    fragPosition = worldPos.xyz;
    fragNormal = mat3(normal_matrix) * inNormal;
    fragUV = inUV;

    fragPosLightSpace = vec4(worldPos.xyz, 1.0) * scene.light_view_projection;
    
    gl_Position = scene.view_projection * worldPos;
    
}