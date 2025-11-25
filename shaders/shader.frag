#version 450

struct Material {
    vec3 albedo;
    vec3 specular;  
    float shininess;
};

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

layout(binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 view_position;
    uint point_light_count;
};

layout(binding = 1) uniform ModelUniforms {
    mat4 model;
    mat4 normal_matrix;
    Material material;
};

layout(location = 0) out vec4 outColor;

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 lightDir = normalize(vec3(0.0, -1.0, 0.0));
    float diff = max(dot(normal, lightDir), 0.0);

    vec3 ambient = fragColor * 0.2;
    vec3 diffuse = fragColor * diff;
    
    vec3 viewDir = normalize(view_position - fragPosition);  // Фиксированная камера
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    vec3 specular = material.specular * spec;  
    
    

    // 👇 DOT PRODUCT: насколько камера видит отражённый свет
    float dot_product = dot(viewDir, reflectDir);
    
    vec3 face_color = normal * 0.5 + 0.5;  // Цвет по нормалям
    vec3 dot_color = vec3(dot_product);     // Белые пятна
    

    


    outColor = vec4(ambient + diffuse + specular, 1.0);  // 👈 Всё ещё БЕЗ specular
}