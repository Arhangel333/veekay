#version 450

// 👇 ДОБАВИМ СТРУКТУРУ MATERIAL
struct Material {
    vec3 albedo;
    vec3 specular;  
    float shininess;
};

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;

layout(binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 view_position;     // 👈 ДОБАВИМ ПОЗИЦИЮ КАМЕРЫ!
	float _pad0;
    uint point_light_count;
	uint spot_light_count;
	float _pad1[2];
};

layout(binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    mat4 normal_matrix; 
    Material material;
};

// 👇 ПЕРЕДАЁМ БОЛЬШЕ ДАННЫХ ДЛЯ ОСВЕЩЕНИЯ
layout(location = 0) out vec3 fragPosition;    // Позиция в мировых координатах
layout(location = 1) out vec3 fragNormal;      // Нормаль в мировых координатах  
layout(location = 2) out vec3 fragColor;       // Цвет материала

void main() {
    // 👇 ПРАВИЛЬНОЕ ПРЕОБРАЗОВАНИЕ ПОЗИЦИИ
    vec4 worldPosition = model * vec4(v_position, 1.0);
     gl_Position = view_projection * model * vec4(v_position, 1.0);

    // 👇 ПРАВИЛЬНОЕ ПРЕОБРАЗОВАНИЕ НОРМАЛЕЙ
    fragNormal = mat3(normal_matrix) * v_normal;
    
    // Передаём данные для освещения
    fragPosition = worldPosition.xyz;
    fragColor = material.albedo;
}