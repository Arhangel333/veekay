#version 450

struct Material {
    vec3 albedo;
    vec3 specular;  
    float shininess;
};

// Входные данные от вершинного шейдера
layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    // 👇 ОСНОВЫ ОСВЕЩЕНИЯ
    
    // 1. Нормализуем нормаль (ОЧЕНЬ ВАЖНО!)
    vec3 normal = normalize(fragNormal);
    
    // 2. Направление света (сверху-справа)
    vec3 lightDir = normalize(vec3(-1.0, 1.0, 0.5));
    
    // 3. Диффузная составляющая (основной свет)
    float diff = max(dot(normal, lightDir), 0.0);
    
    // 4. Фоновое освещение (чтобы не было совсем темно)
    float ambient = 1.5;
    
    // 5. Итоговый свет
    float light = ambient + diff;
    
    // 👇 РИСУЕМ С УЧЁТОМ ОСВЕЩЕНИЯ
    outColor = vec4(fragColor * light, 1.0);
}