#version 460

struct PointLight {
    vec3 position;
    float _pad0;
    vec3 color;
    float _pad1;
    float intensity;
    float constant;
    float linear;
    float quadratic;
};

struct SpotLight {
    vec3 position;
    float _pad0;
    vec3 direction;
    float _pad1;
    vec3 color;
    float _pad2;
    float intensity;
    float cutOff;
    float outerCutOff;
    float constant;
    float linear;
    float quadratic;
    float _pad3[2];
};

struct DirectionalLight {
    vec3 direction;
    float _pad0;
    vec3 color;
    float intensity;
};

struct Material {
    vec3 albedo;
    vec3 specular;  
    float shininess;
};

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;
layout(location = 3) in vec4 fragPosLightSpace;

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

layout(set = 0, binding = 1) uniform ModelUniforms {
    mat4 model;
    mat4 normal_matrix;
    Material material;
};

layout(set = 0, binding = 2) uniform sampler2D texture_sampler;
layout(set = 0, binding = 3) uniform sampler2DShadow shadow_sampler;

layout(set = 1, binding = 0) readonly buffer PointLightsSSBO {
    PointLight point_lights[];
};

layout(set = 1, binding = 1) readonly buffer SpotLightsSSBO {
    SpotLight spot_lights[];
};

layout(set = 1, binding = 2) readonly buffer DirectionalLightsSSBO {
    DirectionalLight directional_lights[];
};

layout(location = 0) out vec4 outColor;

// ФУНКЦИЯ ДОЛЖНА БЫТЬ ОБЪЯВЛЕНА ПЕРЕД ИСПОЛЬЗОВАНИЕМ!
float calculateShadow(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir) {
    if (scene.directional_light_count == 0) {
        return 1.0;
    }
    
    // Перспективное деление
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    
    // Преобразуем из [-1, 1] в [0, 1]
    projCoords = projCoords * 0.5 + 0.5;
    
    // Проверяем границы
    if (projCoords.z > 1.0 || projCoords.x < 0.0 || projCoords.x > 1.0 || 
        projCoords.y < 0.0 || projCoords.y > 1.0) {
        return 1.0;
    }
    
    // Depth bias
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    
    // Читаем shadow map
    float shadow = texture(shadow_sampler, vec3(projCoords.xy, projCoords.z - bias));
    
    return shadow;
}

vec3 calculatePointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec4 texColor) {
    vec3 lightDir = normalize(light.position - fragPos);
    
    // Диффузная составляющая
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = light.color * diff * light.intensity * material.albedo * texColor.rgb;
    
    // Specular (Блинн-Фонг)
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), material.shininess);
    vec3 specular = light.color * spec * material.specular * light.intensity;
    
    // Затухание
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + 
                              light.quadratic * (distance * distance));
    
    return (diffuse + specular) * attenuation;
}

vec3 calculateSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec4 texColor) {
    vec3 lightDir = normalize(light.position - fragPos);
    
    // Проверка угла
    float theta = dot(lightDir, normalize(light.direction));
    float epsilon = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    
    if (theta > light.outerCutOff) {
        // Диффуз
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 diffuse = light.color * diff * light.intensity * material.albedo * texColor.rgb;
        
        // Specular (Блинн-Фонг)
        vec3 halfDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(normal, halfDir), 0.0), material.shininess);
        vec3 specular = light.color * spec * material.specular * light.intensity;
        
        // Attenuation
        float distance = length(light.position - fragPos);
        float attenuation = 1.0 / (light.constant + light.linear * distance + 
                                  light.quadratic * (distance * distance));
        
        return (diffuse + specular) * attenuation * intensity;
    }
    
    return vec3(0.0);
}

vec3 calculateDirectionalLight(DirectionalLight light, vec3 normal, vec3 viewDir, vec4 texColor, vec3 fragPos) {
    vec3 lightDir = normalize(light.direction);
    
    // Расчет тени для направленного света
    float shadow = calculateShadow(fragPosLightSpace, normal, lightDir);
    
    // Диффузная составляющая
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = light.color * diff * light.intensity * material.albedo * texColor.rgb;
    
    // Specular (Блинн-Фонг)
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), material.shininess);
    vec3 specular = light.color * spec * material.specular * light.intensity;
    
    return (diffuse + specular) * shadow;
}

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 viewDir = normalize(scene.view_position - fragPosition);
    vec4 texColor = texture(texture_sampler, fragUV);
    
    // Ambient
    vec3 ambient = scene.ambientColor * scene.ambientIntensity * material.albedo * texColor.rgb;
    
    vec3 result = ambient;


 

 vec2 uv = fragUV;
    float depth = texture(shadow_sampler, vec3(uv, 0.5)).r;
    
    // Если depth = 1.0 - shadow map пустая
    // Если depth = 0.0 - всё близко
    // Если значения разные - работает
    
    outColor = vec4(depth, depth, depth, 1.0);
return;
  



    
    
    // Направленный свет с тенями
    for (uint i = 0u; i < scene.directional_light_count; i++) {
        result += calculateDirectionalLight(directional_lights[i], normal, viewDir, texColor, fragPosition);
    }





    //outColor = vec4(result, 1.0);
    

     if (fragPosLightSpace.w == 0.0) {
        outColor = vec4(1.0, 0.0, 1.0, 1.0); // Фиолетовый = ошибка матрицы
        return;
    }
    
    // Точечные источники
    for (uint i = 0u; i < scene.point_light_count; i++) {
        result += calculatePointLight(point_lights[i], normal, fragPosition, viewDir, texColor);
    }
    
    // Прожекторы
    for (uint i = 0u; i < scene.spot_light_count; i++) {
        result += calculateSpotLight(spot_lights[i], normal, fragPosition, viewDir, texColor);
    }
    
    //outColor = vec4(result, 1.0);
}