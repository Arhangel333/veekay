#version 450

struct PointLight
	{
		vec3 position; // 12 bytes
		float _pad0;		   // 👈 4 bytes padding
		vec3 color;	   // 12 bytes
		float _pad1;		   // 👈 4 bytes padding
		float intensity;	   // 4 bytes
		float constant;		   // 4 bytes
		float linear;		   // 4 bytes
		float quadratic;	   // 4 bytes
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
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec2 fragUV;

layout(std140, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 view_position;
    float _pad0;
    uint point_light_count;
    uint spot_light_count;
    uint directional_light_count;
    float _pad1;
    vec3 ambientColor;
    float ambientIntensity;
    
};

layout(binding = 1) uniform ModelUniforms {
    mat4 model;
    mat4 normal_matrix;
    Material material;
};

layout(binding = 2) uniform sampler2D texture_sampler;


layout(set = 1, binding = 0) readonly buffer PointLightsSSBO {
    PointLight point_lights[];
};

layout(set = 1, binding = 1) readonly buffer SpotLightsSSBO {
    SpotLight spot_lights[];
};

layout(set = 1, binding = 2, std430) readonly buffer DirectionalLightsSSBO {
    DirectionalLight directional_lights[];
};

layout(location = 0) out vec4 outColor;

vec3 calculatePointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec4 texColor) {
    vec3 lightDir = normalize(light.position - fragPos);
    
    // Диффузная составляющая
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = light.color * diff * light.intensity * material.albedo * texColor.rgb;
    
    // 👇 SPECULAR - БЛИНН-ФОНГ (ИСПРАВЛЕНО)
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), material.shininess);
    vec3 specular = light.color * spec * material.specular * light.intensity;
    
    // Затухание (attenuation)
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

vec3 calculateDirectionalLight(DirectionalLight light, vec3 normal, vec3 viewDir, vec4 texColor) {
    vec3 lightDir = normalize(-light.direction);
    
    // Диффузная составляющая
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = light.color * diff * light.intensity * material.albedo * texColor.rgb;
    
    // Specular (Блинн-Фонг)
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), material.shininess);
    vec3 specular = light.color * spec * material.specular * light.intensity;
    
    return diffuse + specular;
}

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 viewDir = normalize(view_position - fragPosition);
    vec4 texColor = texture(texture_sampler, fragUV);
    
    
    //vec3 ambient = ambientColor * ambientIntensity * material.albedo;
    vec3 ambient = ambientColor * ambientIntensity * material.albedo * texColor.rgb;


    vec3 result = ambient;


    // 👇 НАПРАВЛЕННЫЙ СВЕТ (Блинн-Фонг)
    for (int i = 0; i < directional_light_count; i++) {
        result += calculateDirectionalLight(directional_lights[i], normal, viewDir, texColor);
    }

    // 👇 ТОЧЕЧНЫЕ ИСТОЧНИКИ (теперь используют Блинн-Фонг)
    for (int i = 0; i < point_light_count; i++) {
        result += calculatePointLight(point_lights[i], normal, fragPosition, viewDir, texColor);
    }
    
    for (int i = 0; i < spot_light_count; i++) {
        result += calculateSpotLight(spot_lights[i], normal, fragPosition, viewDir, texColor);
    }

    

    outColor = vec4(result, 1.0);

    
    
}