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

struct Material {
    vec3 albedo;
    vec3 specular;  
    float shininess;
};

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

layout(binding = 0) uniform SceneUniforms {
    mat4 view_projection;     // 64 bytes
    vec3 view_position;       // 12 bytes  
    float _pad0;              // 👈 4 bytes padding (до 16)
    uint point_light_count;   // 4 bytes
    float _pad1[3];           // 👈 12 bytes padding (до 16)
};

layout(binding = 1) uniform ModelUniforms {
    mat4 model;
    mat4 normal_matrix;
    Material material;
};

layout(set = 1, binding = 0) readonly buffer PointLightsSSBO {
    PointLight point_lights[];
};

layout(location = 0) out vec4 outColor;

vec3 calculatePointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
    vec3 lightDir = normalize(light.position - fragPos);
    
    // Диффузная составляющая
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = light.color * diff * light.intensity;
    
    // Specular составляющая
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    vec3 specular = light.color * spec * material.specular * light.intensity;
    
    // Затухание (attenuation)
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + 
                              light.quadratic * (distance * distance));
    
    return (diffuse + specular) * attenuation;
}


void main() {
    vec3 normal = normalize(fragNormal);
    vec3 viewDir = normalize(view_position - fragPosition);
    
    bool Dir_light = false;
    if(Dir_light){
    // Направленный свет (опционально)
    vec3 lightDir = normalize(vec3(0.0, -1.0, 1.0));
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = fragColor * diff;
    
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    vec3 specular = material.specular * spec;
    

    // Ambient составляющая
    vec3 ambient = fragColor * 0.2;

    vec3 result = ambient + diffuse + specular;
    }

    vec3 result = fragColor * 0.2;


    for (int i = 0; i < point_light_count; i++) {
        result += calculatePointLight(point_lights[i], normal, fragPosition, viewDir);
    }

    outColor = vec4(result, 1.0);
}