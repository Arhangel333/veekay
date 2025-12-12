#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // Z [0, 1] вместо [-1, 1]
#define GLM_FORCE_LEFT_HANDED		// Левая система координат
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace
{

	constexpr uint32_t max_models = 1024;
	constexpr uint32_t max_lights = 1024;

	struct Vertex
	{
		veekay::vec3 position;
		veekay::vec3 normal;
		veekay::vec2 uv;
		// NOTE: You can add more attributes
	};

	struct Material
	{
		veekay::vec3 albedo; // 12 bytes
		float _pad0;
		veekay::vec3 specular; // 12 bytes
		float shininess;	   // 4 bytes
							   // Итого: 32 bytes (совпадает с GLSL)
	};

	struct ModelUniforms
	{
		veekay::mat4 model;
		veekay::mat4 normal_matrix; // Для преобразования нормалей
		Material material;
	};

	struct Mesh
	{
		veekay::graphics::Buffer *vertex_buffer;
		veekay::graphics::Buffer *index_buffer;
		uint32_t indices;
	};

	struct Transform
	{
		veekay::vec3 position = {};
		veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
		veekay::vec3 rotation = {};

		// NOTE: Model matrix (translation, rotation and scaling)
		veekay::mat4 matrix() const;
	};

	struct Model
	{
		Mesh mesh;
		Transform transform;
		Material material; // Заменяем vec3 albedo_color на Material
	};

	struct Camera
	{
		constexpr static float default_fov = 60.0f;
		constexpr static float default_near_plane = 0.01f;
		constexpr static float default_far_plane = 100.0f;

		veekay::vec3 position = {};
		veekay::vec3 rotation = {};

		float fov = default_fov;
		float near_plane = default_near_plane;
		float far_plane = default_far_plane;

		// NOTE: View matrix of camera (inverse of a transform)
		veekay::mat4 view() const;

		// NOTE: View and projection composition
		veekay::mat4 view_projection(float aspect_ratio) const;
	};

	struct PointLight
	{
		veekay::vec3 position; // 12 bytes
		float _pad0;
		veekay::vec3 color; // 12 bytes
		float _pad1;
		float intensity; // 4 bytes
		float constant;	 // 4 bytes
		float linear;	 // 4 bytes
		float quadratic; // 4 bytes
	};

	struct SpotLight
	{
		veekay::vec3 position;	// 12 bytes
		float _pad0;			// 4 bytes padding
		veekay::vec3 direction; // 12 bytes
		float _pad1;			// 4 bytes padding
		veekay::vec3 color;		// 12 bytes
		float _pad2;			// 4 bytes padding
		float intensity;		// 4 bytes
		float cutOff;			// 4 bytes (cos(внутренний угол))
		float outerCutOff;		// 4 bytes (cos(внешний угол))
		float constant;			// 4 bytes
		float linear;			// 4 bytes
		float quadratic;		// 4 bytes
		float _pad3[2];
	};

	struct DirectionalLight
	{
		veekay::vec3 direction;
		float _pad0;
		veekay::vec3 color;
		float intensity;
	};

	struct SceneUniforms
	{
		veekay::mat4 view_projection; // 64 bytes
		veekay::mat4 light_view_projection;
		veekay::vec3 view_position;		  // 12 bytes
		float _pad0;					  // 4 bytes (16)
		uint32_t point_light_count;		  // 4 bytes
		uint32_t spot_light_count;		  // 4 bytes
		uint32_t directional_light_count; // 4 bytes (12)
		float _pad1[1];
		veekay::vec3 ambientColor; // 12 bytes
		float ambientIntensity;
	};

	// NOTE: Scene objects
	inline namespace
	{
		Camera camera{
			.position = {0.0f, -0.5f, -3.0f} // {0.0f, -0.5f, -3.0f}
		};

		VkImageLayout shadow_map_layout = VK_IMAGE_LAYOUT_UNDEFINED;

		std::vector<Model> models;

		// SSBO для точечных источников света
		veekay::graphics::Buffer *point_lights_ssbo;
		std::vector<PointLight> point_lights;

		veekay::graphics::Buffer *spot_lights_ssbo;
		std::vector<SpotLight> spot_lights;

		veekay::graphics::Buffer *directional_lights_ssbo;
		std::vector<DirectionalLight> directional_lights;

		veekay::vec3 ambientColor = {1.0f, 1.0f, 1.0f};
		float ambientIntensity = 1.0f;

		// Дескрипторы для SSBO
		VkDescriptorSetLayout ssbo_descriptor_set_layout;
		VkDescriptorSet ssbo_descriptor_set;

		// VkDescriptorSet spot_lights_descriptor_set;
		// VkDescriptorSetLayout spot_lights_layout;
	}

	// NOTE: Vulkan objects
	inline namespace
	{
		VkShaderModule vertex_shader_module;
		VkShaderModule fragment_shader_module;

		VkDescriptorPool descriptor_pool;
		VkDescriptorSetLayout descriptor_set_layout;
		VkDescriptorSet descriptor_set;

		VkPipelineLayout pipeline_layout;
		VkPipeline pipeline;

		veekay::graphics::Buffer *scene_uniforms_buffer;
		veekay::graphics::Buffer *model_uniforms_buffer;

		Mesh plane_mesh;
		Mesh cube_mesh;

		veekay::graphics::Texture *missing_texture;
		VkSampler missing_texture_sampler;

		veekay::graphics::Texture *texture;
		VkSampler texture_sampler;

		// Shadow map ресурсы
		VkImage shadow_depth_image;
		VkDeviceMemory shadow_depth_memory;
		VkImageView shadow_depth_view;
		VkSampler shadow_sampler;
		VkPipeline shadow_pipeline;
		VkShaderModule shadow_vert_shader;
		VkPipelineLayout shadow_pipeline_layout;
		VkRenderPass shadow_render_pass;
		VkFramebuffer shadow_framebuffer;
		veekay::mat4 curr_light_view_projection;
	}

	float toRadians(float degrees)
	{
		return degrees * float(M_PI) / 180.0f;
	}

	uint32_t findMemoryType(VkPhysicalDevice physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties mem_properties;
		vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

		for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++)
		{
			if ((type_filter & (1 << i)) &&
				(mem_properties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				return i;
			}
		}

		throw std::runtime_error("Failed to find suitable memory type!");
		return 0;
	}

	veekay::vec4 cross(const veekay::vec4 &a, const veekay::vec4 &b)
	{
		veekay::vec4 result;
		result.x = a.y * b.z - a.z * b.y;
		result.y = a.z * b.x - a.x * b.z;
		result.z = a.x * b.y - a.y * b.x;
		result.w = 0.0f; // Вектор, не точка!
		return result;
	}

	veekay::vec3 cross(const veekay::vec3 &a, const veekay::vec3 &b)
	{
		veekay::vec3 result;
		result.x = a.y * b.z - a.z * b.y;
		result.y = a.z * b.x - a.x * b.z;
		result.z = a.x * b.y - a.y * b.x;
		return result;
	}

	veekay::vec4 multiplyVec4Mat4(const veekay::vec4 &v, const veekay::mat4 &m)
			{
				return veekay::vec4{
					v.x * m[0][0] + v.y * m[1][0] + v.z * m[2][0] + v.w * m[3][0],
					v.x * m[0][1] + v.y * m[1][1] + v.z * m[2][1] + v.w * m[3][1],
					v.x * m[0][2] + v.y * m[1][2] + v.z * m[2][2] + v.w * m[3][2],
					v.x * m[0][3] + v.y * m[1][3] + v.z * m[2][3] + v.w * m[3][3]};
			}

	veekay::mat4 lookAt(const veekay::vec3& eye, const veekay::vec3& target, const veekay::vec3& up_dir)
{

	// 1. Forward = normalize(target - eye)
    veekay::vec3 f = target - eye;
    float len_f = sqrt(f.x*f.x + f.y*f.y + f.z*f.z);
    if (len_f > 0.0001f) {
        f.x /= len_f; f.y /= len_f; f.z /= len_f;
    } else {
        f = {0.0f, 0.0f, 1.0f};
    }
    
    // 2. Right = normalize(cross(f, up_dir))
    veekay::vec3 r = {
        f.y * up_dir.z - f.z * up_dir.y,
        f.z * up_dir.x - f.x * up_dir.z,
        f.x * up_dir.y - f.y * up_dir.x
    };
    
    float len_r = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
    if (len_r > 0.0001f) {
        r.x /= len_r; r.y /= len_r; r.z /= len_r;
    } else {
        r = {1.0f, 0.0f, 0.0f};
    }
    
    // 3. Up = cross(r, f) - уже ортогонален
    veekay::vec3 u = {
        r.y * f.z - r.z * f.y,
        r.z * f.x - r.x * f.z,
        r.x * f.y - r.y * f.x
    };
    
    // НЕ инвертируем u для Vulkan! View матрица одинакова!
    // Инверсия Y делается в проекционной матрице!
    
    // 4. Translation = -dot(axis, eye)
    float tx = -(r.x * eye.x + r.y * eye.y + r.z * eye.z);
    float ty = -(u.x * eye.x + u.y * eye.y + u.z * eye.z);
    float tz = -(f.x * eye.x + f.y * eye.y + f.z * eye.z);
    
    // 5. Row-major матрица
    return veekay::mat4{
        r.x, r.y, r.z, tx,   // Строка 0: right
        u.x, u.y, u.z, ty,   // Строка 1: up
        f.x, f.y, f.z, tz,   // Строка 2: forward
        0.0f, 0.0f, 0.0f, 1.0f
    };
}

	// Добавьте где-нибудь в начале файла (после includes)
	void printMatrix(const char *name, const veekay::mat4 &m)
	{
		printf("\n=== %s ===\n", name);

		for (size_t i = 0; i < 4; i++)
		{
			printf("[");
			for (size_t j = 0; j < 4; j++)
			{
				printf("%8.3f ", m[i][j]);
			}
			printf("]\n");
		}
		printf("\n");
	}

	void printVector3(const char *name, const veekay::vec3 &v)
	{
		printf("%s: [%6.2f %6.2f %6.2f]\n", name, v.x, v.y, v.z);
	}

	void printVector4(const char *name, const veekay::vec4 &v)
	{
		printf("%s: [%6.2f %6.2f %6.2f %6.2f]\n", name, v.x, v.y, v.z, v.w);
	}

	veekay::mat4 orthographic(float left, float right, float bottom, float top, float near, float far)
{
    // Vulkan: Y вниз, Z [0, 1]
    float rl = right - left;
    float tb = top - bottom;    // top - bottom!
    float fn = near - far;
    
    // Row-major матрица:
    return veekay::mat4{
        2.0f / rl, 0.0f, 0.0f, -(right + left) / rl,           // Строка 0: X
        0.0f, -2.0f / tb, 0.0f, (top + bottom) / tb,           // Строка 1: Y (минус для Vulkan!)
        0.0f, 0.0f, 1.0f / fn, -near / fn,                     // Строка 2: Z
        0.0f, 0.0f, 0.0f, 1.0f                                 // Строка 3
    };
}

	// Добавим эти функции если их нет в veekay
	veekay::mat4 rotation_x(float angle)
	{
		float c = cos(angle);
		float s = sin(angle);
		return veekay::mat4{
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, c, -s, 0.0f,
			0.0f, s, c, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f};
	}

	veekay::mat4 rotation_y(float angle)
	{
		float c = cos(angle);
		float s = sin(angle);
		return veekay::mat4{
			c, 0.0f, s, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			-s, 0.0f, c, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f};
	}

	veekay::mat4 rotation_z(float angle)
	{
		float c = cos(angle);
		float s = sin(angle);
		return veekay::mat4{
			c, -s, 0.0f, 0.0f,
			s, c, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f};
	}

	veekay::mat4 scaling(const veekay::vec3 &scale)
	{
		return veekay::mat4{
			scale.x, 0.0f, 0.0f, 0.0f,
			0.0f, scale.y, 0.0f, 0.0f,
			0.0f, 0.0f, scale.z, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f};
	}

	veekay::mat4 Transform::matrix() const
	{
		auto scale_mat = scaling(scale);
		auto rot_x = rotation_x(rotation.x);
		auto rot_y = rotation_y(rotation.y);
		auto rot_z = rotation_z(rotation.z);
		auto rot_mat = rot_z * rot_y * rot_x;
		auto trans_mat = veekay::mat4::translation(position);
		return trans_mat * rot_mat * scale_mat;
	}

	veekay::mat4 Camera::view() const
	{
		// TODO: Rotation
		auto rot_x = rotation_x(-rotation.x); // Используем нашу функцию
		auto rot_y = rotation_y(rotation.y);  // Используем нашу функцию
		auto rot_mat = rot_y * rot_x;		  // Yaw then pitch
		auto trans_mat = veekay::mat4::translation(-position);
		return trans_mat * rot_mat;
	}

	veekay::mat4 Camera::view_projection(float aspect_ratio) const
	{
		auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
		return view() * projection;
	}

	// NOTE: Loads shader byte code from file
	// NOTE: Your shaders are compiled via CMake with this code too, look it up
	VkShaderModule loadShaderModule(const char *path)
	{
		std::ifstream file(path, std::ios::binary | std::ios::ate);
		size_t size = file.tellg();
		// printf("shader %s size: %d\n", path, (int)size);
		std::vector<uint32_t> buffer(size / sizeof(uint32_t));
		file.seekg(0);
		file.read(reinterpret_cast<char *>(buffer.data()), size);
		file.close();

		VkShaderModuleCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = size,
			.pCode = buffer.data(),
		};

		VkShaderModule result;
		if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS)
		{
			return nullptr;
		}

		return result;
	}

	void barier_to_Read(VkImageLayout &shadow_map_layout, VkCommandBuffer cmd)
	{
		if (shadow_map_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
		{
			printf("ERROR: Expected ATTACHMENT layout, but got: %i\n", shadow_map_layout);
			return; // или assert
		}
		printf("Need ATTACHMENT Now: %i\n", shadow_map_layout);
		VkImageMemoryBarrier barrier_to_read = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
			.oldLayout = shadow_map_layout,
			.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = shadow_depth_image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};

		shadow_map_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

		printf("Set READ_ONLY Now: %i\n", shadow_map_layout);

		vkCmdPipelineBarrier(cmd,
							 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
							 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
							 0,
							 0, nullptr,
							 0, nullptr,
							 1, &barrier_to_read);
		return;
	}

	void barier_to_Write(VkImageLayout &shadow_map_layout, VkCommandBuffer cmd)
	{
		if (shadow_map_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)
		{
			printf("ERROR: Expected READ_ONLY layout, but got: %i\n", shadow_map_layout);
			return; // или assert
		}
		printf("Need READ_ONLY Now: %i\n", shadow_map_layout);
		// Переход shadow map в layout для записи глубины
		VkImageMemoryBarrier barrier_to_write = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
			.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.oldLayout = shadow_map_layout,
			.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = shadow_depth_image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},

		};

		shadow_map_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		printf("Set ATTACHMENT Now: %i\n", shadow_map_layout);

		vkCmdPipelineBarrier(cmd,
							 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
							 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
							 0,
							 0, nullptr,
							 0, nullptr,
							 1, &barrier_to_write);
		return;
	}

	void initialize(VkCommandBuffer cmd)
	{

		VkDevice &device = veekay::app.vk_device;
		VkPhysicalDevice &physical_device = veekay::app.vk_physical_device;

		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};
		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0,						  // NOTE: First attribute
				.binding = 0,						  // NOTE: First vertex buffer
				.format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
				.offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
			},
			{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, normal),
			},
			{
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT,
				.offset = offsetof(Vertex, uv),
			},
		};
		// NOTE: Every three vertices make up a triangle,
		//       so our vertex buffer contains a "list of triangles"
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};
		// NOTE: Let rasterizer draw on the entire window
		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

			.viewportCount = 1,
			.pViewports = &viewport,

			.scissorCount = 1,
			.pScissors = &scissor,
		};
		// NOTE: Use 1 sample per pixel
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		{ // NOTE: Build graphics pipeline
			vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
			if (!vertex_shader_module)
			{
				std::cerr << "Failed to load Vulkan vertex shader from file\n";
				veekay::app.running = false;
				return;
			}

			fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
			if (!fragment_shader_module)
			{
				std::cerr << "Failed to load Vulkan fragment shader from file\n";
				veekay::app.running = false;
				return;
			}

			VkPipelineShaderStageCreateInfo stage_infos[2];

			// NOTE: Vertex shader stage
			stage_infos[0] = VkPipelineShaderStageCreateInfo{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_VERTEX_BIT,
				.module = vertex_shader_module,
				.pName = "main",
			};

			// NOTE: Fragment shader stage
			stage_infos[1] = VkPipelineShaderStageCreateInfo{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
				.module = fragment_shader_module,
				.pName = "main",
			};

			// NOTE: Declare vertex attributes

			// NOTE: Describe inputs
			VkPipelineVertexInputStateCreateInfo input_state_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
				.vertexBindingDescriptionCount = 1,
				.pVertexBindingDescriptions = &buffer_binding,
				.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
				.pVertexAttributeDescriptions = attributes,
			};

			// NOTE: Declare clockwise triangle order as front-facing
			//       Discard triangles that are facing away
			//       Fill triangles, don't draw lines instaed

			VkPipelineRasterizationStateCreateInfo raster_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
				.polygonMode = VK_POLYGON_MODE_FILL,
				.cullMode = VK_CULL_MODE_BACK_BIT,
				.frontFace = VK_FRONT_FACE_CLOCKWISE,
				.depthBiasEnable = VK_TRUE,
				.depthBiasConstantFactor = 2.0f,
				.depthBiasClamp = 0.0f,
				.depthBiasSlopeFactor = 2.0f,
				.lineWidth = 1.0f,
			};

			// NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
			VkPipelineDepthStencilStateCreateInfo depth_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
				.depthTestEnable = true,
				.depthWriteEnable = true,
				.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
			};

			// NOTE: Let fragment shader write all the color channels
			VkPipelineColorBlendAttachmentState attachment_info{
				.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
								  VK_COLOR_COMPONENT_G_BIT |
								  VK_COLOR_COMPONENT_B_BIT |
								  VK_COLOR_COMPONENT_A_BIT,
			};

			// NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
			VkPipelineColorBlendStateCreateInfo blend_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

				.logicOpEnable = false,
				.logicOp = VK_LOGIC_OP_COPY,

				.attachmentCount = 1,
				.pAttachments = &attachment_info};

			{
				VkDescriptorPoolSize pools[] = {
					{
						.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
						.descriptorCount = 8,
					},
					{
						.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
						.descriptorCount = 8,
					},
					{
						.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						.descriptorCount = 16,
					},
					{
						.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, //  ДЛЯ SSBO!
						.descriptorCount = 3,					   //  2 слота для SSBO
					}};

				VkDescriptorPoolCreateInfo info{
					.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
					.maxSets = 4,
					.poolSizeCount = sizeof(pools) / sizeof(pools[0]),
					.pPoolSizes = pools,
				};

				if (vkCreateDescriptorPool(device, &info, nullptr,
										   &descriptor_pool) != VK_SUCCESS)
				{
					std::cerr << "Failed to create Vulkan descriptor pool\n";
					veekay::app.running = false;
					return;
				}
			}

			// NOTE: Descriptor set layout specification
			{
				VkDescriptorSetLayoutBinding bindings[] = {
					{
						.binding = 0,
						.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
					},
					{
						.binding = 1,
						.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
					},
					{
						.binding = 2,
						.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
					},
					{
						.binding = 3,
						.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
					},
				};

				VkDescriptorSetLayoutCreateInfo info{
					.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
					.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
					.pBindings = bindings,
				};

				if (vkCreateDescriptorSetLayout(device, &info, nullptr,
												&descriptor_set_layout) != VK_SUCCESS)
				{
					std::cerr << "Failed to create Vulkan descriptor set layout\n";
					veekay::app.running = false;
					return;
				}
			}

			{
				VkDescriptorSetAllocateInfo info{
					.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
					.descriptorPool = descriptor_pool,
					.descriptorSetCount = 1,
					.pSetLayouts = &descriptor_set_layout,
				};

				if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS)
				{
					std::cerr << "Failed to create Vulkan descriptor set\n";
					veekay::app.running = false;
					return;
				}
			}

			// NOTE: Declare external data sources, only push constants this time
			// 1. Создаем макет дескриптора для SSBO
			{ // ssbo_bindings
				VkDescriptorSetLayoutBinding ssbo_bindings[] = {
					{
						.binding = 0, // point lights
						.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, // Используем во фрагментном шейдере
					},
					{
						.binding = 1, // spot_lights
						.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
					},
					{
						.binding = 2, // directional lights
						.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
					}};

				VkDescriptorSetLayoutCreateInfo ssbo_layout_info{
					.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
					.bindingCount = 3,
					.pBindings = ssbo_bindings,
				};

				if (vkCreateDescriptorSetLayout(device, &ssbo_layout_info, nullptr,
												&ssbo_descriptor_set_layout) != VK_SUCCESS)
				{
					std::cerr << "Failed to create Vulkan SSBO descriptor set layout\n";
					veekay::app.running = false;
					return;
				}
			}

			VkDescriptorSetLayout descriptor_set_layouts[] = {
				descriptor_set_layout, //  Первый: для UBO (камера, материалы)
				ssbo_descriptor_set_layout};

			VkPipelineLayoutCreateInfo layout_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = 2,
				.pSetLayouts = descriptor_set_layouts, //  Оба макета
			};

			// NOTE: Create pipeline layout
			if (vkCreatePipelineLayout(device, &layout_info,
									   nullptr, &pipeline_layout) != VK_SUCCESS)
			{
				std::cerr << "Failed to create Vulkan pipeline layout\n";
				veekay::app.running = false;
				return;
			}

			VkGraphicsPipelineCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
				.stageCount = 2,
				.pStages = stage_infos,
				.pVertexInputState = &input_state_info,
				.pInputAssemblyState = &assembly_state_info,
				.pViewportState = &viewport_info,
				.pRasterizationState = &raster_info,
				.pMultisampleState = &sample_info,
				.pDepthStencilState = &depth_info,
				.pColorBlendState = &blend_info,
				.layout = pipeline_layout,
				.renderPass = veekay::app.vk_render_pass,
			};

			// NOTE: Create graphics pipeline
			if (vkCreateGraphicsPipelines(device, nullptr,
										  1, &info, nullptr, &pipeline) != VK_SUCCESS)
			{
				std::cerr << "Failed to create Vulkan pipeline\n";
				veekay::app.running = false;
				return;
			}
		}

		// 1. Загружаем shadow vertex shader
		shadow_vert_shader = loadShaderModule("./shaders/shadow.vert.spv");
		if (!shadow_vert_shader)
		{
			std::cerr << "Failed to load Vulkan shadow vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		{
			// Добавляем push constants для матрицы модели
			VkPushConstantRange push_constant_range{
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
				.offset = 0,
				.size = sizeof(veekay::mat4) * 2,
			};

			VkPipelineLayoutCreateInfo layout_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = 0,
				.pSetLayouts = nullptr,
				.pushConstantRangeCount = 1,
				.pPushConstantRanges = &push_constant_range,
			};

			if (vkCreatePipelineLayout(veekay::app.vk_device, &layout_info,
									   nullptr, &shadow_pipeline_layout) != VK_SUCCESS)
			{
				std::cerr << "Failed to create Vulkan shadow pipeline layout\n";
				veekay::app.running = false;
				return;
			}
		}

		scene_uniforms_buffer = new veekay::graphics::Buffer(
			sizeof(SceneUniforms),
			nullptr,
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

		model_uniforms_buffer = new veekay::graphics::Buffer(
			max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
			nullptr,
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

		// 2. Выделяем дескрипторный сет для SSBO из пула
		{
			VkDescriptorSetAllocateInfo ssbo_alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &ssbo_descriptor_set_layout,
			};

			if (vkAllocateDescriptorSets(device, &ssbo_alloc_info, &ssbo_descriptor_set) != VK_SUCCESS)
			{
				std::cerr << "Failed to create Vulkan SSBO descriptor set\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: This texture and sampler is used when texture could not be loaded
		{
			VkSamplerCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
				.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			};

			if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS)
			{
				std::cerr << "Failed to create Vulkan texture sampler\n";
				veekay::app.running = false;
				return;
			}

			uint32_t pixels[] = {
				0xff000000,
				0xffff00ff,
				0xffff00ff,
				0xff000000,
			};

			missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
															VK_FORMAT_B8G8R8A8_UNORM,
															pixels);
		}

		{
			std::vector<unsigned char> image;
			unsigned width, height;

			// Загружаем PNG
			unsigned error = lodepng::decode(image, width, height, "textures/mai.png"); // cyber.png
			if (error)
			{
				std::cerr << "Failed to load texture: " << lodepng_error_text(error) << std::endl;
				texture = missing_texture;
				texture_sampler = missing_texture_sampler;
			}
			else
			{
				// Создаем текстуру из загруженного изображения
				texture = new veekay::graphics::Texture(
					cmd,
					width,
					height,
					VK_FORMAT_R8G8B8A8_UNORM,
					image.data());

				// Создаем нормальный сэмплер (сейчас у texture_sampler нет создания!)
				VkSamplerCreateInfo sampler_info{
					.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
					.magFilter = VK_FILTER_LINEAR,
					.minFilter = VK_FILTER_LINEAR,
					.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
					.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
					.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
					.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
					.anisotropyEnable = VK_TRUE,
					.maxAnisotropy = 16.0f,
				};

				if (vkCreateSampler(device, &sampler_info, nullptr, &texture_sampler) != VK_SUCCESS)
				{
					std::cerr << "Failed to create texture sampler\n";
					texture_sampler = missing_texture_sampler;
				}
			}
		}

		// Создание shadow map texture
		{
			VkDevice &device = veekay::app.vk_device;
			VkPhysicalDevice &physical_device = veekay::app.vk_physical_device;

			const uint32_t SHADOW_WIDTH = 1024;
			const uint32_t SHADOW_HEIGHT = 1024;

			// 1. Создаем VkImage
			VkImageCreateInfo image_info{
				.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
				.imageType = VK_IMAGE_TYPE_2D,
				.format = VK_FORMAT_D32_SFLOAT,
				.extent = {SHADOW_WIDTH, SHADOW_HEIGHT, 1},
				.mipLevels = 1,
				.arrayLayers = 1,
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.tiling = VK_IMAGE_TILING_OPTIMAL,
				.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
						 VK_IMAGE_USAGE_SAMPLED_BIT,
				.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			};

			if (vkCreateImage(device, &image_info, nullptr, &shadow_depth_image) != VK_SUCCESS)
			{
				std::cerr << "Failed to create shadow depth image\n";
				veekay::app.running = false;
				return;
			}

			// 2. Выделяем память
			VkMemoryRequirements mem_reqs;
			vkGetImageMemoryRequirements(device, shadow_depth_image, &mem_reqs);

			VkMemoryAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
				.allocationSize = mem_reqs.size,
				.memoryTypeIndex = findMemoryType(
					physical_device,
					mem_reqs.memoryTypeBits,
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
			};

			if (vkAllocateMemory(device, &alloc_info, nullptr, &shadow_depth_memory) != VK_SUCCESS)
			{
				std::cerr << "Failed to allocate shadow depth memory\n";
				veekay::app.running = false;
				return;
			}

			vkBindImageMemory(device, shadow_depth_image, shadow_depth_memory, 0);

			// 3. Создаем Image View
			VkImageViewCreateInfo view_info{
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = shadow_depth_image,
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = VK_FORMAT_D32_SFLOAT,
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};

			if (vkCreateImageView(device, &view_info, nullptr, &shadow_depth_view) != VK_SUCCESS)
			{
				std::cerr << "Failed to create shadow depth view\n";
				veekay::app.running = false;
				return;
			}

			{ // shadow_render_pass Создаем
				VkAttachmentDescription depth_attachment = {
					.format = VK_FORMAT_D32_SFLOAT,
					.samples = VK_SAMPLE_COUNT_1_BIT,
					.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
					.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
					.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
					.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
					.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
					.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
				};

				VkAttachmentReference depth_attachment_ref = {
					.attachment = 0,
					.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
				};

				VkSubpassDescription subpass = {
					.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
					.colorAttachmentCount = 0,
					.pDepthStencilAttachment = &depth_attachment_ref,
				};

				VkRenderPassCreateInfo render_pass_info = {
					.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
					.attachmentCount = 1,
					.pAttachments = &depth_attachment,
					.subpassCount = 1,
					.pSubpasses = &subpass,
				};

				vkCreateRenderPass(device, &render_pass_info, nullptr, &shadow_render_pass);
			}

			{ // 2. Создать shadow framebuffer
				VkFramebufferCreateInfo fb_info = {
					.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
					.renderPass = shadow_render_pass,
					.attachmentCount = 1,
					.pAttachments = &shadow_depth_view,
					.width = 1024,
					.height = 1024,
					.layers = 1,
				};

				vkCreateFramebuffer(device, &fb_info, nullptr, &shadow_framebuffer);
			}

			// 3. Создаем shadow pipeline
			{
				VkPipelineShaderStageCreateInfo shadow_stage{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
					.stage = VK_SHADER_STAGE_VERTEX_BIT,
					.module = shadow_vert_shader,
					.pName = "main",
				};

				VkVertexInputAttributeDescription shadow_attributes[] = {
					{.location = 0,
					 .binding = 0,
					 .format = VK_FORMAT_R32G32B32_SFLOAT,
					 .offset = offsetof(Vertex, position)}};

				VkPipelineVertexInputStateCreateInfo shadow_input_state_info{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
					.vertexBindingDescriptionCount = 1,
					.pVertexBindingDescriptions = &buffer_binding,
					.vertexAttributeDescriptionCount = 1,
					.pVertexAttributeDescriptions = shadow_attributes,
				};

				VkPipelineRasterizationStateCreateInfo shadow_raster_info{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
					.depthClampEnable = VK_FALSE,
					.rasterizerDiscardEnable = VK_FALSE,
					.polygonMode = VK_POLYGON_MODE_FILL,
					.cullMode = VK_CULL_MODE_BACK_BIT,
					.frontFace = VK_FRONT_FACE_CLOCKWISE,
					.depthBiasEnable = VK_TRUE,
					.depthBiasConstantFactor = 2.0f,
					.depthBiasClamp = 0.0f,
					.depthBiasSlopeFactor = 2.0f,
					.lineWidth = 1.0f,
				};

				VkPipelineDepthStencilStateCreateInfo shadow_depth_info{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
					.depthTestEnable = VK_TRUE,
					.depthWriteEnable = VK_TRUE,
					.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
					.depthBoundsTestEnable = VK_FALSE,
					.stencilTestEnable = VK_FALSE,
				};

				VkGraphicsPipelineCreateInfo shadow_pipeline_info{
					.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
					.pNext = nullptr,
					.stageCount = 1,
					.pStages = &shadow_stage,
					.pVertexInputState = &shadow_input_state_info,
					.pInputAssemblyState = &assembly_state_info,
					.pViewportState = &viewport_info,
					.pRasterizationState = &shadow_raster_info,
					.pMultisampleState = &sample_info,
					.pDepthStencilState = &shadow_depth_info,
					.pColorBlendState = nullptr,
					.pDynamicState = nullptr,
					.layout = shadow_pipeline_layout,
					.renderPass = shadow_render_pass,
					.subpass = 0,
					.basePipelineHandle = VK_NULL_HANDLE,
					.basePipelineIndex = -1,
				};

				if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE,
											  1, &shadow_pipeline_info, nullptr, &shadow_pipeline) != VK_SUCCESS)
				{
					std::cerr << "Failed to create Vulkan shadow pipeline\n";
					veekay::app.running = false;
					return;
				}
			}

			// 4. Создаем Sampler с compareEnable
			VkSamplerCreateInfo sampler_info{
				.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
				.magFilter = VK_FILTER_LINEAR,
				.minFilter = VK_FILTER_LINEAR,
				.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
				.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
				.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
				.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
				.compareEnable = VK_TRUE,
				.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
				.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
				.unnormalizedCoordinates = VK_FALSE,
			};

			if (vkCreateSampler(device, &sampler_info, nullptr, &shadow_sampler) != VK_SUCCESS)
			{
				std::cerr << "Failed to create shadow sampler\n";
				veekay::app.running = false;
				return;
			}

			std::cout << "Shadow map created: " << SHADOW_WIDTH << "x" << SHADOW_HEIGHT << std::endl;
		}

		{
			VkDescriptorImageInfo shadow_image_info{
				.sampler = shadow_sampler,
				.imageView = shadow_depth_view,
				.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
			};

			VkWriteDescriptorSet shadow_write{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 3, // Новый binding для shadow map
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &shadow_image_info,
			};

			vkUpdateDescriptorSets(device, 1, &shadow_write, 0, nullptr);
		}

		{
			VkDescriptorBufferInfo buffer_infos[] = {
				{
					.buffer = scene_uniforms_buffer->buffer,
					.offset = 0,
					.range = sizeof(SceneUniforms),
				},
				{
					.buffer = model_uniforms_buffer->buffer,
					.offset = 0,
					.range = sizeof(ModelUniforms),
				},
			};

			VkWriteDescriptorSet write_infos[] = {
				{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = descriptor_set,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &buffer_infos[0],
				},
				{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = descriptor_set,
					.dstBinding = 1,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.pBufferInfo = &buffer_infos[1],
				},
			};

			vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
								   write_infos, 0, nullptr);
		}

		{
			VkDescriptorImageInfo texture_image_info{
				.sampler = texture_sampler,
				.imageView = texture->view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};

			VkWriteDescriptorSet texture_write{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 2, // Binding 2 для текстуры
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &texture_image_info,
			};

			vkUpdateDescriptorSets(device, 1, &texture_write, 0, nullptr);
		}

		// NOTE: Plane mesh initialization
		{
			// (v0)------(v1)
			//  |  \       |
			//  |   `--,   |
			//  |       \  |
			// (v3)------(v2)
			std::vector<Vertex> vertices = {
				{{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
				{{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
				{{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
				{{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
			};

			std::vector<uint32_t> indices = {
				0, 1, 2, 2, 3, 0};

			plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
				vertices.size() * sizeof(Vertex), vertices.data(),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

			plane_mesh.index_buffer = new veekay::graphics::Buffer(
				indices.size() * sizeof(uint32_t), indices.data(),
				VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

			plane_mesh.indices = uint32_t(indices.size());
		}

		// NOTE: Cube mesh initialization
		{
			std::vector<Vertex> vertices = {
				{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
				{{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
				{{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
				{{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

				{{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
				{{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
				{{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
				{{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

				{{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
				{{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
				{{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
				{{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},

				{{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
				{{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
				{{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
				{{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

				{{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
				{{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
				{{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
				{{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},

				{{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
				{{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
				{{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
				{{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
			};

			std::vector<uint32_t> indices = {
				0,
				1,
				2,
				2,
				3,
				0,
				4,
				5,
				6,
				6,
				7,
				4,
				8,
				9,
				10,
				10,
				11,
				8,
				12,
				13,
				14,
				14,
				15,
				12,
				16,
				17,
				18,
				18,
				19,
				16,
				20,
				21,
				22,
				22,
				23,
				20,
			};

			cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
				vertices.size() * sizeof(Vertex), vertices.data(),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

			cube_mesh.index_buffer = new veekay::graphics::Buffer(
				indices.size() * sizeof(uint32_t), indices.data(),
				VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

			cube_mesh.indices = uint32_t(indices.size());
		}

		// NOTE: Add models to scene
		models.emplace_back(Model{
			.mesh = plane_mesh,
			.transform = Transform{},
			//.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f} {0.0f, 0.0f, 0.0f},
			.material = Material{
				.albedo = {1.0f, 1.0f, 1.0f},
				.specular = {0.5f, 0.5f, 0.5f},
				.shininess = 60.0f}});

		models.emplace_back(
			Model{
				.mesh = cube_mesh,
				.transform = Transform{
					.position = {-2.0f, -0.5f, -1.5f},
				},
				.material = Material{.albedo = {1.0f, 0.0f, 0.0f}, .specular = {0.7f, 0.7f, 0.7f}, .shininess = 100.0f}});

		models.emplace_back(Model{
			.mesh = cube_mesh,
			.transform = Transform{
				.position = {1.5f, -0.5f, -0.5f},
			},
			.material = Material{.albedo = {0.0f, 1.0f, 0.0f}, .specular = {0.7f, 0.7f, 0.7f}, .shininess = 100.0f}});

		models.emplace_back(Model{
			.mesh = cube_mesh,
			.transform = Transform{
				.position = {0.0f, -0.5f, 1.0f},
			},
			.material = Material{.albedo = {0.0f, 0.0f, 1.0f}, .specular = {1.0f, 1.0f, 1.0f}, .shininess = 100.0f}});

		float intens = 1.0f;
		// 1. Создаем несколько источников света
		point_lights.push_back(PointLight{
			.position = {-2.0f, -2.0f, 2.0f}, //  Свет сверху
			.color = {0.0f, 1.0f, 0.0f},	  //  Green light
			.intensity = intens,
			.constant = 1.0f,
			.linear = 0.09f,
			.quadratic = 0.032f});

		point_lights.push_back(PointLight{
			.position = {2.0f, -2.0f, 0.0f}, //  Свет справа-спереди
			.color = {1.0f, 0.0f, 0.0f},	 //  Красный свет
			.intensity = intens,
			.constant = 1.0f,
			.linear = 0.09f,
			.quadratic = 0.032f});

		point_lights.push_back(PointLight{
			.position = {-1.0f, -2.0f, -2.0f}, //  Свет слева-сзади
			.color = {0.0f, 0.0f, 1.0f},	   //  Синий свет
			.intensity = intens,
			.constant = 1.0f,
			.linear = 0.09f,
			.quadratic = 0.032f});

		spot_lights.push_back(SpotLight{
			.position = {0.0f, -5.0f, 0.0f},
			.direction = {0.0f, -1.0f, 0.0f},
			.color = {1.0f, 1.0f, 0.0f}, // Желтый прожектор
			.intensity = 1.5f,
			.cutOff = float(cos(toRadians(30.0f))),
			.outerCutOff = float(cos(toRadians(45.0f))),
			.constant = 1.0f,
			.linear = 0.09f,
			.quadratic = 0.032f});

		directional_lights.push_back(DirectionalLight{
			.direction = {1.0f, -1.0f, 0.0f},
			.color = {1.0f, 1.0f, 1.0f}, // Желтый прожектор
			.intensity = 1.5f,
		});

		// 2. Создаем SSBO буфер и заполняем его источниками света
		point_lights_ssbo = new veekay::graphics::Buffer(
			sizeof(PointLight) * max_lights,   //  Размер = размер одного источника × количество
			point_lights.data(),			   //  Данные = массив источников света
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT //  Тип = Storage Buffer
		);

		// Создаю SSBO для прожекторов
		spot_lights_ssbo = new veekay::graphics::Buffer(
			sizeof(SpotLight) * max_lights,
			spot_lights.data(),
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

		directional_lights_ssbo = new veekay::graphics::Buffer(
			sizeof(DirectionalLight) * max_lights,
			nullptr,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

		// 3. Связываем SSBO буфер с дескриптором
		{
			VkDescriptorBufferInfo ssbo_buffer_info{
				.buffer = point_lights_ssbo->buffer, //  Наш SSBO буфер
				.offset = 0,
				.range = sizeof(PointLight) * max_lights, //  Весь размер буфера
			};

			VkWriteDescriptorSet ssbo_write{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = ssbo_descriptor_set, //  Дескриптор для SSBO
				.dstBinding = 0,			   //  Binding = 0 (как в макете)
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, //  Тип = Storage Buffer
				.pBufferInfo = &ssbo_buffer_info,
			};

			vkUpdateDescriptorSets(device, 1, &ssbo_write, 0, nullptr);
		}

		{
			VkDescriptorBufferInfo spot_buffer_info{
				.buffer = spot_lights_ssbo->buffer,
				.offset = 0,
				.range = sizeof(SpotLight) * max_lights,
			};

			VkWriteDescriptorSet spot_write{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = ssbo_descriptor_set,
				.dstBinding = 1,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &spot_buffer_info,
			};

			vkUpdateDescriptorSets(device, 1, &spot_write, 0, nullptr);
		}
		{
			VkDescriptorBufferInfo directional_buffer_info{
				.buffer = directional_lights_ssbo->buffer,
				.offset = 0,
				.range = sizeof(DirectionalLight) * max_lights,
			};

			VkWriteDescriptorSet directional_write{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = ssbo_descriptor_set,
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &directional_buffer_info,
			};

			vkUpdateDescriptorSets(device, 1, &directional_write, 0, nullptr);
		}
		printf("Before init  Now: %i\n", shadow_map_layout);
		// Переводим shadow map из UNDEFINED в READ_ONLY один раз при инициализации
		VkImageMemoryBarrier init_barrier = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = shadow_depth_image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};

		vkCmdPipelineBarrier(cmd, // ← cmd из параметра initialize()
							 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
							 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
							 0,
							 0, nullptr,
							 0, nullptr,
							 1, &init_barrier);
		shadow_map_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
		printf("After init  Now: %i\n", shadow_map_layout);
	}

	// NOTE: Destroy resources here, do not cause leaks in your program!
	void shutdown()
	{

		VkDevice &device = veekay::app.vk_device;

		vkDestroySampler(device, missing_texture_sampler, nullptr);
		delete missing_texture;

		delete cube_mesh.index_buffer;
		delete cube_mesh.vertex_buffer;

		delete plane_mesh.index_buffer;
		delete plane_mesh.vertex_buffer;

		delete model_uniforms_buffer;
		delete scene_uniforms_buffer;

		vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
		vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
		vkDestroyShaderModule(device, fragment_shader_module, nullptr);
		vkDestroyShaderModule(device, vertex_shader_module, nullptr);
		vkDestroySampler(device, texture_sampler, nullptr);

		delete directional_lights_ssbo;
		delete texture;

		// shadowPipeline
		vkDestroyPipelineLayout(device, shadow_pipeline_layout, nullptr);
		vkDestroyPipeline(device, shadow_pipeline, nullptr);
		vkDestroyShaderModule(device, shadow_vert_shader, nullptr);
		vkDestroySampler(device, shadow_sampler, nullptr);
		vkDestroyImageView(device, shadow_depth_view, nullptr);
		vkDestroyImage(device, shadow_depth_image, nullptr);
		vkFreeMemory(device, shadow_depth_memory, nullptr);

		vkDestroyFramebuffer(veekay::app.vk_device, shadow_framebuffer, nullptr);
		vkDestroyRenderPass(veekay::app.vk_device, shadow_render_pass, nullptr);
	}

	void update(double time)
	{

		static float rotation_angle = 0.0f;
		rotation_angle += 0.02f; // Медленное вращение

		// Заставим кубы вращаться
		for (size_t i = 1; i < models.size(); i++)
		{ // Начинаем с 1 чтобы пол не вращался
			models[i].transform.rotation.y = rotation_angle;
		}

		static int frame_count = 0;
		frame_count++;

		ImGui::Begin("Lighting Controls");

		// Информация о камере
		ImGui::Text("Camera pos: (%.2f, %.2f, %.2f)",
					camera.position.x, camera.position.y, camera.position.z);
		ImGui::Text("Camera rot: (%.2f, %.2f, %.2f)",
					camera.rotation.x, camera.rotation.y, camera.rotation.z);

		ImGui::Separator();
		ImGui::Text("Global Lighting");
		ImGui::ColorEdit3("Ambient Color", &ambientColor.x);
		ImGui::DragFloat("Ambient Intensity", &ambientIntensity, 0.01f, 0.0f, 5.0f);

		ImGui::Separator();
		ImGui::Text("Directional Lights (Count: %d):", (int)directional_lights.size());

		for (size_t i = 0; i < directional_lights.size(); i++)
		{
			ImGui::PushID(int(i) + 2000);
			if (ImGui::CollapsingHeader(("Directional Light " + std::to_string(i + 1)).c_str()))
			{
				ImGui::DragFloat3("Direction", &directional_lights[i].direction.x, 0.1f);
				ImGui::ColorEdit3("Color", &directional_lights[i].color.x);
				ImGui::DragFloat("Intensity", &directional_lights[i].intensity, 0.1f, 0.0f, 5.0f);
			}
			ImGui::PopID();
		}

		// Кнопки для направленных источников
		if (ImGui::Button("Add Directional Light") && directional_lights.size() < max_lights)
		{
			directional_lights.push_back(DirectionalLight{
				.direction = {0.0f, -1.0f, 0.0f}, // Свет сверху
				.color = {1.0f, 1.0f, 1.0f},
				.intensity = 0.8f});
		}

		ImGui::SameLine();
		if (ImGui::Button("Remove Directional Light") && !directional_lights.empty())
		{
			directional_lights.pop_back();
		}

		ImGui::Separator();
		ImGui::Text("Point Lights (Count: %d):", (int)point_lights.size());

		// Управление точечными источниками
		for (size_t i = 0; i < point_lights.size(); i++)
		{
			ImGui::PushID(int(i));
			if (ImGui::CollapsingHeader(("Point Light " + std::to_string(i + 1)).c_str()))
			{
				ImGui::DragFloat3("Position", &point_lights[i].position.x, 0.1f);
				ImGui::ColorEdit3("Color", &point_lights[i].color.x);
				ImGui::DragFloat("Intensity", &point_lights[i].intensity, 0.1f, 0.0f, 5.0f);
				// ... остальные параметры ...
			}
			ImGui::PopID();
		}

		ImGui::Separator();
		ImGui::Text("Spot Lights (Count: %d):", (int)spot_lights.size());

		// Управление прожекторами - ПЕРЕМЕСТИТЕ ЭТОТ БЛОК СЮДА
		for (size_t i = 0; i < spot_lights.size(); i++)
		{
			ImGui::PushID(int(i) + 1000); // Добавьте смещение для уникальности ID
			if (ImGui::CollapsingHeader(("Spot Light " + std::to_string(i + 1)).c_str()))
			{
				ImGui::DragFloat3("Position", &spot_lights[i].position.x, 0.1f);
				ImGui::DragFloat3("Direction", &spot_lights[i].direction.x, 0.1f);
				ImGui::ColorEdit3("Color", &spot_lights[i].color.x);
				ImGui::DragFloat("Intensity", &spot_lights[i].intensity, 0.1f, 0.0f, 5.0f);

				// Преобразуем обратно в градусы для удобства
				float cutOff_deg = acos(spot_lights[i].cutOff) * 180.0f / M_PI;
				float outerCutOff_deg = acos(spot_lights[i].outerCutOff) * 180.0f / M_PI;

				if (ImGui::DragFloat("CutOff (degrees)", &cutOff_deg, 1.0f, 0.0f, 90.0f))
					spot_lights[i].cutOff = float(cos(toRadians(cutOff_deg)));

				if (ImGui::DragFloat("OuterCutOff (degrees)", &outerCutOff_deg, 1.0f, 0.0f, 90.0f))
					spot_lights[i].outerCutOff = float(cos(toRadians(outerCutOff_deg)));
			}
			ImGui::PopID();
		}

		ImGui::Separator();

		// Кнопки добавления/удаления - ТЕПЕРЬ ПОСЛЕ ВСЕХ ИСТОЧНИКОВ
		if (ImGui::Button("Add Point Light") && point_lights.size() < max_lights)
		{
			point_lights.push_back(PointLight{
				.position = {0.0f, -2.0f, 0.0f},
				.color = {1.0f, 1.0f, 1.0f},
				.intensity = 1.0f,
				.constant = 1.0f,
				.linear = 0.09f,
				.quadratic = 0.032f});
		}

		ImGui::SameLine();
		if (ImGui::Button("Remove Point Light") && !point_lights.empty())
		{
			point_lights.pop_back();
		}

		if (ImGui::Button("Add Spot Light") && spot_lights.size() < max_lights)
		{
			if (spot_lights.size() < max_lights)
			{
				spot_lights.push_back(SpotLight{
					.position = {0.0f, -5.0f, 0.0f},
					.direction = {0.0f, -1.0f, 0.0f},
					.color = {1.0f, 1.0f, 0.0f},
					.intensity = 1.5f,
					.cutOff = float(cos(toRadians(30.0f))),
					.outerCutOff = float(cos(toRadians(45.0f))),
					.constant = 1.0f,
					.linear = 0.09f,
					.quadratic = 0.032f});
			}
			else
			{
				printf("FAILED: max lights reached\n");
			}
		}

		ImGui::SameLine();
		if (ImGui::Button("Remove Spot Light") && !spot_lights.empty())
		{
			spot_lights.pop_back();
		}

		ImGui::End();

		if (!ImGui::IsWindowHovered())
		{
			// ... управление камерой (ваш существующий код) ...
		}

		float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);

		// Рассчитываем матрицу для направленного источника света (ортографическая проекция)
		veekay::mat4 light_view_projection;

		// Если направление указывает вниз (y отрицательный), помещаем камеру ВЫШЕ
		if (!directional_lights.empty())
		{
			const auto &main_light = directional_lights[0];

			// 1. Нормализуем направление света
			veekay::vec3 light_dir = main_light.direction;
			float len = sqrt(light_dir.x * light_dir.x + light_dir.y * light_dir.y + light_dir.z * light_dir.z);
			if (len > 0.0001f)
			{
				light_dir.x /= len;
				light_dir.y /= len;
				light_dir.z /= len;
			}

			// 2. Позиция камеры: отступаем от цели против направления света
			float camera_distance = 10.0f;
			veekay::vec3 target = {0.0f, 0.0f, 0.0f}; // Всегда смотрим в центр сцены

			veekay::vec3 camera_pos = {
				target.x - light_dir.x * camera_distance,
				target.y - light_dir.y * camera_distance,
				target.z - light_dir.z * camera_distance};

			// 3. Выбираем вектор "вверх", который не параллелен направлению взгляда
			// Направление взгляда камеры = из camera_pos в target = light_dir
			veekay::vec3 forward = light_dir; // Камера смотрит в том же направлении что и свет

			veekay::vec3 up;

			// Проверяем, не параллелен ли forward оси Y (основной кандидат на "верх")
			if (fabs(forward.y) < 0.9f)
			{
				// forward не вертикальный - можно использовать ось Y как "верх"
				up = {0.0f, 1.0f, 0.0f};
			}
			else
			{
				// forward почти вертикальный - используем ось Z как "верх"
				up = {0.0f, 0.0f, 1.0f};
			}
			float size = 50.0f;

			auto light_view = lookAt(camera_pos, target, up);
			auto light_projection = orthographic(-size, size, -size, size, 1.0f, 50.0f);

			light_view_projection = light_view * light_projection;
			curr_light_view_projection = light_view_projection;

			/* printMatrix("LookAt", light_view);
			printMatrix("orthographic", light_projection); */
			//printMatrix("curr_light_view_projection Updata", curr_light_view_projection);

		}
		else
		{
			// Если нет направленных источников, используем дефолтную матрицу
			// Если нет направленных источников, ВСЕ РАВНО создаем камеру сверху!
			veekay::vec3 camera_pos = {0.0f, 30.0f, 0.0f};
			veekay::vec3 target = {0.0f, 0.0f, 0.0f};
			veekay::vec3 up = {0.0f, 0.0f, 1.0f};
			float size = 50.0f;

			auto light_view = lookAt(camera_pos, target, up);
			// printMatrix("light_view", light_view);
			auto light_projection = orthographic(-size, size, -size, size, 1.0f, 100.0f);
			// printMatrix("light_projection", light_projection);

			light_view_projection = light_view * light_projection;

			// printMatrix("light_view_projection", light_view_projection);
		}

		SceneUniforms scene_uniforms{
			.view_projection = camera.view_projection(aspect_ratio),
			.light_view_projection = light_view_projection,
			.view_position = camera.position,
			.point_light_count = uint32_t(point_lights.size()),
			.spot_light_count = uint32_t(spot_lights.size()),
			.directional_light_count = uint32_t(directional_lights.size()),
			.ambientColor = ambientColor,
			.ambientIntensity = ambientIntensity};

		/* SceneUniforms scene_uniforms{
			.view_projection = camera.view_projection(aspect_ratio),
			.view_position = camera.position,
			.point_light_count = uint32_t(point_lights.size()),
			.spot_light_count = uint32_t(spot_lights.size()),
			.directional_light_count = uint32_t(directional_lights.size()),
			.ambientColor = ambientColor,
			.ambientIntensity = ambientIntensity}; */

		if (point_lights_ssbo && point_lights_ssbo->mapped_region)
		{
			// Копируем обновленные данные источников света в SSBO
			memcpy(point_lights_ssbo->mapped_region, point_lights.data(),
				   sizeof(PointLight) * point_lights.size());
		}

		if (spot_lights_ssbo && spot_lights_ssbo->mapped_region)
		{
			memcpy(spot_lights_ssbo->mapped_region, spot_lights.data(),
				   sizeof(SpotLight) * spot_lights.size());
		}

		if (directional_lights_ssbo && directional_lights_ssbo->mapped_region)
		{
			memcpy(directional_lights_ssbo->mapped_region, directional_lights.data(),
				   sizeof(DirectionalLight) * directional_lights.size());
		}

		ImGui::Begin("Debug Info");

		ImGui::Text("Frame: %d", frame_count);
		ImGui::Text("Camera pos: (%.2f, %.2f, %.2f)",
					camera.position.x, camera.position.y, camera.position.z);
		ImGui::Text("Camera rot: (%.2f, %.2f, %.2f)",
					camera.rotation.x, camera.rotation.y, camera.rotation.z);

		// Простая проверка ввода
		using namespace veekay::input;
		bool w_pressed = keyboard::isKeyDown(keyboard::Key::w);
		bool mouse_pressed = mouse::isButtonDown(mouse::Button::left);
		ImGui::Text("W pressed: %s", w_pressed ? "YES" : "NO");
		ImGui::Text("Mouse pressed: %s", mouse_pressed ? "YES" : "NO");

		ImGui::End();

		if (!ImGui::IsWindowHovered())
		{
			using namespace veekay::input;

			if (mouse::isButtonDown(mouse::Button::left))
			{
				auto move_delta = mouse::cursorDelta();

				// TODO: Use mouse_delta to update camera rotation
				const float sensitivity = 0.01f;
				camera.rotation.y += move_delta.x * sensitivity; // yaw
				camera.rotation.x += move_delta.y * sensitivity; // pitch

				// Ограничиваем pitch чтобы не переворачивать камеру
				if (camera.rotation.x > 1.57f)
					camera.rotation.x = 1.57f;
				if (camera.rotation.x < -1.57f)
					camera.rotation.x = -1.57f;
			}
			auto view = camera.view();

			// Вычисляем векторы направления из матрицы вида
			veekay::vec3 right = {view[0][0], view[1][0], view[2][0]};
			veekay::vec3 up = {-view[0][1], -view[1][1], -view[2][1]};
			veekay::vec3 front = {view[0][2], view[1][2], view[2][2]};

			float camera_speed = 0.1f;
			if (keyboard::isKeyDown(keyboard::Key::w))
				camera.position += front * camera_speed;

			if (keyboard::isKeyDown(keyboard::Key::s))
				camera.position -= front * camera_speed;

			if (keyboard::isKeyDown(keyboard::Key::d))
				camera.position += right * camera_speed;

			if (keyboard::isKeyDown(keyboard::Key::a))
				camera.position -= right * camera_speed;

			if (keyboard::isKeyDown(keyboard::Key::q))
				camera.position += up * camera_speed;

			if (keyboard::isKeyDown(keyboard::Key::z))
				camera.position -= up * camera_speed;
		}

		std::vector<ModelUniforms> model_uniforms(models.size());
		for (size_t i = 0, n = models.size(); i < n; ++i)
		{
			const Model &model = models[i];
			ModelUniforms &uniforms = model_uniforms[i];

			uniforms.model = model.transform.matrix();
			uniforms.material = model.material;

			uniforms.normal_matrix = uniforms.model;
		}

		*(SceneUniforms *)scene_uniforms_buffer->mapped_region = scene_uniforms;

		const size_t alignment =
			veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

		for (size_t i = 0, n = model_uniforms.size(); i < n; ++i)
		{
			const ModelUniforms &uniforms = model_uniforms[i];

			char *const pointer = static_cast<char *>(model_uniforms_buffer->mapped_region) + i * alignment;
			*reinterpret_cast<ModelUniforms *>(pointer) = uniforms;
		}
	}

	void render(VkCommandBuffer cmd, VkFramebuffer framebuffer)
	{

		vkResetCommandBuffer(cmd, 0);

		{ // NOTE: Start recording rendering commands
			VkCommandBufferBeginInfo info{
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
			};
			vkBeginCommandBuffer(cmd, &info);
		}

		barier_to_Write(shadow_map_layout, cmd);
		{
			// barier_to_Write(shadow_map_layout, cmd);

			// Dynamic rendering для shadow pass

			VkClearValue clear_value = {.depthStencil = {1.0f, 0}};

			VkRenderPassBeginInfo render_pass_info = {
				.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
				.renderPass = shadow_render_pass,
				.framebuffer = shadow_framebuffer,
				.renderArea = {{0, 0}, {1024, 1024}},
				.clearValueCount = 1,
				.pClearValues = &clear_value,
			};

			vkCmdBeginRenderPass(cmd, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

			// Устанавливаем viewport для shadow pass
			VkViewport viewport{
				.x = 0.0f,
				.y = 0.0f,
				.width = 1024.0f,
				.height = 1024.0f,
				.minDepth = 0.0f,
				.maxDepth = 1.0f,
			};

			VkRect2D scissor{
				.offset = {0, 0},
				.extent = {1024, 1024},
			};

			vkCmdSetViewport(cmd, 0, 1, &viewport);
			vkCmdSetScissor(cmd, 0, 1, &scissor);

			// Рендерим сцены с точки зрения источника света
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);

			VkDeviceSize zero_offset = 0;
			VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
			VkBuffer current_index_buffer = VK_NULL_HANDLE;

			const size_t model_uniforms_alignment =
				veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

			// Нужно как-то передать эту матрицу из update() в render()
			// Пока используем identity для теста
			for (size_t i = 0, n = models.size(); i < n; ++i)
			{

				const Model &model = models[i];
				const Mesh &mesh = model.mesh;

				if (current_vertex_buffer != mesh.vertex_buffer->buffer)
				{
					current_vertex_buffer = mesh.vertex_buffer->buffer;
					vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
				}

				if (current_index_buffer != mesh.index_buffer->buffer)
				{
					current_index_buffer = mesh.index_buffer->buffer;
					vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
				}

				// Push constants: модель + матрица света
				struct
				{
					veekay::mat4 model;
					veekay::mat4 light_view_projection;
				} push_constants;

				veekay::mat4 light_view_projection1 = {
					1.0f, 0.0f, 0.0f, 0.0f,
					0.0f, 1.0f, 0.0f, 0.0f,
					0.0f, 0.0f, 1.0f, 0.0f,
					0.0f, 0.0f, 0.0f, 1.0f};

				push_constants.model = model.transform.matrix();
				push_constants.light_view_projection = curr_light_view_projection;
				// push_constants.light_view_projection = light_view_projection1;
					//printMatrix("push_constants.light_view_projection Updata", push_constants.light_view_projection);
				if (i == 0)
				{ // Только для первого объекта
					std::cout << "[DEBUG] Light view projection matrix (first row): ";
					printMatrix("Matr:", push_constants.light_view_projection);
				}

				vkCmdPushConstants(cmd, shadow_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
								   0, sizeof(push_constants), &push_constants);

				// Рисуем
				vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
			}
			vkCmdEndRenderPass(cmd);

			barier_to_Read(shadow_map_layout, cmd);

			// === 2. Основной рендер ===
			{
				VkViewport viewport{
					.x = 0.0f,
					.y = 0.0f,
					.width = static_cast<float>(veekay::app.window_width),
					.height = static_cast<float>(veekay::app.window_height),
					.minDepth = 0.0f,
					.maxDepth = 1.0f,
				};

				VkRect2D scissor{
					.offset = {0, 0},
					.extent = {veekay::app.window_width, veekay::app.window_height},
				};

				vkCmdSetViewport(cmd, 0, 1, &viewport);
				vkCmdSetScissor(cmd, 0, 1, &scissor);
			}
		}

		{
			VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
			VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

			VkClearValue clear_values[] = {clear_color, clear_depth};

			VkRenderPassBeginInfo info{
				.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
				.renderPass = veekay::app.vk_render_pass,
				.framebuffer = framebuffer,
				.renderArea = {
					.extent = {
						veekay::app.window_width,
						veekay::app.window_height},
				},
				.clearValueCount = 2,
				.pClearValues = clear_values,
			};

			vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
		}

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		VkDeviceSize zero_offset = 0;

		VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
		VkBuffer current_index_buffer = VK_NULL_HANDLE;

		const size_t model_uniorms_alignment =
			veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

		for (size_t i = 0, n = models.size(); i < n; ++i)
		{
			const Model &model = models[i];
			const Mesh &mesh = model.mesh;

			if (current_vertex_buffer != mesh.vertex_buffer->buffer)
			{
				current_vertex_buffer = mesh.vertex_buffer->buffer;
				vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
			}

			if (current_index_buffer != mesh.index_buffer->buffer)
			{
				current_index_buffer = mesh.index_buffer->buffer;
				vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
			}

			uint32_t offset = i * model_uniorms_alignment;

			VkDescriptorSet descriptor_sets[] = {
				descriptor_set,		//  Первый: UBO (камера, материалы)
				ssbo_descriptor_set //  Второй: SSBO (источники света)
			};

			// Привязываем ОБА дескрипторных сета
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
									0, 2, descriptor_sets, 1, &offset);

			vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
		}

		vkCmdEndRenderPass(cmd);
		vkEndCommandBuffer(cmd);
	}

} // namespace

int main()
{
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}
