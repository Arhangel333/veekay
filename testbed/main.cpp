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

namespace
{

	constexpr uint32_t max_models = 1024;

	struct Vertex
	{
		veekay::vec3 position;
		veekay::vec3 normal;
		veekay::vec2 uv;
		// NOTE: You can add more attributes
	};

	struct Material {
    veekay::vec3 albedo;     // 12 bytes
    float _pad0;             // 👈 ДОБАВЬ ПАДДИНГ до 16 bytes
    veekay::vec3 specular;   // 12 bytes  
    float shininess;         // 4 bytes
    // Итого: 32 bytes (совпадает с GLSL)
};

	struct SceneUniforms
	{
		veekay::mat4 view_projection;
		veekay::vec3 view_position;
		float _pad0;
		uint32_t point_light_count;
		float _pad1[3];
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
		veekay::vec3 position;
		veekay::vec3 color;
		float intensity;
		float constant;
		float linear;
		float quadratic;
	};

	// NOTE: Scene objects
	inline namespace
	{
		Camera camera{
			.position = {0.0f, -0.5f, -3.0f} // {0.0f, -0.5f, -3.0f}
		};

		std::vector<Model> models;

		// SSBO для точечных источников света
		veekay::graphics::Buffer *point_lights_ssbo;
		std::vector<PointLight> point_lights;

		// Дескрипторы для SSBO
		VkDescriptorSetLayout ssbo_descriptor_set_layout;
		VkDescriptorSet ssbo_descriptor_set;
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
	}

	float toRadians(float degrees)
	{
		return degrees * float(M_PI) / 180.0f;
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
		printf("shader %s size: %d\n", path, (int)size);
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

	void initialize(VkCommandBuffer cmd)
	{
		VkDevice &device = veekay::app.vk_device;
		VkPhysicalDevice &physical_device = veekay::app.vk_physical_device;

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

			// NOTE: How many bytes does a vertex take?
			VkVertexInputBindingDescription buffer_binding{
				.binding = 0,
				.stride = sizeof(Vertex),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
			};

			// NOTE: Declare vertex attributes
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

			// NOTE: Describe inputs
			VkPipelineVertexInputStateCreateInfo input_state_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
				.vertexBindingDescriptionCount = 1,
				.pVertexBindingDescriptions = &buffer_binding,
				.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
				.pVertexAttributeDescriptions = attributes,
			};

			// NOTE: Every three vertices make up a triangle,
			//       so our vertex buffer contains a "list of triangles"
			VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
				.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			};

			// NOTE: Declare clockwise triangle order as front-facing
			//       Discard triangles that are facing away
			//       Fill triangles, don't draw lines instaed
			VkPipelineRasterizationStateCreateInfo raster_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
				.polygonMode = VK_POLYGON_MODE_FILL,
				.cullMode = VK_CULL_MODE_BACK_BIT,
				.frontFace = VK_FRONT_FACE_CLOCKWISE,
				.lineWidth = 1.0f,
			};

			// NOTE: Use 1 sample per pixel
			VkPipelineMultisampleStateCreateInfo sample_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
				.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
				.sampleShadingEnable = false,
				.minSampleShading = 1.0f,
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
						.descriptorCount = 8,
					},
					{
						.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, //  ДЛЯ SSBO!
						.descriptorCount = 2,					   //  2 слота для SSBO
					}};

				VkDescriptorPoolCreateInfo info{
					.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
					.maxSets = 2, //  УВЕЛИЧИВАЕМ до 2 (UBO + SSBO)
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
			/* VkPipelineLayoutCreateInfo layout_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = 1,
				.pSetLayouts = &descriptor_set_layout,
			}; */

			// 1. Создаем макет дескриптора для SSBO
			{
				VkDescriptorSetLayoutBinding ssbo_binding{
					.binding = 0, // binding = 0 для SSBO
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, // Используем во фрагментном шейдере
				};

				VkDescriptorSetLayoutCreateInfo ssbo_layout_info{
					.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
					.bindingCount = 1,
					.pBindings = &ssbo_binding,
				};

				if (vkCreateDescriptorSetLayout(device, &ssbo_layout_info, nullptr,
												&ssbo_descriptor_set_layout) != VK_SUCCESS)
				{
					std::cerr << "Failed to create Vulkan SSBO descriptor set layout\n";
					veekay::app.running = false;
					return;
				}
			}

			// 👇 СОЗДАЕМ МАССИВ макетов дескрипторов
			VkDescriptorSetLayout descriptor_set_layouts[] = {
				descriptor_set_layout,	   //  Первый: для UBO (камера, материалы)
				ssbo_descriptor_set_layout //  Второй: для SSBO (источники света)
			};

			VkPipelineLayoutCreateInfo layout_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = 2,				   //  Теперь 2 макета!
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
			//.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f}
			.material = Material{
				.albedo = {0.0f, 0.0f, 0.0f},
				.specular = {0.5f, 0.5f, 0.5f},
				.shininess = 60.0f}});

		models.emplace_back(Model{
			.mesh = cube_mesh,
			.transform = Transform{
				.position = {-2.0f, -0.5f, -1.5f},
			},
			.material = Material{.albedo = {1.0f, 0.0f, 0.0f}, .specular = {0.7f, 0.7f, 0.7f}, .shininess = 1.0f}});

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

		// 1. Создаем несколько источников света
		point_lights.push_back(PointLight{
			.position = {0.0f, 2.0f, 0.0f}, //  Свет сверху
			.color = {1.0f, 1.0f, 1.0f},	//  Белый свет
			.intensity = 1.0f,
			.constant = 1.0f,
			.linear = 0.09f,
			.quadratic = 0.032f});

		point_lights.push_back(PointLight{
			.position = {2.0f, 1.0f, 2.0f}, //  Свет справа-спереди
			.color = {1.0f, 0.0f, 0.0f},	//  Красный свет
			.intensity = 0.8f,
			.constant = 1.0f,
			.linear = 0.09f,
			.quadratic = 0.032f});

		point_lights.push_back(PointLight{
			.position = {-2.0f, 1.0f, -1.0f}, //  Свет слева-сзади
			.color = {0.0f, 0.0f, 1.0f},	  //  Синий свет
			.intensity = 0.8f,
			.constant = 1.0f,
			.linear = 0.09f,
			.quadratic = 0.032f});

		// 2. Создаем SSBO буфер и заполняем его источниками света
		point_lights_ssbo = new veekay::graphics::Buffer(
			sizeof(PointLight) * point_lights.size(), //  Размер = размер одного источника × количество
			point_lights.data(),					  //  Данные = массив источников света
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT		  //  Тип = Storage Buffer
		);

		// 3. Связываем SSBO буфер с дескриптором
		{
			VkDescriptorBufferInfo ssbo_buffer_info{
				.buffer = point_lights_ssbo->buffer, //  Наш SSBO буфер
				.offset = 0,
				.range = sizeof(PointLight) * point_lights.size(), //  Весь размер буфера
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

		// 👇 ДОБАВЛЯЕМ НОВЫЙ UI ДЛЯ УПРАВЛЕНИЯ СВЕТОМ:
		ImGui::Begin("Lighting Controls");

		// Информация о камере
		ImGui::Text("Camera pos: (%.2f, %.2f, %.2f)",
					camera.position.x, camera.position.y, camera.position.z);
		ImGui::Text("Camera rot: (%.2f, %.2f, %.2f)",
					camera.rotation.x, camera.rotation.y, camera.rotation.z);

		ImGui::Separator();
		ImGui::Text("Point Lights (Count: %d):", (int)point_lights.size());

		// Управление каждым источником света
		for (size_t i = 0; i < point_lights.size(); i++)
		{
			ImGui::PushID(int(i));

			if (ImGui::CollapsingHeader(("Light " + std::to_string(i + 1)).c_str()))
			{
				ImGui::DragFloat3("Position", &point_lights[i].position.x, 0.1f);
				ImGui::ColorEdit3("Color", &point_lights[i].color.x);
				ImGui::DragFloat("Intensity", &point_lights[i].intensity, 0.1f, 0.0f, 5.0f);

				ImGui::Text("Attenuation:");
				ImGui::DragFloat("Constant", &point_lights[i].constant, 0.01f, 0.1f, 10.0f);
				ImGui::DragFloat("Linear", &point_lights[i].linear, 0.001f, 0.0f, 1.0f);
				ImGui::DragFloat("Quadratic", &point_lights[i].quadratic, 0.001f, 0.0f, 1.0f);
			}

			ImGui::PopID();
		}

		ImGui::Separator();

		// Кнопка для добавления нового источника света
		if (ImGui::Button("Add New Light"))
		{
			point_lights.push_back(PointLight{
				.position = {0.0f, 2.0f, 0.0f},
				.color = {1.0f, 1.0f, 1.0f},
				.intensity = 1.0f,
				.constant = 1.0f,
				.linear = 0.09f,
				.quadratic = 0.032f});
		}

		// Кнопка для удаления последнего источника света
		if (ImGui::Button("Remove Last Light") && !point_lights.empty())
		{
			point_lights.pop_back();
		}

		ImGui::End();

		// 👇 ОСТАЛЬНАЯ ЧАСТЬ ФУНКЦИИ update() БЕЗ ИЗМЕНЕНИЙ:
		if (!ImGui::IsWindowHovered())
		{
			// ... управление камерой (ваш существующий код) ...
		}

		// 👇 ОБНОВЛЯЕМ ДАННЫЕ СЦЕНЫ С УЧЕТОМ ИСТОЧНИКОВ СВЕТА:
		float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
		SceneUniforms scene_uniforms{
			.view_projection = camera.view_projection(aspect_ratio),
			.view_position = camera.position,
			.point_light_count = uint32_t(point_lights.size()) //  ПЕРЕДАЕМ актуальное количество!
		};

		// 👇 ОБНОВЛЯЕМ SSBO С ИСТОЧНИКАМИ СВЕТА:
		if (point_lights_ssbo)
		{
			// Копируем обновленные данные источников света в SSBO
			memcpy(point_lights_ssbo->mapped_region, point_lights.data(),
				   sizeof(PointLight) * point_lights.size());
		}
		/* static int frame_count = 0;
		frame_count++; */

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

			// TODO: Calculate right, up and front from view matrix
			/* veekay::vec3 right = {1.0f, 0.0f, 0.0f};
			veekay::vec3 up = {0.0f, -1.0f, 0.0f};*/
			// veekay::vec3 front = {0.0f, 0.0f, 1.0f};

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

		{ // NOTE: Use current swapchain framebuffer and clear it
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
			/* vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
									0, 1, &descriptor_set, 1, &offset);
 */
			// 👇 СОЗДАЕМ МАССИВ дескрипторных сетов
			VkDescriptorSet descriptor_sets[] = {
				descriptor_set,		//  Первый: UBO (камера, материалы)
				ssbo_descriptor_set //  Второй: SSBO (источники света)
			};

			// Привязываем ОБА дескрипторных сета
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
									0, 2, descriptor_sets, 1, &offset); //  setCount = 2!

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
