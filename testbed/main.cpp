
#include "objects.hpp"

namespace
{
	// NOTE: Loads shader byte code from file
	// NOTE: Your shaders are compiled via CMake with this code too, look it up
	VkShaderModule loadShaderModule(const char *path)
	{
		std::ifstream file(path, std::ios::binary | std::ios::ate);
		size_t size = file.tellg();
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

	VulkanBuffer createBuffer(size_t size, void *data, VkBufferUsageFlags usage)
	{
		VkDevice &device = veekay::app.vk_device;
		VkPhysicalDevice &physical_device = veekay::app.vk_physical_device;

		VulkanBuffer result{};

		{
			// NOTE: We create a buffer of specific usage with specified size
			VkBufferCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
				.size = size,
				.usage = usage,
				.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			};

			if (vkCreateBuffer(device, &info, nullptr, &result.buffer) != VK_SUCCESS)
			{
				std::cerr << "Failed to create Vulkan buffer\n";
				return {};
			}
		}

		// NOTE: Creating a buffer does not allocate memory,
		//       only a buffer **object** was created.
		//       So, we allocate memory for the buffer

		{
			// NOTE: Ask buffer about its memory requirements
			VkMemoryRequirements requirements;
			vkGetBufferMemoryRequirements(device, result.buffer, &requirements);

			// NOTE: Ask GPU about types of memory it supports
			VkPhysicalDeviceMemoryProperties properties;
			vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);

			// NOTE: We want type of memory which is visible to both CPU and GPU
			// NOTE: HOST is CPU, DEVICE is GPU; we are interested in "CPU" visible memory
			// NOTE: COHERENT means that CPU cache will be invalidated upon mapping memory region
			const VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
												VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

			// NOTE: Linear search through types of memory until
			//       one type matches the requirements, thats the index of memory type
			uint32_t index = UINT_MAX;
			for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
			{
				const VkMemoryType &type = properties.memoryTypes[i];

				if ((requirements.memoryTypeBits & (1 << i)) &&
					(type.propertyFlags & flags) == flags)
				{
					index = i;
					break;
				}
			}

			if (index == UINT_MAX)
			{
				std::cerr << "Failed to find required memory type to allocate Vulkan buffer\n";
				return {};
			}

			// NOTE: Allocate required memory amount in appropriate memory type
			VkMemoryAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
				.allocationSize = requirements.size,
				.memoryTypeIndex = index,
			};

			if (vkAllocateMemory(device, &info, nullptr, &result.memory) != VK_SUCCESS)
			{
				std::cerr << "Failed to allocate Vulkan buffer memory\n";
				return {};
			}

			// NOTE: Link allocated memory with a buffer
			if (vkBindBufferMemory(device, result.buffer, result.memory, 0) != VK_SUCCESS)
			{
				std::cerr << "Failed to bind Vulkan  buffer memory\n";
				return {};
			}

			// NOTE: Get pointer to allocated memory
			void *device_data;
			vkMapMemory(device, result.memory, 0, requirements.size, 0, &device_data);

			memcpy(device_data, data, size);

			vkUnmapMemory(device, result.memory);
		}

		return result;
	}

	void destroyBuffer(const VulkanBuffer &buffer)
	{
		VkDevice &device = veekay::app.vk_device;

		vkFreeMemory(device, buffer.memory, nullptr);
		vkDestroyBuffer(device, buffer.buffer, nullptr);
	}

	void initialize()
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
			// NOTE: If you want more attributes per vertex, declare them here
#if 0
			{
				.location = 1, // NOTE: Second attribute
				.binding = 0,
				.format = VK_FORMAT_XXX,
				.offset = offset(Vertex, your_attribute),
			},
#endif
			};

			// NOTE: Bring
			VkPipelineVertexInputStateCreateInfo input_state_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
				.vertexBindingDescriptionCount = 1,
				.pVertexBindingDescriptions = &buffer_binding,
				.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
				.pVertexAttributeDescriptions = attributes,
			};

			// NOTE: Every three cylindervertices make up a triangle,
			//       so our vertex buffer contains a "list of triangles"
			VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
				.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, // VK_PRIMITIVE_TOPOLOGY_LINE_LIST VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
			};

			// NOTE: Declare clockwise triangle order as front-facing
			//       Discard triangles that are facing away
			//       Fill triangles, don't draw lines instaed
			VkPipelineRasterizationStateCreateInfo raster_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
				.polygonMode = VK_POLYGON_MODE_FILL,
				.cullMode = VK_CULL_MODE_BACK_BIT, /* VK_CULL_MODE_NONE, VK_CULL_MODE_BACK_BIT */
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

			// NOTE: Declare constant memory region visible to vertex and fragment shaders
			VkPushConstantRange push_constants{
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
							  VK_SHADER_STAGE_FRAGMENT_BIT,
				.size = sizeof(ShaderConstants),
			};

			// NOTE: Declare external data sources, only push constants this time
			VkPipelineLayoutCreateInfo layout_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.pushConstantRangeCount = 1,
				.pPushConstantRanges = &push_constants,
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

		// TODO: You define model cylindervertices and create buffers here
		// TODO: Index buffer has to be created here too
		// NOTE: Look for createBuffer function

		// (v0)------(v1)
		//  |  \       |
		//  |   `--,   |
		//  |       \  |
		// (v3)------(v2)

		/* Vertex cylindervertices[] = {
			{{-1.0f, -1.0f, 0.0f}},
			{{1.0f, -1.0f, 0.0f}},
			{{1.0f, 1.0f, 0.0f}},
			{{-1.0f, 1.0f, 0.0f}},
		};

		uint32_t cylinderindices[] = { 0, 1, 2, 2, 3, 0 };
		 */

		// cylinder
		int cylindersegments = 30;
		Vertex cylindervertices[(cylindersegments + 1) * 2];
		uint32_t cylinderindices[cylindersegments * 6];

		generateCylinderVertices(cylindervertices, cylindersegments, 0.5f, 2.0f);
		generateCylinderIndices(cylinderindices, cylindersegments);

		if (sizeof(cylinderindices))
			cylinder_indices_count = sizeof(cylinderindices) / sizeof(cylinderindices[0]);

		vertex_buffer = createBuffer(sizeof(cylindervertices), cylindervertices,
									 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		index_buffer = createBuffer(sizeof(cylinderindices), cylinderindices,
									VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		// cube
		int cubesegments = 4;
		Vertex cubevertices[(cubesegments + 1) * 2];
		uint32_t cubeindices[cubesegments * 6 + (cubesegments - 2) * 2 * 3];

		generateCylinderVertices(cubevertices, cubesegments, 0.25f, 0.5f);
		generateCubeIndices(cubeindices, cubesegments);
		if (sizeof(cubeindices))
			cube_indices_count = sizeof(cubeindices) / sizeof(cubeindices[0]);

		cube_vertex_buffer = createBuffer(sizeof(cubevertices), cubevertices,
										  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		cube_index_buffer = createBuffer(sizeof(cubeindices), cubeindices,
										 VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		size_t vertex_size = (sizeof(cylindervertices) + sizeof(cubevertices)) / sizeof(cylindervertices[0]);
		Vertex allvertexes[vertex_size];
		for (size_t i = 0, n = 0; i < vertex_size; i++)
		{
			if (i < sizeof(cylindervertices) / sizeof(cylindervertices[0]))
			{
				allvertexes[i] = cylindervertices[i];
				n++;
			}
			else
			{
				allvertexes[i] = cubevertices[i - n];
			}
		}

		// together objects
		size_t cylinder_idx_size = sizeof(cylinderindices) / sizeof(cylinderindices[0]);
		size_t cube_idx_size = sizeof(cubeindices) / sizeof(cubeindices[0]);
		size_t index_size = cylinder_idx_size + cube_idx_size;

		uint32_t result_idx[index_size];

		for (size_t i = 0, n = 0; i < index_size; i++)
		{
			if (i < cylinder_idx_size)
			{
				result_idx[i] = cylinderindices[i];
				n++;
			}
			else
			{
				result_idx[i] = cubeindices[i - n] + sizeof(cylindervertices) / sizeof(cylindervertices[0]);
			}
		}

		/* all_vertex_buffer = createBuffer(sizeof(allvertexes), allvertexes,
									 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		all_index_buffer = createBuffer(sizeof(result_idx), result_idx,
									VK_BUFFER_USAGE_INDEX_BUFFER_BIT); */
	}

	void shutdown()
	{
		VkDevice &device = veekay::app.vk_device;

		// NOTE: Destroy resources here, do not cause leaks in your program!
		destroyBuffer(index_buffer);
		destroyBuffer(vertex_buffer);
		destroyBuffer(cube_index_buffer);
		destroyBuffer(cube_vertex_buffer);

		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
		vkDestroyShaderModule(device, fragment_shader_module, nullptr);
		vkDestroyShaderModule(device, vertex_shader_module, nullptr);
	}

	void update(double time)
	{
		ImGui::Begin("Controls:");
		ImGui::ColorEdit3("Model Color", reinterpret_cast<float *>(&model_color));
		ImGui::InputFloat3("Translation", reinterpret_cast<float *>(&cilinder.model_position));
		ImGui::SliderFloat("Cilinder Trajectory Radius", &trajectoryRadius, 0.1f, 5.0f);
		ImGui::SliderFloat("Cilinder Animation Speed", &animationSpeed, 0.1f, 3.0f);
		ImGui::SliderFloat("Rotation", &cilinder.model_rotation, 0.0f, 2.0f * M_PI);
		ImGui::Checkbox("Spin", &cilinder.model_spin);
		ImGui::Checkbox("Ort View", &ortografics);
		ImGui::Checkbox("Tracing", &obj_rotation);
		ImGui::Checkbox("Add satellite", &satellite);
		if (satellite)
		{
			ImGui::SliderFloat("Satellite Trajectory Radius", &cube_trajectoryRadius, 0.1f, 5.0f);
			ImGui::SliderFloat("Satellite Animation Speed", &cube_animationSpeed, 0.0f, 3.0f);
			ImGui::Checkbox("Satellite Spin", &cube.model_spin);
		}
		
		
		// TODO: Your GUI stuff here
		ImGui::End();

		// NOTE: Animation code and other runtime variable updates go here
		if (cilinder.model_spin)
		{
			cilinder.model_rotation = float(time);
		}

		if (obj_rotation)
		{
			float angle = newangle * 0.1f;
			newangle += animationSpeed;
			// angle = fmodf(angle, 2.0f * M_PI);
			float center_x = 0.0f; // центр окружности по X
			float center_z = 5.0f; // центр окружности по Z

			cilinder.model_position.x = center_x + trajectoryRadius * cosf(angle);
			cilinder.model_position.z = center_z + trajectoryRadius * sinf(angle);
		}
		if (satellite)
		{
			float angle = cube_newangle * 0.1f;
			cube_newangle += cube_animationSpeed;
			float center_x = cilinder.model_position.x; // центр окружности по X
			float center_z = cilinder.model_position.z; // центр окружности по Z

			cube.model_position.x = center_x + cube_trajectoryRadius * cosf(angle);
			cube.model_position.z = center_z + cube_trajectoryRadius * sinf(angle);
			if (cube.model_spin)
		{
			cube.model_rotation = float(time);
		}
		
		}else{
			cube.model_position = {4.0f, 0.0f, 0.0f};
		}

		cilinder.model_rotation = fmodf(cilinder.model_rotation, 2.0f * M_PI);
		// model_pos = fmodf(model_pos, 2.0f * M_PI);
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

		// TODO: Vulkan rendering code here
		// NOTE: ShaderConstant updates, vkCmdXXX expected to be here
		// NOTE: Use our new shiny graphics pipeline
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		{
			

			// NOTE: Use our quad vertex buffer
			VkDeviceSize offset = 0;
			vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &offset);

			// NOTE: Use our quad index buffer
			vkCmdBindIndexBuffer(cmd, index_buffer.buffer, offset, VK_INDEX_TYPE_UINT32);

			// NOTE: Variables like model_XXX were declared globally
			ShaderConstants constants;
			Matrix transform = multiply(

				rotation({0.0f, 1.0f, 0.0f}, cilinder.model_rotation), // вращение вокруг Y
				translation(cilinder.model_position)		  // перемещение
			);

			constants = ShaderConstants{
				.projection = {},

				.transform = transform,
				.color = model_color,
			};
			if (!ortografics)
			{ // Перспективная проекция
				constants.projection = projection(camera_fov, float(veekay::app.window_width) / float(veekay::app.window_height), camera_near_plane, camera_far_plane);
			}
			else
			{ // Ортографическая проекция
				constants.projection = orthographicProjection(float(veekay::app.window_width) / float(veekay::app.window_height));
			}

			// NOTE: Update constant memory with new shader constants
			vkCmdPushConstants(cmd, pipeline_layout,
							   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
							   0, sizeof(ShaderConstants), &constants);

			// NOTE: Draw 6 cylinderindices (3 cylindervertices * 2 triangles), 1 group, no offsets
			vkCmdDrawIndexed(cmd, cylinder_indices_count, 1, 0, 0, 0);
		}
		{
			VkDeviceSize offset = 0;

			vkCmdBindVertexBuffers(cmd, 0, 1, &cube_vertex_buffer.buffer, &offset);
			vkCmdBindIndexBuffer(cmd, cube_index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

			ShaderConstants constants_cube;
			Matrix transform = multiply(

				rotation({1.0f, 0.0f, 0.0f}, cube.model_rotation), // вращение вокруг Y
				translation(cube.model_position)		  // перемещение
			); // Примерная позиция для куба

			constants_cube = ShaderConstants{
				.projection = {},
				.transform = transform,
				.color = {1.0f, 0.0f, 0.0f}, // Красный цвет для куба
			};

			if (!ortografics)
			{ // Перспективная проекция
				constants_cube.projection = projection(camera_fov, float(veekay::app.window_width) / float(veekay::app.window_height), camera_near_plane, camera_far_plane);
			}
			else
			{ // Ортографическая проекция
				constants_cube.projection = orthographicProjection(float(veekay::app.window_width) / float(veekay::app.window_height));
			}

			vkCmdPushConstants(cmd, pipeline_layout,
							   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
							   0, sizeof(ShaderConstants), &constants_cube);

			vkCmdDrawIndexed(cmd, cube_indices_count, 1, 0, 0, 0);
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
