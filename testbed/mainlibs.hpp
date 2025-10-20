#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

namespace
{

	constexpr float camera_fov = 70.0f;
	constexpr float camera_near_plane = 0.01f;
	constexpr float camera_far_plane = 100.0f;

	struct Matrix
	{
		float m[4][4];
	};

	struct Vector
	{
		float x, y, z;
	};

	struct Vertex
	{
		Vector position;
		// NOTE: You can add more attributes
	};

	// NOTE: These variable will be available to shaders through push constant uniform
	struct ShaderConstants
	{
		Matrix projection;
		Matrix transform;
		Vector color;
	};

	struct VulkanBuffer
	{
		VkBuffer buffer;
		VkDeviceMemory memory;
	};


    struct Object
    {
        
    };
    
	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;
	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;

	// NOTE: Declare buffers and other variables here
	VulkanBuffer vertex_buffer;
	VulkanBuffer index_buffer;
	bool ortografics = 0;  // 1 - ортографическая проекция; 0 - обычная
	bool obj_rotation = 0; // rotation if == 1
	int indices_count = 0;

	Vector model_position = {0.0f, 0.0f, 5.0f}; //{0.0f, 0.0f, 5.0f};
	float model_rotation;						// вращение вокруг оси У
	// float model_pos;		//
	float animationSpeed = 1.0f;   // скорость движения по траектории
	float trajectoryRadius = 1.0f; // радиус траектории
	float newangle = 0.0f;

	Vector model_color = {0.5f, 1.0f, 0.7f};
	bool model_spin = false;

	Matrix identity()
	{
		Matrix result{};

		result.m[0][0] = 1.0f;
		result.m[1][1] = 1.0f;
		result.m[2][2] = 1.0f;
		result.m[3][3] = 1.0f;

		return result;
	}

	Matrix projection(float fov, float aspect_ratio, float near, float far)
	{
		Matrix result{};

		const float radians = fov * M_PI / 180.0f;
		const float cot = 1.0f / tanf(radians / 2.0f);

		result.m[0][0] = cot / aspect_ratio;
		result.m[1][1] = cot;
		result.m[2][3] = 1.0f;

		result.m[2][2] = far / (far - near);
		result.m[3][2] = (-near * far) / (far - near);

		return result;
	}

	Matrix orthographicProjection(float aspect_ratio)
	{
		Matrix result = identity();
		float size = 8.0f;

		result.m[0][0] = 1.0f / (size * aspect_ratio);
		result.m[1][1] = 1.0f / size;
		result.m[2][2] = 0.1f;

		return result;
	}

	Matrix translation(Vector vector)
	{
		Matrix result = identity();

		result.m[3][0] = vector.x;
		result.m[3][1] = vector.y;
		result.m[3][2] = vector.z;

		return result;
	}

	Matrix rotation(Vector axis, float angle)
	{
		Matrix result{};

		float length = sqrtf(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);

		axis.x /= length;
		axis.y /= length;
		axis.z /= length;

		float sina = sinf(angle);
		float cosa = cosf(angle);
		float cosv = 1.0f - cosa;

		result.m[0][0] = (axis.x * axis.x * cosv) + cosa;
		result.m[0][1] = (axis.x * axis.y * cosv) + (axis.z * sina);
		result.m[0][2] = (axis.x * axis.z * cosv) - (axis.y * sina);

		result.m[1][0] = (axis.y * axis.x * cosv) - (axis.z * sina);
		result.m[1][1] = (axis.y * axis.y * cosv) + cosa;
		result.m[1][2] = (axis.y * axis.z * cosv) + (axis.x * sina);

		result.m[2][0] = (axis.z * axis.x * cosv) + (axis.y * sina);
		result.m[2][1] = (axis.z * axis.y * cosv) - (axis.x * sina);
		result.m[2][2] = (axis.z * axis.z * cosv) + cosa;

		result.m[3][3] = 1.0f;

		return result;
	}

	Matrix multiply(const Matrix &a, const Matrix &b)
	{
		Matrix result{};

		for (int j = 0; j < 4; j++)
		{
			for (int i = 0; i < 4; i++)
			{
				for (int k = 0; k < 4; k++)
				{
					result.m[j][i] += a.m[j][k] * b.m[k][i];
				}
			}
		}

		return result;
	}

}