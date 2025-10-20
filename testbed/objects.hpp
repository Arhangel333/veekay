#include "mainlibs.hpp"


namespace
{
void generateCylinderVertices(Vertex *vertices, int segments, float radius, float height)
	{
		for (int i = 0; i <= segments; ++i)
		{
			float angle = 2.0f * M_PI * i / segments;
			float x = radius * cosf(angle);
			float z = radius * sinf(angle);

			vertices[i * 2] = Vertex{{x, height / 2, z}};	   // НИЗ
			vertices[i * 2 + 1] = Vertex{{x, -height / 2, z}}; // ВЕРХ
		}
	}

	void generateCylinderIndices(uint32_t *indices, int segments)
	{
		for (int i = 0; i < segments; ++i)
		{
			uint32_t base = i * 2;
			indices[i * 6] = base + 1;
			indices[i * 6 + 1] = base + 2;
			indices[i * 6 + 2] = base;
			indices[i * 6 + 3] = base + 3;
			indices[i * 6 + 4] = base + 2;
			indices[i * 6 + 5] = base + 1;
		}
	}

}