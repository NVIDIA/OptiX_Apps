/* 
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#ifndef DEV_MESH_H
#define DEV_MESH_H

// Always include this before any OptiX headers!
#include <cuda_runtime.h>

#include <optix.h>

#include <vector>

// glm/gtx/component_wise.hpp doesn't compile when not setting GLM_ENABLE_EXPERIMENTAL.
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>


// GLTF specifies a Mesh as a number of Primitives.

#include "DeviceBuffer.h"
#include "cuda/geometry_data.h"

// Just some namespace ("development") to distinguish from fastgltf::Mesh.
namespace dev
{

  class Primitive
  {
  public:
    Primitive()
      : currentMaterial(-1)
      , indexMaterial(-1) 
    {
    }

    // This is required because of the DeviceBuffer implementation needs move operators.
    Primitive::Primitive(Primitive&& that) noexcept
    {
      operator=(std::move(that));
    }
    Primitive& operator=(const Primitive&) = delete;
    Primitive& operator=(Primitive&)       = delete;
    Primitive& operator=(Primitive&& that) = default;

#if 0
    // DEBUG Check tangent attributes for consistency. 
    // BUG Some Khronos GLTF sample models provide invalid tangents (at least AnimatedMorphCube.gltf and Sponza.gltf)
    // which means either the host code or the device code needs to check that and generate working tangents. 
    bool checkTangents()
    {
      bool result = true;

      // If the primitive doesn't have tangents the automatic generation is used inside the device code.
      if (tangents.h_ptr == nullptr)
      {
        return result;
      }

      // Here there are tangents inside the host array.
      if (indices.h_ptr) // Indexed triangles when there are indices.
      {
        const size_t numTris = indices.count / 3;

        const uint3*  ui3Indices  = reinterpret_cast<const uint3*>(indices.h_ptr);
        const float3* f3Positions = reinterpret_cast<const float3*>(positions.h_ptr);
        const float4* f4Tangents  = reinterpret_cast<const float4*>(tangents.h_ptr);

        for (size_t tri = 0; tri < numTris; ++tri)
        {
          const uint3 idx = ui3Indices[tri];

          const float3 p0 = f3Positions[idx.x];
          const float3 p1 = f3Positions[idx.y];
          const float3 p2 = f3Positions[idx.z];

          const float3 Ng = normalize(cross(p1 - p0, p2 - p0));

          const float4 t0 = f4Tangents[idx.x];
          const float4 t1 = f4Tangents[idx.y];
          const float4 t2 = f4Tangents[idx.z];

          const float c0 = 1.0f - fabsf(dot(make_float3(t0), Ng));
          const float c1 = 1.0f - fabsf(dot(make_float3(t1), Ng));
          const float c2 = 1.0f - fabsf(dot(make_float3(t2), Ng));

          // If all three tangents are collinear to the geometric normal, the interpolation of them over the triangle will always generate collinear tangents.
          // This is not taking into account completely opposite directions of tangents, but that would be really bad geometry anyway.
          if (c0 < DENOMINATOR_EPSILON &&
              c1 < DENOMINATOR_EPSILON &&
              c2 < DENOMINATOR_EPSILON)
          {
            std::cout << "ERROR: Invalid tangent at indexed triangle " << tri << '\n';
            result = false;
          }
        }
      }
      else // Independent triangles when there are no indices.
      {
        const size_t numTris = positions.count / 3;

        const float3* f3Positions = reinterpret_cast<const float3*>(positions.h_ptr);
        const float4* f4Tangents  = reinterpret_cast<const float4*>(tangents.h_ptr);

        for (size_t tri = 0; tri < numTris; tri)
        {
          const size_t idx = tri * 3;

          const float3 p0 = f3Positions[idx    ];
          const float3 p1 = f3Positions[idx + 1];
          const float3 p2 = f3Positions[idx + 2];

          const float3 Ng = normalize(cross(p1 - p0, p2 - p0));

          const float4 t0 = f4Tangents[idx    ];
          const float4 t1 = f4Tangents[idx + 1];
          const float4 t2 = f4Tangents[idx + 2];

          const float c0 = 1.0f - fabsf(dot(make_float3(t0), Ng));
          const float c1 = 1.0f - fabsf(dot(make_float3(t1), Ng));
          const float c2 = 1.0f - fabsf(dot(make_float3(t2), Ng));

          if (c0 < DENOMINATOR_EPSILON &&
              c1 < DENOMINATOR_EPSILON && 
              c2 < DENOMINATOR_EPSILON)
          {
            std::cout << "ERROR: Invalid tangent at independent triangle " << tri << ", c0 = " << c0<< ", c1 = " << c1 << ", c2 = " << c2 << '\n';
            result = false;
          }
        }
      }
      return result;
    }

    // DEBUG Check tangent attributes for consistency. 
    // BUG Some Khronos GLTF sample models provide invalid tangents (at least AnimatedMorphCube.gltf and Sponza.gltf)
    // which means either the host code or the device code needs to check that and generate working tangents. 
    bool checkNormals()
    {
      bool result = true;

      // If the primitive doesn't have tangents the automatic generation is used inside the device code.
      if (normals.h_ptr == nullptr)
      {
        return result;
      }

      // Here there are tangents inside the host array.
      if (indices.h_ptr) // Indexed triangles when there are indices.
      {
        const size_t numTris = indices.count / 3;

        const uint3*  ui3Indices  = reinterpret_cast<const uint3*>(indices.h_ptr);
        const float3* f3Positions = reinterpret_cast<const float3*>(positions.h_ptr);
        const float3* f3Normals   = reinterpret_cast<const float3*>(normals.h_ptr);

        for (size_t tri = 0; tri < numTris; ++tri)
        {
          const uint3 idx = ui3Indices[tri];

          const float3 p0 = f3Positions[idx.x];
          const float3 p1 = f3Positions[idx.y];
          const float3 p2 = f3Positions[idx.z];

          const float3 Ng = normalize(cross(p1 - p0, p2 - p0));

          const float3 n0 = f3Normals[idx.x];
          const float3 n1 = f3Normals[idx.y];
          const float3 n2 = f3Normals[idx.z];

          // Check is the normal attribute is perpendicular to the geometry normal.
          // TransmissionThinwallTestGrid.gltf does that for the font geometry which breaks the TBN space calculation.
          const float c0 = fabsf(dot(n0, Ng));
          const float c1 = fabsf(dot(n1, Ng));
          const float c2 = fabsf(dot(n2, Ng));

          // If all three normals are perpendicular to the geometric normal, the normal space TBN calculation migth result in Nan results and break the rendering.
          if (c0 < DENOMINATOR_EPSILON &&
              c1 < DENOMINATOR_EPSILON &&
              c2 < DENOMINATOR_EPSILON)
          {
            std::cout << "ERROR: Normal attribute at indexed triangle " << tri << " perpendicular to geometry normal. Might result in NaN results for TBN.\n";
            result = false;
          }
        }
      }
      else // Independent triangles when there are no indices.
      {
        const size_t numTris = positions.count / 3;

        const float3* f3Positions = reinterpret_cast<const float3*>(positions.h_ptr);
        const float3* f3Normals   = reinterpret_cast<const float3*>(normals.h_ptr);

        for (size_t tri = 0; tri < numTris; tri)
        {
          const size_t idx = tri * 3;

          const float3 p0 = f3Positions[idx    ];
          const float3 p1 = f3Positions[idx + 1];
          const float3 p2 = f3Positions[idx + 2];

          const float3 Ng = normalize(cross(p1 - p0, p2 - p0));

          const float3 n0 = f3Normals[idx    ];
          const float3 n1 = f3Normals[idx + 1];
          const float3 n2 = f3Normals[idx + 2];

          const float c0 = fabsf(dot(n0, Ng));
          const float c1 = fabsf(dot(n1, Ng));
          const float c2 = fabsf(dot(n2, Ng));

          // If all three normals are perpendicular to the geometric normal, the normal space TBN calculation migth result in Nan results and break the rendering.
          if (c0 < DENOMINATOR_EPSILON &&
              c1 < DENOMINATOR_EPSILON && 
              c2 < DENOMINATOR_EPSILON)
          {
            std::cout << "ERROR: Invalid tangent at independent triangle " << tri << ", c0 = " << c0<< ", c1 = " << c1 << ", c2 = " << c2 << '\n';
            result = false;
          }
        }
      }
      return result;
    }

#endif

public:
    DeviceBuffer indices;                       // unsigned int
    DeviceBuffer positions;                     // float3 (The only mandatory attribute!)
    DeviceBuffer normals;                       // float3
    DeviceBuffer texcoords[NUM_ATTR_TEXCOORDS]; // float2
    DeviceBuffer colors;                        // float4
    DeviceBuffer tangents;                      // float4 (.w == 1.0 or -1.0 for the handedness)
    DeviceBuffer joints[NUM_ATTR_JOINTS];       // ushort4
    DeviceBuffer weights[NUM_ATTR_WEIGHTS];     // float4
    
    // This is the currently active material index used on device side.
    // Because of the KHR_materials_variants mappings below, each primitive needs to know 
    // which material is currently used to be able to determine if an AS needs to be rebuilt
    // due to a material change after a variant switch.
    int32_t currentMaterial;
    
    // This is the default material index when there are no variants. 
    // -1 when there is no material assigned at all, which will use default material parameters.
    int32_t indexMaterial; 

    // KHR_materials_variants
    // This vector contains a mapping from variant index to material index.
    // If the mapping is empty, the primitive uses the indexMaterial above.
    std::vector<int32_t> mappings;
  };

  class Mesh
  {
  public:
    Mesh()
      : gas(0)
      , d_gas(0)
      , isDirty(true)
    {
    }

    ~Mesh()
    {
      if (d_gas)
      {
        CUDA_CHECK_NO_THROW( cudaFree(reinterpret_cast<void*>(d_gas)) );
      }
    }
  
    // This is required because of the DeviceBuffer implementation needs move operators.
    Mesh(Mesh&& that) noexcept
    {
      operator=(std::move(that));
    }
    Mesh& operator=(const Mesh&) = delete;
    Mesh& operator=(Mesh&) = delete;
    Mesh& operator=(Mesh&& that) noexcept
    {
      name       = that.name;
      gas        = that.gas;
      d_gas      = that.d_gas;
      isDirty    = that.isDirty;
      primitives = std::move(that.primitives); // Need to move these because they use DeviceBuffers.
      
      that.gas     = 0;
      that.d_gas   = 0; // This makes sure the destructor on "that" is not freeing the copied d_gas.
      that.isDirty = false;
      //that.primitives.clear(); // Cleared by the std::move.

      return *this;
    }

  public:
    std::string            name;
    OptixTraversableHandle gas;
    CUdeviceptr            d_gas;
    bool                   isDirty; // true when the GAS needs to be rebuilt.

    std::vector<dev::Primitive> primitives; // If this vector is not empty, there are triangle primitives inside the mesh.
  };

  struct Instance
  {
    glm::mat4x4 transform;
    int         indexMesh;
  };

} // namespace dev

#endif // DEV_MESH_H

