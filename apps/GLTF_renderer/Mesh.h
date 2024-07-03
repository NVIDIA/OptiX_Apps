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
  class HostPrimitive
  {
  public:
    HostPrimitive()
      : numTargets(0)
      , maskTargets(0)
      , currentMaterial(-1)
      , indexMaterial(-1) 
    {
    }

    // This is required because the HostBuffer implementation uses move operators.
    HostPrimitive::HostPrimitive(HostPrimitive&& that) noexcept
    {
      operator=(std::move(that));
    }
    HostPrimitive& operator=(const HostPrimitive&) = delete;
    HostPrimitive& operator=(HostPrimitive&)       = delete;
    HostPrimitive& operator=(HostPrimitive&& that) = default;

public:
    HostBuffer indices;                       // unsigned int
    HostBuffer positions;                     // float3 (The only mandatory attribute!)
    HostBuffer tangents;                      // float4 (.w == 1.0 or -1.0 for the handedness)
    HostBuffer normals;                       // float3
    HostBuffer colors;                        // float4
    HostBuffer texcoords[NUM_ATTR_TEXCOORDS]; // float2
    HostBuffer joints[NUM_ATTR_JOINTS];       // ushort4
    HostBuffer weights[NUM_ATTR_WEIGHTS];     // float4

    // Skinning animation.
    HostBuffer positionsSkinned; // float3
    HostBuffer tangentsSkinned;  // float4
    HostBuffer normalsSkinned;   // float3

    // Morphing.
    size_t numTargets;  // Number of morph tagets.
    int    maskTargets; // Bitfield which encodes which attributes have morph targets.

    // Vector of morph targets attributes. 
    // Only sized to numTargets when there are targets for the respective attribute.
    std::vector<HostBuffer> positionsTarget;
    std::vector<HostBuffer> tangentsTarget;
    std::vector<HostBuffer> normalsTarget;
    std::vector<HostBuffer> colorsTarget;
    std::vector<HostBuffer> texcoordsTarget[NUM_ATTR_TEXCOORDS];

    HostBuffer positionsMorphed;                     // float3
    HostBuffer tangentsMorphed;                      // float3 (No morphed handedness!)
    HostBuffer normalsMorphed;                       // float3
    HostBuffer colorsMorphed;                        // float4
    HostBuffer texcoordsMorphed[NUM_ATTR_TEXCOORDS]; // float2

    // This is the currently active material index used on device side.
    // Because of the KHR_materials_variants mappings below, each primitive needs to know 
    // which material is currently used to be able to determine if an AS needs to be rebuilt
    // due to a material change after a variant switch.
    int currentMaterial;
    
    // This is the default material index when there are no variants. 
    // -1 when there is no material assigned at all, which will use default material parameters.
    int indexMaterial; 

    // KHR_materials_variants
    // This vector contains a mapping from variant index to material index.
    // If the mapping is empty, the primitive uses the indexMaterial above.
    std::vector<int> mappings;
  };


  class HostMesh
  {
  public:
    HostMesh()
      : isDirty(false)
      , isMorphed(false)
      , numTargets(0)
    {
    }

    // This is required because the HostBuffer implementation uses move operators.
    HostMesh::HostMesh(HostMesh&& that) noexcept
    {
      operator=(std::move(that));
    }
    HostMesh& operator=(const HostMesh&) = delete;
    HostMesh& operator=(HostMesh&)       = delete;
    HostMesh& operator=(HostMesh&& that) = default;

  public:
    std::string name;

    bool isDirty;   // true when the variant on this material changed.
    bool isMorphed; // true when any of the HostPrimitives contains morph targets for attributes supported by the renderer.

    size_t numTargets;          // The number of morph targets inside the mesh.
    std::vector<float> weights; // Optional morph weights on the HostMesh itself. Only used when there are no morph weights on the parent node.
    
    std::vector<dev::HostPrimitive> primitives; // If this vector is not empty, there are triangle primitives inside the mesh.
  };


  class DevicePrimitive
  {
  public:
    DevicePrimitive()
      : currentMaterial(-1)
    {
    }

    // This is required because of the DeviceBuffer implementation needs move operators.
    DevicePrimitive::DevicePrimitive(DevicePrimitive&& that) noexcept
    {
      operator=(std::move(that));
    }
    DevicePrimitive& operator=(const DevicePrimitive&) = delete;
    DevicePrimitive& operator=(DevicePrimitive&)       = delete;
    DevicePrimitive& operator=(DevicePrimitive&& that) = default;

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
  };


  class KeyTuple
  {
  public:
    // When using this as key in a map, operator<() needs to be implemented.
    bool operator<(const KeyTuple& rhs) const
    {
      return (idxNode <  rhs.idxNode) ||
             (idxNode == rhs.idxNode && idxSkin <  rhs.idxSkin) ||
             (idxNode == rhs.idxNode && idxSkin == rhs.idxSkin && idxMesh < rhs.idxMesh);
    }

    int idxNode = -1; // The node index when it contains morph weights.
    int idxSkin = -1; // The skin index on the node.
    int idxMesh = -1; // The (host) mesh index on the node.
  };

  class DeviceMesh
  {
  public:
    DeviceMesh()
      : gas(0)
      , d_gas(0)
      , isDirty(true)
    {
    }

    ~DeviceMesh()
    {
      if (d_gas)
      {
        CUDA_CHECK_NO_THROW( cudaFree(reinterpret_cast<void*>(d_gas)) );
      }
    }
  
    // This is required because of the DeviceBuffer implementation needs move operators.
    DeviceMesh(DeviceMesh&& that) noexcept
    {
      operator=(std::move(that));
    }
    DeviceMesh& operator=(const DeviceMesh&) = delete;
    DeviceMesh& operator=(DeviceMesh&) = delete;
    DeviceMesh& operator=(DeviceMesh&& that) noexcept
    {
      gas           = that.gas;
      d_gas         = that.d_gas;
      key           = that.key;
      isDirty       = that.isDirty;
      primitives    = std::move(that.primitives); // Need to move these because they use DeviceBuffers.
      
      that.gas     = 0;
      that.d_gas   = 0; // This makes sure the destructor on "that" is not freeing the copied d_gas.

      return *this;
    }

  public:
    OptixTraversableHandle gas;
    CUdeviceptr            d_gas;

    KeyTuple key;

    bool isDirty; // true when the GAS needs to be rebuilt.

    std::vector<dev::DevicePrimitive> primitives; // If this vector is not empty, there are triangle primitives inside the mesh.
  };


  struct Instance
  {
    glm::mat4x4 transform;
    int         indexDeviceMesh; // Index into m_deviceMeshes.
  };

} // namespace dev

#endif // DEV_MESH_H

