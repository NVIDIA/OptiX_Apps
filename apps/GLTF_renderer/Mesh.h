/* 
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include "DeviceBuffer.h"
#include "HostBuffer.h"

#include "cuda/geometry_data.h"
#include "cuda/material_data.h"

// Just some namespace ("development") to distinguish from fastgltf::Mesh.
namespace dev
{
  /// Triangles, Points, ...
  /// Keep this file independent on fastgltf.
  enum PrimitiveType : uint32_t
  {
    Undefined,
    Triangles,
    Points
    //Lines, TODO
  };


  class IPrimitive
  {
  public:

    PrimitiveType getPrimitiveType() const 
    { 
      return primitiveType;
    }

    /// So that builders of device meshes know what to do.
    void          setPrimitiveType(const dev::PrimitiveType t)
    {
      primitiveType = t;
    }

  protected:
    PrimitiveType primitiveType{ Undefined };
  };


  /// Can be Triangles or Points or Lines, host version.
  class HostPrimitive : public IPrimitive
  {
  public:

    HostPrimitive()
      : numTargets(0)
      , maskTargets(0)
    {
    }

    /// Set a debug name
    void setName(std::string name)
    {
      this->name = name;
    }

    /// Get the debug name
    const std::string& getName() const
    {
      return name;
    }

    // This is required because the HostBuffer implementation uses move operators.
    HostPrimitive(HostPrimitive&& that) noexcept
    {
      operator=(std::move(that));
    }

    HostPrimitive& operator=(const HostPrimitive&) = delete;
    HostPrimitive& operator=(HostPrimitive&)       = delete;
    HostPrimitive& operator=(HostPrimitive&& that) = default;

  public:
    // These are the base attributes without morphing or skinning applied.
    HostBuffer indices;                       // unsigned int
    HostBuffer positions;                     // float3 (The only mandatory attribute!)
    HostBuffer tangents;                      // float4 (.w == 1.0 or -1.0 for the handedness)
    HostBuffer normals;                       // float3
    HostBuffer colors;                        // float4
    HostBuffer texcoords[NUM_ATTR_TEXCOORDS]; // float2
    HostBuffer joints[NUM_ATTR_JOINTS];       // ushort4
    HostBuffer weights[NUM_ATTR_WEIGHTS];     // float4

    // Vector of morph targets attributes. 
    // Only sized to numTargets when there are targets for the respective attribute.
    std::vector<HostBuffer> positionsTarget;
    std::vector<HostBuffer> tangentsTarget;
    std::vector<HostBuffer> normalsTarget;
    std::vector<HostBuffer> colorsTarget;
    std::vector<HostBuffer> texcoordsTarget[NUM_ATTR_TEXCOORDS];

    // Morphing.
    size_t numTargets;  // Number of morph tagets.
    int    maskTargets; // Bitfield which encodes which attributes have morph targets.

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
    std::string      name;          // debug
  };

  /// For triangles only
  class HostMesh
  {
  public:
    HostMesh()
      : isDirty(false)
      , numTargets(0)
    {
    }

    // This is required because the HostBuffer implementation uses move operators.
    HostMesh(HostMesh&& that) noexcept
    {
      operator=(std::move(that));
    }
    HostMesh& operator=(const HostMesh&) = delete;
    HostMesh& operator=(HostMesh&)       = delete;
    HostMesh& operator=(HostMesh&& that) = default;

  public:
    std::string name;

    bool isDirty;               // true when the variant on this material changed.

    size_t numTargets;          // The number of morph targets inside the mesh.
    std::vector<float> weights; // Optional morph weights on the mesh itself.

    /// @return A new primitive to initialise.
    HostPrimitive& createNewPrimitive(const dev::PrimitiveType t, std::string name)
    {
      MY_ASSERT(t != Undefined);
      HostPrimitive& prim = primitives.emplace_back();
      prim.setPrimitiveType(t);
      prim.setName(name);
      return prim;
    }

  public:

    /// GLTF specifies a Mesh as a number of Primitives (in the sense of OpenGL draw calls, which map to different OptixBuildInputs in a GAS.)
    /// If this vector is not empty, there are primitives inside the mesh. 
    std::vector<dev::HostPrimitive> primitives;
  };


  ///  Triangles or Points or Lines, device version.
  class DevicePrimitive : public IPrimitive
  {
  public:
        DevicePrimitive()
      : vertexBuffer{0}
      , numTargets(0)
      , maskTargets(0)
      , currentMaterial(-1)
    {
    }

    /// Set a debug name.
    void setName(const std::string& name)
    {
      this->name = name;
    }

    // This is required because of the DeviceBuffer implementation needs move operators.
    DevicePrimitive(DevicePrimitive&& that) noexcept
    {
      operator=(std::move(that));
    }
    DevicePrimitive& operator=(const DevicePrimitive&) = delete;
    DevicePrimitive& operator=(DevicePrimitive&)       = delete;
    DevicePrimitive& operator=(DevicePrimitive&& that) = default;

    // Convenience functions to get the final position attribute device pointer.
    CUdeviceptr getPositionsPtr() const
    {
      if (positionsSkinned.d_ptr)
      {
        return positionsSkinned.d_ptr;
      }
      if (positionsMorphed.d_ptr)
      {
        return positionsMorphed.d_ptr;
      }
      return positions.d_ptr;
    }

    CUdeviceptr getTangentsPtr() const
    {
      if (tangentsSkinned.d_ptr)
      {
        return tangentsSkinned.d_ptr;
      }
      if (tangentsMorphed.d_ptr)
      {
        return tangentsMorphed.d_ptr;
      }
      return tangents.d_ptr;
    }

    CUdeviceptr getNormalsPtr() const
    {
      if (normalsSkinned.d_ptr)
      {
        return normalsSkinned.d_ptr;
      }
      if (normalsMorphed.d_ptr)
      {
        return normalsMorphed.d_ptr;
      }
      return normals.d_ptr;
    }

    CUdeviceptr getTexcoordsPtr(const int idx) const
    {
      if (texcoordsMorphed[idx].d_ptr)
      {
        return texcoordsMorphed[idx].d_ptr;
      }
      return texcoords[idx].d_ptr;
    }

    CUdeviceptr getColorsPtr() const
    {
      if (colorsMorphed.d_ptr)
      {
        return colorsMorphed.d_ptr;
      }
      return colors.d_ptr;
    }

    /// Builds triangles or spheres or lines, depending on the primitive type.
    /// Non-virtual: currently it's the simplest approach (avoids a class hierarchy and a factory).
    /// @param buildInput   Is not 0-initialised here, gets the relevant members set, depending on
    ///                     the primitive type.
    /// @param sphereRadius A very small fraction of the scene size e.g. 0.001f * scene.aabb.diameter().
    /// @return True if the buildInput was successfully set up.
    bool setupBuildInput(OptixBuildInput& buildInput,
                         const std::vector<MaterialData>& materials,
                         const unsigned int * inputFlagsOpaque,
                         const unsigned int * inputFlagsMask,
                         const unsigned int * inputFlagsBlend,
                         const float          sphereRadius) const;

private:

    /// Select an OptixBuildInput mesh flag depending on the material's alpha mode.
    /// Also works when no material is selected.
    /// @return The address of one of the given flags.
    const unsigned int* getBuildInputFlags(const std::vector<MaterialData>& materials,
                                           const unsigned int* inputFlagsOpaque,
                                           const unsigned int* inputFlagsMask,
                                           const unsigned int* inputFlagsBlend) const;
public:
    std::string  name;                          // debug
    DeviceBuffer indices;                       // unsigned int
    DeviceBuffer positions;                     // float3 (The only mandatory attribute!)
    DeviceBuffer normals;                       // float3
    DeviceBuffer texcoords[NUM_ATTR_TEXCOORDS]; // float2
    DeviceBuffer colors;                        // float4
    DeviceBuffer tangents;                      // float4 (.w == 1.0 or -1.0 for the handedness)
    DeviceBuffer joints[NUM_ATTR_JOINTS];       // ushort4
    DeviceBuffer weights[NUM_ATTR_WEIGHTS];     // float4

    // Morphed attributes (only initialized for attributes which have morph targets.
    DeviceBuffer positionsMorphed;                     // float3
    DeviceBuffer tangentsMorphed;                      // float3 (No morphed handedness!)
    DeviceBuffer normalsMorphed;                       // float3
    DeviceBuffer colorsMorphed;                        // float4
    DeviceBuffer texcoordsMorphed[NUM_ATTR_TEXCOORDS]; // float2

    // Skinned attributes (sources either base or morph attributes).
    DeviceBuffer positionsSkinned; // float3
    DeviceBuffer tangentsSkinned;  // float4
    DeviceBuffer normalsSkinned;   // float3

    // Vectors of morph target attributes.
    // Only sized to numTargets when there are targets for the respective attribute.
    std::vector<DeviceBuffer> positionsTarget;
    std::vector<DeviceBuffer> tangentsTarget;
    std::vector<DeviceBuffer> normalsTarget;
    std::vector<DeviceBuffer> colorsTarget;
    std::vector<DeviceBuffer> texcoordsTarget[NUM_ATTR_TEXCOORDS];

    // This is an array of all enabled morph target CUdeviceptr from the above morph target vector.
    // The maskTargets bitfield defines wich target pointers are included in the order of the ATTR_* bitfield defines.
    // There are numTarget pointers for each enabled morph attribute.
    DeviceBuffer targetPointers;

    // This CUdeviceptr will hold the final position attribute pointer (precedence: skinned, morphed, base).
    // This is initialized inside createDevicePrimitive() and used by the OptixBuildInput vertexBuffers pointer.
    CUdeviceptr vertexBuffer;

    int numTargets;  // Number of targets per morphed attribute in this primitive.
    int maskTargets; // Bitfield indicating which attributes have morph targets.

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
      // This is ordered from most to least often changing indices inside keys.
      return (idxHostMesh <  rhs.idxHostMesh) ||                                                  // Most often different meshes.
             (idxHostMesh == rhs.idxHostMesh && idxSkin <  rhs.idxSkin) ||                        // The same mesh with different skin nodes.
             (idxHostMesh == rhs.idxHostMesh && idxSkin == rhs.idxSkin && idxNode < rhs.idxNode); // The same mesh and skin with different morph weights.
    }

    int idxHostMesh = -1; // The mesh index on the node.
    int idxSkin = -1;     // The skin index on the node.
    int idxNode = -1;     // The node index when it contains morph weights.
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

    /// If this vector is not empty, there are triangle primitives inside the mesh.
    std::vector<dev::DevicePrimitive> primitives;
  };


} // namespace dev

#endif // DEV_MESH_H

