/* 
 * Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
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
 
#ifndef DEVICE_H
#define DEVICE_H

#include "shaders/config.h"

#include <cuda.h>
// This is needed for the __align__ only.
#include <cuda_runtime.h>

#include <optix.h>

// OptiX 7 function table structure.
#include <optix_function_table.h>

#include "inc/Arena.h"
#include "inc/LightGUI.h"
//#include "inc/MaterialGUI.h"
#include "inc/Picture.h"
#include "inc/SceneGraph.h"
#include "inc/Texture.h"
#include "inc/MyAssert.h"

#include "inc/CompileResult.h"
#include "inc/MaterialMDL.h"
#include "inc/ShaderConfiguration.h"
#include "shaders/shader_configuration.h" // Device side of the the shader configuration.

#include "shaders/system_data.h"
#include "shaders/per_ray_data.h"
#include "shaders/texture_handler.h"

#include <map>
#include <memory>
#include <vector>

struct DeviceAttribute
{
  int maxThreadsPerBlock;
  int maxBlockDimX;
  int maxBlockDimY;
  int maxBlockDimZ;
  int maxGridDimX;
  int maxGridDimY;
  int maxGridDimZ;
  int maxSharedMemoryPerBlock;
  int sharedMemoryPerBlock;
  int totalConstantMemory;
  int warpSize;
  int maxPitch;
  int maxRegistersPerBlock;
  int registersPerBlock;
  int clockRate;
  int textureAlignment;
  int gpuOverlap;
  int multiprocessorCount;
  int kernelExecTimeout;
  int integrated;
  int canMapHostMemory;
  int computeMode;
  int maximumTexture1dWidth;
  int maximumTexture2dWidth;
  int maximumTexture2dHeight;
  int maximumTexture3dWidth;
  int maximumTexture3dHeight;
  int maximumTexture3dDepth;
  int maximumTexture2dLayeredWidth;
  int maximumTexture2dLayeredHeight;
  int maximumTexture2dLayeredLayers;
  int maximumTexture2dArrayWidth;
  int maximumTexture2dArrayHeight;
  int maximumTexture2dArrayNumslices;
  int surfaceAlignment;
  int concurrentKernels;
  int eccEnabled;
  int pciBusId;
  int pciDeviceId;
  int tccDriver;
  int memoryClockRate;
  int globalMemoryBusWidth;
  int l2CacheSize;
  int maxThreadsPerMultiprocessor;
  int asyncEngineCount;
  int unifiedAddressing;
  int maximumTexture1dLayeredWidth;
  int maximumTexture1dLayeredLayers;
  int canTex2dGather;
  int maximumTexture2dGatherWidth;
  int maximumTexture2dGatherHeight;
  int maximumTexture3dWidthAlternate;
  int maximumTexture3dHeightAlternate;
  int maximumTexture3dDepthAlternate;
  int pciDomainId;
  int texturePitchAlignment;
  int maximumTexturecubemapWidth;
  int maximumTexturecubemapLayeredWidth;
  int maximumTexturecubemapLayeredLayers;
  int maximumSurface1dWidth;
  int maximumSurface2dWidth;
  int maximumSurface2dHeight;
  int maximumSurface3dWidth;
  int maximumSurface3dHeight;
  int maximumSurface3dDepth;
  int maximumSurface1dLayeredWidth;
  int maximumSurface1dLayeredLayers;
  int maximumSurface2dLayeredWidth;
  int maximumSurface2dLayeredHeight;
  int maximumSurface2dLayeredLayers;
  int maximumSurfacecubemapWidth;
  int maximumSurfacecubemapLayeredWidth;
  int maximumSurfacecubemapLayeredLayers;
  int maximumTexture1dLinearWidth;
  int maximumTexture2dLinearWidth;
  int maximumTexture2dLinearHeight;
  int maximumTexture2dLinearPitch;
  int maximumTexture2dMipmappedWidth;
  int maximumTexture2dMipmappedHeight;
  int computeCapabilityMajor;
  int computeCapabilityMinor;
  int maximumTexture1dMipmappedWidth;
  int streamPrioritiesSupported;
  int globalL1CacheSupported;
  int localL1CacheSupported;
  int maxSharedMemoryPerMultiprocessor;
  int maxRegistersPerMultiprocessor;
  int managedMemory;
  int multiGpuBoard;
  int multiGpuBoardGroupId;
  int hostNativeAtomicSupported;
  int singleToDoublePrecisionPerfRatio;
  int pageableMemoryAccess;
  int concurrentManagedAccess;
  int computePreemptionSupported;
  int canUseHostPointerForRegisteredMem;
  int canUse64BitStreamMemOps;
  int canUseStreamWaitValueNor;
  int cooperativeLaunch;
  int cooperativeMultiDeviceLaunch;
  int maxSharedMemoryPerBlockOptin;
  int canFlushRemoteWrites;
  int hostRegisterSupported;
  int pageableMemoryAccessUsesHostPageTables;
  int directManagedMemAccessFromHost;
};

struct DeviceProperty
{
  unsigned int rtcoreVersion;
  unsigned int limitMaxTraceDepth;
  unsigned int limitMaxTraversableGraphDepth;
  unsigned int limitMaxPrimitivesPerGas;
  unsigned int limitMaxInstancesPerIas;
  unsigned int limitMaxInstanceId;
  unsigned int limitNumBitsInstanceVisibilityMask;
  unsigned int limitMaxSbtRecordsPerGas;
  unsigned int limitMaxSbtOffset;
};


// None of the programs is using SBT data!
struct SbtRecordHeader
{
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};


enum ModuleIdentifier
{
  MODULE_ID_RAYGENERATION,
  MODULE_ID_EXCEPTION,
  MODULE_ID_MISS,
  MODULE_ID_HIT,
  MODULE_ID_LENS_SHADER,
  MODULE_ID_LIGHT_SAMPLE,

  NUM_MODULE_IDENTIFIERS
};


enum ProgramGroupId
{
  PGID_RAYGENERATION,
  
  PGID_EXCEPTION,
  
  PGID_MISS_RADIANCE,
  PGID_MISS_SHADOW,
  
  // Hit records for triangles:
  // 0 = no emission, no cutout
  PGID_HIT_RADIANCE_0,
  PGID_HIT_SHADOW_0,
  // 1 = emission, no cutout
  PGID_HIT_RADIANCE_1,
  PGID_HIT_SHADOW_1,
  // 2 = no emission, cutout
  PGID_HIT_RADIANCE_2,
  PGID_HIT_SHADOW_2,
  // 3 = emission, cutout
  PGID_HIT_RADIANCE_3,
  PGID_HIT_SHADOW_3,
  // 4 = cubic B-spline curves.
  PGID_HIT_CURVES,
  PGID_HIT_CURVES_SHADOW,
    
  // Direct Callables
  // Lens shader
  PGID_LENS_PINHOLE,
  PGID_LENS_FISHEYE,
  PGID_LENS_SPHERE,
  // Light sample
  PGID_LIGHT_ENV_CONSTANT,
  PGID_LIGHT_ENV_SPHERE,
  PGID_LIGHT_MESH,
  PGID_LIGHT_POINT, // singular
  PGID_LIGHT_SPOT,  // singular
  PGID_LIGHT_IES,   // singular (same as point light but with additional IES light profile luminance texture.)

  // Number of all hardcoded program group entries. 
  // There are more direct callables appended by the MDL materials.
  NUM_PROGRAM_GROUP_IDS
};

// This is in preparation for more than one geometric primitive type.
enum PrimitiveType
{
  PT_UNKNOWN,   // It's an error when this is still set.
  PT_TRIANGLES,
  PT_CURVES
};


struct GeometryData
{
  GeometryData()
  : primitiveType(PT_UNKNOWN)
  , owner(-1)
  , traversable(0)
  , d_attributes(0)
  , d_indices(0)
  , numAttributes(0)
  , numIndices(0)
  , d_gas(0)
  {
    info = {};
  }

  PrimitiveType            primitiveType; // Triangles or cubic B-spline curves. Used to pick the correct hit record inside the SBT.
  int                      owner;         // The device index which originally allocated all device side memory below. Needed when sharing GeometryData, resp. when freeing it.
  OptixTraversableHandle   traversable;   // The traversable handle for this GAS. Assigned to the Instance above it.
  CUdeviceptr              d_attributes;  // Array of VertexAttribute structs.
  CUdeviceptr              d_indices;     // Array of unsigned int indices.
  size_t                   numAttributes; // Count of attributes structs.
  size_t                   numIndices;    // Count of unsigned int indices (not triplets for triangles).
  CUdeviceptr              d_gas;         // The geometry AS for the traversable handle.
#if (OPTIX_VERSION >= 70600)
  OptixRelocationInfo      info;          // The relocation info allows to check if the AS is compatible across OptiX contexts. 
                                          // (In the NVLINK case that isn't really required, because the GPU configuration must be homogeneous inside an NVLINK island.)
#else
  OptixAccelRelocationInfo info;
#endif
};

struct InstanceData
{
  InstanceData(const unsigned int geometry,
               const int material,
               const int light,
               const int object)
  : idGeometry(geometry)
  , idMaterial(material)
  , idLight(light)
  , idObject(object)
  {
  }

  unsigned int idGeometry;
  int          idMaterial; // Negative is an error.
  int          idLight;    // Negative means no light.
  int          idObject;   // Application-specific identifier used in state.object_id. It's the sg::Instance ID.
};

// GUI controllable settings in the device.
struct DeviceState
{
  int2     resolution;
  int2     tileSize;
  int2     pathLengths;
  int      walkLength;
  int      samplesSqrt;
  TypeLens typeLens;
  float    epsilonFactor;
  float    clockFactor;
  int      directLighting;
};


class Device;

// To be able to share texture CUarrays among devices inside a CUDA peer-to-peer island, 
// the owner device and all data required to create a texture object from the shared CUarray are required.
struct TextureMDLHost
{
  Device* m_owner; // The device which created this Texture. Needed for peer-to-peer sharing, resp. for proper destruction.

  int m_index; // The index inside the per-device cache vector. 
               // Used to maintain the MaterialMDL indices vector with the indices into that cache. 

  CUDA_ARRAY3D_DESCRIPTOR m_descArray3D; // FIXME This is not actually required for the sharing.
  CUDA_RESOURCE_DESC      m_resourceDescription;
  CUDA_TEXTURE_DESC       m_textureDescription;

  CUarray m_d_array; // FIXME This is also in host.m_resourceDescription.res.array.hArray

  // How much memory the CUarray required in bytes, input to cuMemcpy3d()
  // (without m_deviceAttribute.textureAlignment or potential row padding on the device).
  size_t m_sizeBytesArray;

  // The per-device texture objects and sizes are tracked in this struct 
  // which is used inside the Texture_handler for the texture lookup functions.
  TextureMDL m_texture;
};


struct MbsdfHost
{
  Device* m_owner; // The device which created this Texture. Needed for peer-to-peer sharing, resp. for proper destruction.

  int m_index; // The index inside the per-device cache vector. 

  CUDA_ARRAY3D_DESCRIPTOR m_descArray3D[2]; // FIXME This is not actually required for the sharing.
  CUDA_RESOURCE_DESC      m_resourceDescription[2];
  CUDA_TEXTURE_DESC       m_textureDescription; // Same settings for both parts!

  CUarray m_d_array[2]; // FIXME This is also in host.m_resourceDescription.res.array.hArray
  // How much memory the CUarray required in bytes, input to cuMemcpy3d()
  // (without m_deviceAttribute.textureAlignment or potential row padding on the device).
  size_t m_sizeBytesArray[2];

  Mbsdf m_mbsdf;
};


struct LightprofileHost
{
  Device* m_owner; // The device which created this Texture. Needed for peer-to-peer sharing, resp. for proper destruction.

  int m_index; // The index inside the per-device cache vector. 

  CUDA_ARRAY3D_DESCRIPTOR m_descArray3D; // FIXME This is not actually required for the sharing.
  CUDA_RESOURCE_DESC      m_resourceDescription;
  CUDA_TEXTURE_DESC       m_textureDescription;

  CUarray m_d_array; // FIXME This is also in host.m_resourceDescription.res.array.hArray
  // How much memory the CUarray required in bytes, input to cuMemcpy3d()
  // (without m_deviceAttribute.textureAlignment or potential row padding on the device).
  size_t m_sizeBytesArray;

  Lightprofile m_profile;
};



class Device
{
public:
  Device(const int ordinal,       // The original CUDA ordinal ID.
         const int index,         // The zero based index of this device, required for multi-GPU work distribution.
         const int count ,        // The number of active devices, required for multi-GPU work distribution.
         const TypeLight typeEnv, // Controls which miss shader to use.
         const int interop,       // The interop mode to use.
         const unsigned int tex,  // OpenGL HDR texture object handle
         const unsigned int pbo,  // OpenGL PBO handle.
         const size_t sizeArena); // The default Arena size in mega-bytes.
  ~Device();

  bool matchUUID(const char* uuid);
  bool matchLUID(const char* luid, const unsigned int nodeMask);
    
  void initCameras(const std::vector<CameraDefinition>& cameras);
  void initLights(const std::vector<LightGUI>& lightsGUI, const std::vector<GeometryData>& geometryData, const unsigned int stride, const unsigned int index); // Must be called after initScene()!
  
  GeometryData createGeometry(std::shared_ptr<sg::Triangles> geometry);
  GeometryData createGeometry(std::shared_ptr<sg::Curves> geometry);
  void destroyGeometry(GeometryData& data);
  void createInstance(const GeometryData& geometryData, const InstanceData& data, const float matrix[12]);
  void createTLAS();
  void createGeometryInstanceData(const std::vector<GeometryData>& geometryData, const unsigned int stride, const unsigned int index);

  void updateCamera(const int idCamera, const CameraDefinition& camera);
  void updateLight(const int idLight, const LightGUI& lightsGUI);
  //void updateLight(const int idLight, const LightDefinition& light);
  void updateMaterial(const int idMaterial, const MaterialMDL* materialMDL);
  
  void setState(const DeviceState& state);
  void compositor(Device* other);

  void activateContext() const ;
  void synchronizeStream() const;
  void render(const unsigned int iterationIndex, void** buffer, const int mode);
  void updateDisplayTexture();
  const void* getOutputBufferHost();

  CUdeviceptr memAlloc(const size_t size, const size_t alignment, const cuda::Usage usage = cuda::USAGE_STATIC);
  void memFree(const CUdeviceptr ptr);

  size_t getMemoryFree() const;
  size_t getMemoryAllocated() const;

  Texture* initTexture(const std::string& name, const Picture* picture, const unsigned int flags);
  void shareTexture(const std::string & name, const Texture* shared);

  unsigned int appendProgramGroupMDL(const int indexModule, const std::string& nameFunction);

  void compileMaterial(mi::neuraylib::ITransaction* transaction,
                       MaterialMDL* material,
                       const Compile_result& res,
                       const ShaderConfiguration& config);

  const TextureMDLHost* prepareTextureMDL(mi::neuraylib::ITransaction* transaction,
                                          mi::base::Handle<mi::neuraylib::IImage_api> image_api,
                                          char const* texture_db_name,
                                          mi::neuraylib::ITarget_code::Texture_shape texture_shape);
  void shareTextureMDL(const TextureMDLHost* shared,
                       char const* texture_db_name,
                       mi::neuraylib::ITarget_code::Texture_shape texture_shape);

  MbsdfHost* prepareMBSDF(mi::neuraylib::ITransaction* transaction,
                          const mi::neuraylib::ITarget_code* code,
                          const int index);
  void shareMBSDF(const MbsdfHost* shared);

  LightprofileHost* prepareLightprofile(mi::neuraylib::ITransaction* transaction,
                                        const mi::neuraylib::ITarget_code* code,
                                        int index);
  void shareLightprofile(const LightprofileHost* shared);

  void initTextureHandler(std::vector<MaterialMDL*>& materialsMDL);

private:
  OptixResult initFunctionTable();
  void initDeviceAttributes();
  void initDeviceProperties();
  void initPipeline();

  bool prepare_mbsdfs_part(mi::neuraylib::Mbsdf_part part,
                           MbsdfHost& host,
                           const mi::neuraylib::IBsdf_measurement* bsdf_measurement);

public:
  // Constructor arguments:
  int          m_ordinal; // The ordinal number of this CUDA device.
  int          m_index;   // The index inside the m_devicesActive vector.
  int          m_count;   // The number of active devices.
  TypeLight    m_typeEnv; // Type of environment miss shader to use.
  int          m_interop; // The interop mode to use.
  unsigned int m_tex;     // The OpenGL HDR texture object.
  unsigned int m_pbo;     // The OpenGL PixelBufferObject handle when interop should be used. 0 when not.

  float m_clockFactor; // Clock Factor scaled by CLOCK_FACTOR_SCALE (1.0e-9f) for USE_TIME_VIEW

  CUuuid m_deviceUUID;
  
  // Not actually used because this only works under Windows WDDM mode, not in TCC mode!
  // (Though using LUID is recommended under Windows when possible because you can potentially
  // get the same UUID for MS hybrid configurations (like when running under Remote Desktop).
  char         m_deviceLUID[8];
  unsigned int m_nodeMask;

  std::string m_deviceName;
  std::string m_devicePciBusId;  // domain:bus:device.function, required to find matching CUDA device via NVML.

  DeviceAttribute m_deviceAttribute; // CUDA 
  DeviceProperty  m_deviceProperty;  // OptiX

  CUcontext m_cudaContext;
  CUstream  m_cudaStream;
  
  OptixFunctionTable m_api;
  OptixDeviceContext m_optixContext;
  
  std::vector<std::string> m_moduleFilenames;

  // These are used in different Device functions. Set them once inside the constructor.
  OptixModuleCompileOptions   m_mco;
  OptixPipelineCompileOptions m_pco;
  OptixPipelineLinkOptions    m_plo;
  OptixProgramGroupOptions    m_pgo; // This is a just placeholder.

  OptixPipeline m_pipeline;
  
  OptixShaderBindingTable m_sbt;
  
  CUdeviceptr m_d_sbtRecordHeaders;
 
  CUdeviceptr m_d_ias;

  std::vector<OptixInstance> m_instances;

  std::vector<InstanceData>  m_instanceData; // idGeometry, idMaterial, idLight, idObject

  std::vector<GeometryInstanceData> m_geometryInstanceData;
  GeometryInstanceData*             m_d_geometryInstanceData;
  
  SystemData  m_systemData;   // This contains the root traversable handle as well.
  SystemData* m_d_systemData; // Device side CUdeviceptr of the system data.

  std::vector<int> m_subFrames; // A host array with all sub-frame indices, used to update the device side sysData.iterationIndex fully asynchronously.
  
  int m_launchWidth;
  
  bool m_isDirtySystemData;
  bool m_isDirtyOutputBuffer;
  bool m_ownsSharedBuffer; // DEBUG This flag is only evaluated in asserts.

  std::map<std::string, Texture*> m_mapTextures;

  std::vector<LightDefinition> m_lights; // Staging data for the device side sysData.lightDefinitions
  
  CUgraphicsResource  m_cudaGraphicsResource; // The handle for the registered OpenGL PBO or texture image resource when using interop.

  CUmodule    m_moduleCompositor;
  CUfunction  m_functionCompositor;
  CUdeviceptr m_d_compositorData;

  std::vector<float4> m_bufferHost;

  cuda::ArenaAllocator* m_allocator;

  size_t m_sizeMemoryTextureArrays; // The sum of all texture CUarray resp. CUmipmappedArray.

  // The modules and device shader configurations are both indexed by the shader index.
  std::vector<OptixModule>               m_modulesMDL;
  std::vector<DeviceShaderConfiguration> m_deviceShaderConfigurations;
  
  // These are all direct callable program groups initialized by all different MDL shaders in the scene.
  // (The shader configuration idxCall* fields are indexing into the full set of program groups though,
  // already including the lens shader plus light sampling count offset.)
  std::vector<OptixProgramGroup> m_programGroupsMDL;

  // The Device needs to track these to be able to get the shader index from a material index
  // which is only stored in the MaterialDefinitionMDL struct on device side.
  std::vector<MaterialDefinitionMDL> m_materialDefinitions; // One element per material reference, indexed by indexMaterial.

  // MDL Resource handling, textures, measured BSDFs, light profiles.
  // All different resources (textures, MBSDFs, light profiles) used by the MDL materials inside the scene are stored per device.
  // The MaterialMDL only holds vectors with indices into these caches. That index is the same on each device.
  // It gets expanded into the Texture_handler structure in Device::initTextureHandler().

  std::map<std::string, int>  m_mapTextureNameToIndex; // Texture cache.
  std::vector<TextureMDLHost> m_textureMDLHosts;       // All textures used by the materials inside the scene.

  // There is no cache implemented for measured BSDFs and light profiles.
  // The assumption is that measured BSDFs or light profiles are not used by multiple material shaders.
  std::vector<MbsdfHost> m_mbsdfHosts; // All MBSDFs used by the materials inside the scene.

  std::vector<LightprofileHost> m_lightprofileHosts; // All light profiles used by materials inside the scene
}; 

#endif // DEVICE_H
