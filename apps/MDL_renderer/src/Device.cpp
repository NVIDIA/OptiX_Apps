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

#include "inc/Device.h"

#include "inc/CheckMacros.h"

#include "shaders/compositor_data.h"


#ifdef _WIN32
#if !defined WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>
// The cfgmgr32 header is necessary for interrogating driver information in the registry.
#include <cfgmgr32.h>
// For convenience the library is also linked in automatically using the #pragma command.
#pragma comment(lib, "Cfgmgr32.lib")
#else
#include <dlfcn.h>
#endif

#include <GL/glew.h>
#if defined( _WIN32 )
#include <GL/wglew.h>
#endif

// CUDA Driver API version of the OpenGL interop header. 
#include <cudaGL.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string.h>

#ifdef _WIN32
// Original code from optix_stubs.h
static void* optixLoadWindowsDll(void)
{
  const char* optixDllName = "nvoptix.dll";
  void* handle = NULL;

  // Get the size of the path first, then allocate
  unsigned int size = GetSystemDirectoryA(NULL, 0);
  if (size == 0)
  {
    // Couldn't get the system path size, so bail
    return NULL;
  }

  size_t pathSize = size + 1 + strlen(optixDllName);
  char*  systemPath = (char*) malloc(pathSize);

  if (GetSystemDirectoryA(systemPath, size) != size - 1)
  {
    // Something went wrong
    free(systemPath);
    return NULL;
  }

  strcat(systemPath, "\\");
  strcat(systemPath, optixDllName);

  handle = LoadLibraryA(systemPath);

  free(systemPath);

  if (handle)
  {
    return handle;
  }

  // If we didn't find it, go looking in the register store.  Since nvoptix.dll doesn't
  // have its own registry entry, we are going to look for the OpenGL driver which lives
  // next to nvoptix.dll. 0 (null) will be returned if any errors occured.

  static const char* deviceInstanceIdentifiersGUID = "{4d36e968-e325-11ce-bfc1-08002be10318}";
  const ULONG        flags = CM_GETIDLIST_FILTER_CLASS | CM_GETIDLIST_FILTER_PRESENT;
  ULONG              deviceListSize = 0;

  if (CM_Get_Device_ID_List_SizeA(&deviceListSize, deviceInstanceIdentifiersGUID, flags) != CR_SUCCESS)
  {
    return NULL;
  }

  char* deviceNames = (char*) malloc(deviceListSize);

  if (CM_Get_Device_ID_ListA(deviceInstanceIdentifiersGUID, deviceNames, deviceListSize, flags))
  {
    free(deviceNames);
    return NULL;
  }

  DEVINST devID = 0;

  // Continue to the next device if errors are encountered.
  for (char* deviceName = deviceNames; *deviceName; deviceName += strlen(deviceName) + 1)
  {
    if (CM_Locate_DevNodeA(&devID, deviceName, CM_LOCATE_DEVNODE_NORMAL) != CR_SUCCESS)
    {
      continue;
    }

    HKEY regKey = 0;
    if (CM_Open_DevNode_Key(devID, KEY_QUERY_VALUE, 0, RegDisposition_OpenExisting, &regKey, CM_REGISTRY_SOFTWARE) != CR_SUCCESS)
    {
      continue;
    }

    const char* valueName = "OpenGLDriverName";
    DWORD       valueSize = 0;

    LSTATUS     ret = RegQueryValueExA(regKey, valueName, NULL, NULL, NULL, &valueSize);
    if (ret != ERROR_SUCCESS)
    {
      RegCloseKey(regKey);
      continue;
    }

    char* regValue = (char*) malloc(valueSize);
    ret = RegQueryValueExA(regKey, valueName, NULL, NULL, (LPBYTE) regValue, &valueSize);
    if (ret != ERROR_SUCCESS)
    {
      free(regValue);
      RegCloseKey(regKey);
      continue;
    }

    // Strip the OpenGL driver dll name from the string then create a new string with
    // the path and the nvoptix.dll name
    for (int i = valueSize - 1; i >= 0 && regValue[i] != '\\'; --i)
    {
      regValue[i] = '\0';
    }

    size_t newPathSize = strlen(regValue) + strlen(optixDllName) + 1;
    char*  dllPath = (char*) malloc(newPathSize);
    strcpy(dllPath, regValue);
    strcat(dllPath, optixDllName);

    free(regValue);
    RegCloseKey(regKey);

    handle = LoadLibraryA((LPCSTR) dllPath);
    free(dllPath);

    if (handle)
    {
      break;
    }
  }

  free(deviceNames);

  return handle;
}
#endif


// Global logger function instead of the Logger class to be able to submit per device date via the cbdata pointer.
static std::mutex g_mutexLogger;

static void callbackLogger(unsigned int level, const char* tag, const char* message, void* cbdata)
{
  std::lock_guard<std::mutex> lock(g_mutexLogger);

  Device* device = static_cast<Device*>(cbdata);

  std::cerr << tag  << " (" << level << ") [" << device->m_ordinal << "]: " << ((message) ? message : "(no message)") << '\n';
}


static std::vector<char> readData(std::string const& filename)
{
  std::ifstream inputData(filename, std::ios::binary);

  if (inputData.fail())
  {
    std::cerr << "ERROR: readData() Failed to open file " << filename << '\n';
    return std::vector<char>();
  }

  // Copy the input buffer to a char vector.
  std::vector<char> data(std::istreambuf_iterator<char>(inputData), {});

  if (inputData.fail())
  {
    std::cerr << "ERROR: readData() Failed to read file " << filename << '\n';
    return std::vector<char>();
  }

  return data;
}


Device::Device(const int ordinal,
               const int index,
               const int count,
               const TypeLight typeEnv,
               const int interop,
               const unsigned int tex,
               const unsigned int pbo, 
               const size_t sizeArena)
: m_ordinal(ordinal)
, m_index(index)
, m_count(count)
, m_typeEnv(typeEnv)
, m_interop(interop)
, m_tex(tex)
, m_pbo(pbo)
, m_nodeMask(0)
, m_launchWidth(0)
, m_ownsSharedBuffer(false)
, m_d_compositorData(0)
, m_cudaGraphicsResource(nullptr)
, m_sizeMemoryTextureArrays(0)
{
  initDeviceAttributes(); // CUDA

  OPTIX_CHECK( initFunctionTable() );

  // Create a CUDA Context and make it current to this thread.
  // PERF What is the best CU_CTX_SCHED_* setting here?
  // CU_CTX_MAP_HOST host to allow pinned memory.
  CU_CHECK( cuCtxCreate(&m_cudaContext, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, ordinal) ); 

  // PERF To make use of asynchronous copies. Currently not really anything happening in parallel due to synchronize calls.
  CU_CHECK( cuStreamCreate(&m_cudaStream, CU_STREAM_NON_BLOCKING) ); 

  size_t sizeFree  = 0;
  size_t sizeTotal = 0;

  CU_CHECK( cuMemGetInfo(&sizeFree, &sizeTotal) );

  std::cout << "Device ordinal " << m_ordinal << ": " << sizeFree << " bytes free; " << sizeTotal << " bytes total\n";

  m_allocator = new cuda::ArenaAllocator(sizeArena * 1024 * 1024); // The ArenaAllocator gets the default Arena size in bytes!

#if 1
  // UUID works under Windows and Linux.
  memset(&m_deviceUUID, 0, 16);
  CU_CHECK( cuDeviceGetUuid(&m_deviceUUID, m_ordinal) );
#else
  // LUID only works under Windows and only in WDDM mode, not in TCC mode!
  // Get the LUID and node mask to be able to determine which device needs to allocate the peer-to-peer staging buffer for the OpenGL interop PBO.
  memset(m_deviceLUID, 0, 8);
  CU_CHECK( cuDeviceGetLuid(m_deviceLUID, &m_nodeMask, m_ordinal) );
#endif

  // FIXME Only load this on the primary device.
  CU_CHECK( cuModuleLoad(&m_moduleCompositor, "./MDL_renderer_core/compositor.ptx") );
  CU_CHECK( cuModuleGetFunction(&m_functionCompositor, m_moduleCompositor, "compositor") );

  OptixDeviceContextOptions options = {};

  options.logCallbackFunction = &callbackLogger;
  options.logCallbackData     = this; // This allows per device logs. It's currently printing the device ordinal.
  options.logCallbackLevel    = 3;    // Keep at warning level to suppress the disk cache messages.

  OPTIX_CHECK( m_api.optixDeviceContextCreate(m_cudaContext, &options, &m_optixContext) );

  initDeviceProperties(); // OptiX

  m_d_systemData = reinterpret_cast<SystemData*>(memAlloc(sizeof(SystemData), 16)); // Currently 8 would be enough.

  m_isDirtySystemData = true; // Trigger SystemData update before the next launch.

  // Initialize all renderer system data.
  //m_systemData.rect                = make_int4(0, 0, 1, 1); // Unused, this is not a tiled renderer.
  m_systemData.topObject              = 0;
  m_systemData.outputBuffer           = 0; // Deferred allocation. Only done in render() of the derived Device classes to allow for different memory spaces!
  m_systemData.tileBuffer             = 0; // For the final frame tiled renderer the intermediate buffer is only tileSize.
  m_systemData.texelBuffer            = 0; // For the final frame tiled renderer. Contains the accumulated result of the current tile.
  m_systemData.geometryInstanceData   = nullptr;
  m_systemData.cameraDefinitions      = nullptr;
  m_systemData.lightDefinitions       = nullptr;
  m_systemData.materialDefinitionsMDL = nullptr;  // The MDL material parameter argument block, texture handler and index into the shader.
  m_systemData.shaderConfigurations   = nullptr;    // Indexed by MaterialDefinitionMDL::indexShader.
  m_systemData.resolution             = make_int2(1, 1); // Deferred allocation after setResolution() when m_isDirtyOutputBuffer == true.
  m_systemData.tileSize               = make_int2(8, 8); // Default value for multi-GPU tiling. Must be power-of-two values. (8x8 covers either 8x4 or 4x8 internal 2D warp shapes.)
  m_systemData.tileShift              = make_int2(3, 3); // The right-shift for the division by tileSize. 
  m_systemData.pathLengths            = make_int2(2, 5); // min, max
  m_systemData.walkLength             = 1;
  m_systemData.deviceCount            = m_count; // The number of active devices.
  m_systemData.deviceIndex            = m_index; // This allows to distinguish multiple devices.
  m_systemData.iterationIndex         = 0;
  m_systemData.samplesSqrt            = 0; // Invalid value! Enforces that there is at least one setState() call before rendering.
  m_systemData.sceneEpsilon           = 500.0f * SCENE_EPSILON_SCALE;
  m_systemData.clockScale             = 1000.0f * CLOCK_FACTOR_SCALE;
  m_systemData.typeLens               = 0;
  m_systemData.numCameras             = 0;
  m_systemData.numLights              = 0;
  m_systemData.numMaterials           = 0;
  m_systemData.directLighting         = 1;

  m_isDirtyOutputBuffer = true; // First render call initializes it. This is done in the derived render() functions.

  m_moduleFilenames.resize(NUM_MODULE_IDENTIFIERS);

  // Starting with OptiX SDK 7.5.0 and CUDA 11.7 either PTX or OptiX IR input can be used to create modules.
  // Just initialize the m_moduleFilenames depending on the definition of USE_OPTIX_IR.
  // That is added to the project definitions inside the CMake script when OptiX SDK 7.5.0 and CUDA 11.7 or newer are found.
#if defined(USE_OPTIX_IR)
  m_moduleFilenames[MODULE_ID_RAYGENERATION]  = std::string("./MDL_renderer_core/raygeneration.optixir");
  m_moduleFilenames[MODULE_ID_EXCEPTION]      = std::string("./MDL_renderer_core/exception.optixir");
  m_moduleFilenames[MODULE_ID_MISS]           = std::string("./MDL_renderer_core/miss.optixir");
  m_moduleFilenames[MODULE_ID_HIT]            = std::string("./MDL_renderer_core/hit.optixir");
  m_moduleFilenames[MODULE_ID_LENS_SHADER]    = std::string("./MDL_renderer_core/lens_shader.optixir");
  m_moduleFilenames[MODULE_ID_LIGHT_SAMPLE]   = std::string("./MDL_renderer_core/light_sample.optixir");
#else
  m_moduleFilenames[MODULE_ID_RAYGENERATION]  = std::string("./MDL_renderer_core/raygeneration.ptx");
  m_moduleFilenames[MODULE_ID_EXCEPTION]      = std::string("./MDL_renderer_core/exception.ptx");
  m_moduleFilenames[MODULE_ID_MISS]           = std::string("./MDL_renderer_core/miss.ptx");
  m_moduleFilenames[MODULE_ID_HIT]            = std::string("./MDL_renderer_core/hit.ptx");
  m_moduleFilenames[MODULE_ID_LENS_SHADER]    = std::string("./MDL_renderer_core/lens_shader.ptx");
  m_moduleFilenames[MODULE_ID_LIGHT_SAMPLE]   = std::string("./MDL_renderer_core/light_sample.ptx");
#endif

  // OptixModuleCompileOptions
  m_mco = {};

  m_mco.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#if USE_DEBUG_EXCEPTIONS
  m_mco.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0; // No optimizations.
  m_mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;     // Full debug. Never profile kernels with this setting!
#else
  m_mco.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3; // All optimizations, is the default.
  // Keep generated line info for Nsight Compute profiling. (NVCC_OPTIONS use --generate-line-info in CMakeLists.txt)
#if (OPTIX_VERSION >= 70400)
  m_mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL; 
#else
  m_mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif
#endif // USE_DEBUG_EXCEPTIONS

  // OptixPipelineCompileOptions
  m_pco = {};

  m_pco.usesMotionBlur        = 0;
  m_pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
  m_pco.numPayloadValues      = 2;  // I need two to encode a 64-bit pointer to the per ray payload structure.
  m_pco.numAttributeValues    = 2;  // The minimum is two for the triangle barycentrics.
#if USE_DEBUG_EXCEPTIONS
  m_pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
                         OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                         OPTIX_EXCEPTION_FLAG_USER |
                         OPTIX_EXCEPTION_FLAG_DEBUG;
#else
  m_pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
  m_pco.pipelineLaunchParamsVariableName = "sysData";
#if (OPTIX_VERSION != 70000)
  // New in OptiX 7.1.0.
  // This renderer supports triangles and cubic B-splines.
  m_pco.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE;
#endif

  // OptixPipelineLinkOptions
  m_plo = {};

  m_plo.maxTraceDepth = 2;
#if (OPTIX_VERSION < 70700)
  // OptixPipelineLinkOptions debugLevel is only present in OptiX SDK versions before 7.7.0.
  #if USE_DEBUG_EXCEPTIONS
    m_plo.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL; // Full debug. Never profile kernels with this setting!
  #else
    // Keep generated line info for Nsight Compute profiling. (NVCC_OPTIONS use --generate-line-info in CMakeLists.txt)
    #if (OPTIX_VERSION >= 70400)
      m_plo.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL; 
    #else
      m_plo.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
    #endif
  #endif // USE_DEBUG_EXCEPTIONS
#endif // 70700
#if (OPTIX_VERSION == 70000)
  m_plo.overrideUsesMotionBlur = 0; // Does not exist in OptiX 7.1.0.
#endif

  // OptixProgramGroupOptions
  m_pgo = {}; // This is a just placeholder.
}


Device::~Device()
{
  CU_CHECK_NO_THROW( cuCtxSetCurrent(m_cudaContext) ); // Activate this CUDA context. Not using activate() because this needs a no-throw check.
  CU_CHECK_NO_THROW( cuCtxSynchronize() );             // Make sure everthing running on this CUDA context has finished.

  if (m_cudaGraphicsResource != nullptr)
  {
    CU_CHECK_NO_THROW( cuGraphicsUnregisterResource(m_cudaGraphicsResource) );
  }

  CU_CHECK_NO_THROW( cuModuleUnload(m_moduleCompositor) );

  for (std::map<std::string, Texture*>::const_iterator it = m_mapTextures.begin(); it != m_mapTextures.end(); ++it)
  {
    if (it->second)
    {
      // The texture array data might be owned by a peer device. 
      // Explicitly destroy only the parts which belong to this device.
      m_sizeMemoryTextureArrays -= it->second->destroy(this);

      delete it->second; // This will delete the CUtexObject which exists per device.
    }
  }
  
  // Destroy MDL CUDA Resources which are not allocated by the arena allocator.
  for (TextureMDLHost& host : m_textureMDLHosts) 
  {
    if (host.m_texture.filtered_object)
    {
      CU_CHECK_NO_THROW( cuTexObjectDestroy(host.m_texture.filtered_object) );
    }
    if (host.m_texture.unfiltered_object)
    {
      CU_CHECK_NO_THROW( cuTexObjectDestroy(host.m_texture.unfiltered_object) );
    }
    // Only destroy the CUarray data if the current device is the owner.
    if (this == host.m_owner)
    {
      CU_CHECK_NO_THROW( cuArrayDestroy(host.m_d_array) );
      m_sizeMemoryTextureArrays -= host.m_sizeBytesArray;
    }
  }

  for (MbsdfHost& host : m_mbsdfHosts) 
  {
    for (int i = 0; i < 2; ++i)
    {
      if (host.m_mbsdf.eval_data[i])
      {
        CU_CHECK_NO_THROW( cuTexObjectDestroy(host.m_mbsdf.eval_data[i]) );
      }
      
      if (this == host.m_owner && host.m_d_array[i])
      {
        CU_CHECK_NO_THROW( cuArrayDestroy(host.m_d_array[i]) );
        m_sizeMemoryTextureArrays -= host.m_sizeBytesArray[i];
      }
    }
  }
  
  for (LightprofileHost& host : m_lightprofileHosts) 
  {
    if (host.m_profile.eval_data)
    {
      CU_CHECK_NO_THROW( cuTexObjectDestroy(host.m_profile.eval_data) );
    }
      
    if (this == host.m_owner && host.m_d_array)
    {
      CU_CHECK_NO_THROW( cuArrayDestroy(host.m_d_array) );
      m_sizeMemoryTextureArrays -= host.m_sizeBytesArray;
    }
  }
  
  MY_ASSERT(m_sizeMemoryTextureArrays == 0); // Make sure the texture memory tracking is correct.

  OPTIX_CHECK_NO_THROW( m_api.optixPipelineDestroy(m_pipeline) );
  OPTIX_CHECK_NO_THROW( m_api.optixDeviceContextDestroy(m_optixContext) );

  delete m_allocator; // This frees all CUDA allocations done with the arena allocator!

  CU_CHECK_NO_THROW( cuStreamDestroy(m_cudaStream) );
  CU_CHECK_NO_THROW( cuCtxDestroy(m_cudaContext) );
}


void Device::initDeviceAttributes()
{
  char buffer[1024];
  buffer[1023] = 0;

  CU_CHECK( cuDeviceGetName(buffer, 1023, m_ordinal) );
  m_deviceName = std::string(buffer);

  CU_CHECK(cuDeviceGetPCIBusId(buffer, 1023, m_ordinal));
  m_devicePciBusId = std::string(buffer);

  std::cout << "Device ordinal " << m_ordinal << " at index " << m_index << ": " << m_deviceName << " visible\n";

  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maxBlockDimX, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maxBlockDimY, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maxBlockDimZ, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maxGridDimX, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maxGridDimY, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maxGridDimZ, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maxSharedMemoryPerBlock, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.sharedMemoryPerBlock, CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.totalConstantMemory, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maxPitch, CU_DEVICE_ATTRIBUTE_MAX_PITCH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maxRegistersPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.registersPerBlock, CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.textureAlignment, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.gpuOverlap, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.multiprocessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.kernelExecTimeout, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.canMapHostMemory, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture1dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture2dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture2dHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture3dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture3dHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture3dDepth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture2dLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture2dLayeredHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture2dLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture2dArrayWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture2dArrayHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture2dArrayNumslices, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.surfaceAlignment, CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.concurrentKernels, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.eccEnabled, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.pciBusId, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.pciDeviceId, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.tccDriver, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.memoryClockRate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.globalMemoryBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.l2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maxThreadsPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.asyncEngineCount, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.unifiedAddressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture1dLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture1dLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.canTex2dGather, CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture2dGatherWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture2dGatherHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture3dWidthAlternate, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture3dHeightAlternate, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture3dDepthAlternate, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.pciDomainId, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.texturePitchAlignment, CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexturecubemapWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexturecubemapLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexturecubemapLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumSurface1dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumSurface2dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumSurface2dHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumSurface3dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumSurface3dHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumSurface3dDepth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumSurface1dLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumSurface1dLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumSurface2dLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumSurface2dLayeredHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumSurface2dLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumSurfacecubemapWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumSurfacecubemapLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumSurfacecubemapLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture1dLinearWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture2dLinearWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture2dLinearHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture2dLinearPitch, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture2dMipmappedWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture2dMipmappedHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maximumTexture1dMipmappedWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.streamPrioritiesSupported, CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.globalL1CacheSupported, CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.localL1CacheSupported, CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maxSharedMemoryPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maxRegistersPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.managedMemory, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.multiGpuBoard, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.multiGpuBoardGroupId, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.hostNativeAtomicSupported, CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.singleToDoublePrecisionPerfRatio, CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.pageableMemoryAccess, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.concurrentManagedAccess, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.computePreemptionSupported, CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.canUseHostPointerForRegisteredMem, CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.canUse64BitStreamMemOps, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.canUseStreamWaitValueNor, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.cooperativeLaunch, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.cooperativeMultiDeviceLaunch, CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.maxSharedMemoryPerBlockOptin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.canFlushRemoteWrites, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.hostRegisterSupported, CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.pageableMemoryAccessUsesHostPageTables, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, m_ordinal) );
  CU_CHECK( cuDeviceGetAttribute(&m_deviceAttribute.directManagedMemAccessFromHost, CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST, m_ordinal) );
}

void Device::initDeviceProperties()
{
  OPTIX_CHECK( m_api.optixDeviceContextGetProperty(m_optixContext, OPTIX_DEVICE_PROPERTY_RTCORE_VERSION, &m_deviceProperty.rtcoreVersion, sizeof(unsigned int)) );
  OPTIX_CHECK( m_api.optixDeviceContextGetProperty(m_optixContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH, &m_deviceProperty.limitMaxTraceDepth, sizeof(unsigned int)) );
  OPTIX_CHECK( m_api.optixDeviceContextGetProperty(m_optixContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH, &m_deviceProperty.limitMaxTraversableGraphDepth, sizeof(unsigned int)) );
  OPTIX_CHECK( m_api.optixDeviceContextGetProperty(m_optixContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS, &m_deviceProperty.limitMaxPrimitivesPerGas, sizeof(unsigned int)) );
  OPTIX_CHECK( m_api.optixDeviceContextGetProperty(m_optixContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS, &m_deviceProperty.limitMaxInstancesPerIas, sizeof(unsigned int)) );
  OPTIX_CHECK( m_api.optixDeviceContextGetProperty(m_optixContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID, &m_deviceProperty.limitMaxInstanceId, sizeof(unsigned int)) );
  OPTIX_CHECK( m_api.optixDeviceContextGetProperty(m_optixContext, OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK, &m_deviceProperty.limitNumBitsInstanceVisibilityMask, sizeof(unsigned int)) );
  OPTIX_CHECK( m_api.optixDeviceContextGetProperty(m_optixContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS, &m_deviceProperty.limitMaxSbtRecordsPerGas, sizeof(unsigned int)) );
  OPTIX_CHECK( m_api.optixDeviceContextGetProperty(m_optixContext, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET, &m_deviceProperty.limitMaxSbtOffset, sizeof(unsigned int)) );

#if 0
  std::cout << "OPTIX_DEVICE_PROPERTY_RTCORE_VERSION                          = " << m_deviceProperty.rtcoreVersion                      << '\n';
  std::cout << "OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH                   = " << m_deviceProperty.limitMaxTraceDepth                 << '\n';
  std::cout << "OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH       = " << m_deviceProperty.limitMaxTraversableGraphDepth      << '\n';
  std::cout << "OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS            = " << m_deviceProperty.limitMaxPrimitivesPerGas           << '\n';
  std::cout << "OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS             = " << m_deviceProperty.limitMaxInstancesPerIas            << '\n';
  std::cout << "OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID                   = " << m_deviceProperty.limitMaxInstanceId                 << '\n';
  std::cout << "OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK = " << m_deviceProperty.limitNumBitsInstanceVisibilityMask << '\n';
  std::cout << "OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS           = " << m_deviceProperty.limitMaxSbtRecordsPerGas           << '\n';
  std::cout << "OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET                    = " << m_deviceProperty.limitMaxSbtOffset                  << '\n';
#endif
}


OptixResult Device::initFunctionTable()
{
#ifdef _WIN32
  void* handle = optixLoadWindowsDll();
  if (!handle)
  {
    return OPTIX_ERROR_LIBRARY_NOT_FOUND;
  }

  void* symbol = reinterpret_cast<void*>(GetProcAddress((HMODULE) handle, "optixQueryFunctionTable"));
  if (!symbol)
  {
    return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
  }
#else
  void* handle = dlopen("libnvoptix.so.1", RTLD_NOW);
  if (!handle)
  {
    return OPTIX_ERROR_LIBRARY_NOT_FOUND;
  }

  void* symbol = dlsym(handle, "optixQueryFunctionTable");
  if (!symbol)

  {
    return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
  }
#endif

  OptixQueryFunctionTable_t* optixQueryFunctionTable = reinterpret_cast<OptixQueryFunctionTable_t*>(symbol);

  return optixQueryFunctionTable(OPTIX_ABI_VERSION, 0, 0, 0, &m_api, sizeof(OptixFunctionTable));
}


void Device::initPipeline()
{
  // This functin needs to be called after all MDL materials have been built,
  // because only then all callable programs are present and can be compiled into the pipeline.

  MY_ASSERT(NUM_RAY_TYPES == 2); // The following code only works for two raytypes.

  // Each source file results in one OptixModule.
  std::vector<OptixModule> modules(NUM_MODULE_IDENTIFIERS);

  // Create all modules:
  for (size_t i = 0; i < m_moduleFilenames.size(); ++i)
  {
    // Since OptiX 7.5.0 the program input can either be *.ptx source code or *.optixir binary code.
    // The module filenames are automatically switched between *.ptx or *.optixir extension based on the definition of USE_OPTIX_IR
    std::vector<char> programData = readData(m_moduleFilenames[i]);

#if (OPTIX_VERSION >= 70700)
    OPTIX_CHECK( m_api.optixModuleCreate(m_optixContext, &m_mco, &m_pco, programData.data(), programData.size(), nullptr, nullptr, &modules[i]) );
#else
    OPTIX_CHECK( m_api.optixModuleCreateFromPTX(m_optixContext, &m_mco, &m_pco, programData.data(), programData.size(), nullptr, nullptr, &modules[i]) );
#endif
  }

  // Get the OptiX internal module with the intersection program for cubic B-spline curves;
  OptixBuiltinISOptions builtinISOptions = {};

  builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
  builtinISOptions.usesMotionBlur      = 0;
  
  OptixModule moduleIntersectionCubicCurves;

  OPTIX_CHECK( m_api.optixBuiltinISModuleGet(m_optixContext, &m_mco, &m_pco, &builtinISOptions, &moduleIntersectionCubicCurves) );

  std::vector<OptixProgramGroupDesc> programGroupDescriptions(NUM_PROGRAM_GROUP_IDS);
  memset(programGroupDescriptions.data(), 0, sizeof(OptixProgramGroupDesc) * programGroupDescriptions.size());
  
  OptixProgramGroupDesc* pgd;

  pgd = &programGroupDescriptions[PGID_RAYGENERATION];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->raygen.module = modules[MODULE_ID_RAYGENERATION];
  if (1 < m_count)
  {
    // Only use the multi-GPU specific raygen program when there are multiple devices enabled.
    pgd->raygen.entryFunctionName = "__raygen__path_tracer_local_copy";
  }
  else
  {
    // Use a single-GPU raygen program which doesn't need compositing.
    pgd->raygen.entryFunctionName = "__raygen__path_tracer";
  }

  pgd = &programGroupDescriptions[PGID_EXCEPTION];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->exception.module            = modules[MODULE_ID_EXCEPTION];
  pgd->exception.entryFunctionName = "__exception__all";

  pgd = &programGroupDescriptions[PGID_MISS_RADIANCE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->miss.module = modules[MODULE_ID_MISS];
  switch (m_typeEnv)
  {
    case TYPE_LIGHT_ENV_CONST:
      pgd->miss.entryFunctionName = "__miss__env_constant";
      break;
    case TYPE_LIGHT_ENV_SPHERE:
      pgd->miss.entryFunctionName = "__miss__env_sphere";
      break;
    default: // Every other ID means there is no environment light, esp. using m_typeEnv == NUM_LIGHT_TYPES for that.
      pgd->miss.entryFunctionName = "__miss__env_null";
      break;
  }

  pgd = &programGroupDescriptions[PGID_MISS_SHADOW];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->miss.module            = nullptr;
  pgd->miss.entryFunctionName = nullptr; // No miss program for shadow rays. 

  // The hit records for the radiance and shadow ray for opaque (instance sbtOffset 0) and cutout opacity (instance sbtOffset 1) hit records.
  // 0 = no emission, no cutout
  pgd = &programGroupDescriptions[PGID_HIT_RADIANCE_0];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleCH            = modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameCH = "__closesthit__radiance_no_emission";

  pgd = &programGroupDescriptions[PGID_HIT_SHADOW_0];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleAH            = modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__shadow";

  // 1 = emission, no cutout
  pgd = &programGroupDescriptions[PGID_HIT_RADIANCE_1];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleCH            = modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameCH = "__closesthit__radiance";

  pgd = &programGroupDescriptions[PGID_HIT_SHADOW_1];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleAH            = modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__shadow";

  // 2 = no emission, cutout
  pgd = &programGroupDescriptions[PGID_HIT_RADIANCE_2];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleCH            = modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameCH = "__closesthit__radiance_no_emission";
  pgd->hitgroup.moduleAH            = modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__radiance_cutout";

  pgd = &programGroupDescriptions[PGID_HIT_SHADOW_2];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleAH            = modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__shadow_cutout";

  // 3 = emission, cutout
  pgd = &programGroupDescriptions[PGID_HIT_RADIANCE_3];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleCH            = modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameCH = "__closesthit__radiance";
  pgd->hitgroup.moduleAH            = modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__radiance_cutout";

  pgd = &programGroupDescriptions[PGID_HIT_SHADOW_3];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleAH            = modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__shadow_cutout";

  pgd = &programGroupDescriptions[PGID_HIT_CURVES];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleCH            = modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameCH = "__closesthit__curves";
  pgd->hitgroup.moduleIS            = moduleIntersectionCubicCurves;
  pgd->hitgroup.entryFunctionNameIS = nullptr; // Uses built-in IS for cubic curves.

  pgd = &programGroupDescriptions[PGID_HIT_CURVES_SHADOW];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleAH            = modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__shadow";
  pgd->hitgroup.moduleIS            = moduleIntersectionCubicCurves;
  pgd->hitgroup.entryFunctionNameIS = nullptr; // Uses built-in IS for cubic curves.

  // CALLABLES
  // Lens Shader
  pgd = &programGroupDescriptions[PGID_LENS_PINHOLE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = modules[MODULE_ID_LENS_SHADER];
  pgd->callables.entryFunctionNameDC = "__direct_callable__pinhole";

  pgd = &programGroupDescriptions[PGID_LENS_FISHEYE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = modules[MODULE_ID_LENS_SHADER];
  pgd->callables.entryFunctionNameDC = "__direct_callable__fisheye";
  
  pgd = &programGroupDescriptions[PGID_LENS_SPHERE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = modules[MODULE_ID_LENS_SHADER];
  pgd->callables.entryFunctionNameDC = "__direct_callable__sphere";

  // Light Sampler
  // Only one of the environment callables will ever be used, but both are required
  // for the proper direct callable index calculation for BXDFs using NUM_LIGHT_TYPES.
  pgd = &programGroupDescriptions[PGID_LIGHT_ENV_CONSTANT];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_env_constant";

  pgd = &programGroupDescriptions[PGID_LIGHT_ENV_SPHERE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_env_sphere";

  pgd = &programGroupDescriptions[PGID_LIGHT_MESH];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = modules[MODULE_ID_HIT]; // Inside the module including texture_support.h.
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_mesh";

  pgd = &programGroupDescriptions[PGID_LIGHT_POINT];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_point";

  pgd = &programGroupDescriptions[PGID_LIGHT_SPOT];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_spot";

  pgd = &programGroupDescriptions[PGID_LIGHT_IES];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_ies";

  std::vector<OptixProgramGroup> programGroups(programGroupDescriptions.size());
  
  OPTIX_CHECK( m_api.optixProgramGroupCreate(m_optixContext, programGroupDescriptions.data(), (unsigned int) programGroupDescriptions.size(), &m_pgo, nullptr, nullptr, programGroups.data()) );

  // Now append all the program groups with the direct callables from the MDL materials.
  programGroups.insert(programGroups.end(), m_programGroupsMDL.begin(), m_programGroupsMDL.end());

  OPTIX_CHECK( m_api.optixPipelineCreate(m_optixContext, &m_pco, &m_plo, programGroups.data(), (unsigned int) programGroups.size(), nullptr, nullptr, &m_pipeline) );

  // STACK SIZES
  OptixStackSizes ssp = {}; // Whole pipeline.

  for (auto pg: programGroups)
  {
    OptixStackSizes ss;

#if (OPTIX_VERSION >= 70700)
    OPTIX_CHECK( m_api.optixProgramGroupGetStackSize(pg, &ss, m_pipeline) );
#else
    OPTIX_CHECK( m_api.optixProgramGroupGetStackSize(pg, &ss) );
#endif

    ssp.cssRG = std::max(ssp.cssRG, ss.cssRG);
    ssp.cssMS = std::max(ssp.cssMS, ss.cssMS);
    ssp.cssCH = std::max(ssp.cssCH, ss.cssCH);
    ssp.cssAH = std::max(ssp.cssAH, ss.cssAH);
    ssp.cssIS = std::max(ssp.cssIS, ss.cssIS);
    ssp.cssCC = std::max(ssp.cssCC, ss.cssCC);
    ssp.dssDC = std::max(ssp.dssDC, ss.dssDC);
  }
  
  // Temporaries
  unsigned int cssCCTree           = ssp.cssCC; // Should be 0. No continuation callables in this pipeline. // maxCCDepth == 0
  unsigned int cssCHOrMSPlusCCTree = std::max(ssp.cssCH, ssp.cssMS) + cssCCTree;
  
  const unsigned int maxDCDepth = 2; // The __direct_callable__light_mesh_mdl calls other direct callables from MDL expressions.

  // Arguments

  unsigned int directCallableStackSizeFromTraversal = ssp.dssDC * maxDCDepth; // FromTraversal: DC is invoked from IS or AH.      // Possible stack size optimizations.
  unsigned int directCallableStackSizeFromState     = ssp.dssDC * maxDCDepth; // FromState:     DC is invoked from RG, MS, or CH. // Possible stack size optimizations.
  unsigned int continuationStackSize = ssp.cssRG + cssCCTree + cssCHOrMSPlusCCTree * (std::max(1u, m_plo.maxTraceDepth) - 1u) +
                                       std::min(1u, m_plo.maxTraceDepth) * std::max(cssCHOrMSPlusCCTree, ssp.cssAH + ssp.cssIS);
  unsigned int maxTraversableGraphDepth = 2;

  OPTIX_CHECK( m_api.optixPipelineSetStackSize(m_pipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState, continuationStackSize, maxTraversableGraphDepth) );

  // Set up the Shader Binding Table (SBT)

  // Put all SbtRecordHeader types in one CUdeviceptr.
  const int numHeaders = static_cast<int>(programGroups.size());

  std::vector<SbtRecordHeader> sbtRecordHeaders(numHeaders);

  for (int i = 0; i < numHeaders; ++i)
  {
    OPTIX_CHECK( m_api.optixSbtRecordPackHeader(programGroups[PGID_RAYGENERATION + i], &sbtRecordHeaders[i]) );
  }

  m_d_sbtRecordHeaders = memAlloc(sizeof(SbtRecordHeader) * numHeaders, OPTIX_SBT_RECORD_ALIGNMENT);
  CU_CHECK( cuMemcpyHtoDAsync(m_d_sbtRecordHeaders, sbtRecordHeaders.data(), sizeof(SbtRecordHeader) * numHeaders, m_cudaStream) );

  // Setup the OptixShaderBindingTable.
  // The order of SBT records match the ProgramGroupId enums.
  m_sbt.raygenRecord = m_d_sbtRecordHeaders + sizeof(SbtRecordHeader) * PGID_RAYGENERATION;

  m_sbt.exceptionRecord = m_d_sbtRecordHeaders + sizeof(SbtRecordHeader) * PGID_EXCEPTION;

  m_sbt.missRecordBase          = m_d_sbtRecordHeaders + sizeof(SbtRecordHeader) * PGID_MISS_RADIANCE;
  m_sbt.missRecordStrideInBytes = (unsigned int) sizeof(SbtRecordHeader);
  m_sbt.missRecordCount         = NUM_RAY_TYPES;

  m_sbt.hitgroupRecordBase          = m_d_sbtRecordHeaders + sizeof(SbtRecordHeader) * PGID_HIT_RADIANCE_0;
  m_sbt.hitgroupRecordStrideInBytes = (unsigned int) sizeof(SbtRecordHeader);
  m_sbt.hitgroupRecordCount         = NUM_RAY_TYPES * 5; // Five hitRecords: 0 to 3 == (no emission, emission) x (no cutout, cutout), and 4 = opaque cubic curves.

  m_sbt.callablesRecordBase          = m_d_sbtRecordHeaders + sizeof(SbtRecordHeader) * PGID_LENS_PINHOLE; // The pinhole lens shader is the first callable.
  m_sbt.callablesRecordStrideInBytes = (unsigned int) sizeof(SbtRecordHeader);
  m_sbt.callablesRecordCount         = static_cast<unsigned int>(programGroups.size()) - PGID_LENS_PINHOLE;

  // After all required optixSbtRecordPackHeader, optixProgramGroupGetStackSize, and optixPipelineCreate
  // calls have been done, the OptixProgramGroup and OptixModule objects can be destroyed.
  for (auto pg: programGroups)
  {
    OPTIX_CHECK( m_api.optixProgramGroupDestroy(pg) );
  }
  // This also destroyed the program groups in m_programGroupsMDL, so these can be cleared.
  m_programGroupsMDL.clear();

  for (auto m : modules)
  {
    OPTIX_CHECK(m_api.optixModuleDestroy(m));
  }
  // Destroy the modules with the MDL generated direct callables which were used to build the m_programGroupsMDL.
  for (auto m : m_modulesMDL)
  {
    OPTIX_CHECK(m_api.optixModuleDestroy(m));
  }
  m_modulesMDL.clear();
}


void Device::initCameras(const std::vector<CameraDefinition>& cameras)
{
  // PERF For simplicity, the public Device functions make sure to set the CUDA context and wait for the previous operation to finish.
  // Faster would be to do that only when needed, which means the caller would be responsible to do the proper synchronization,
  // while the functions themselves work as asynchronously as possible.
  activateContext();
  synchronizeStream();
    
  const int numCameras = static_cast<int>(cameras.size());
  MY_ASSERT(0 < numCameras); // There must be at least one camera defintion or the lens shaders won't work.

  // The default initialization of numCameras is 0.
  if (m_systemData.numCameras != numCameras)
  {
    memFree(reinterpret_cast<CUdeviceptr>(m_systemData.cameraDefinitions));
    m_systemData.cameraDefinitions = reinterpret_cast<CameraDefinition*>(memAlloc(sizeof(CameraDefinition) * numCameras, 16));
  }

  // Update the camera data.
  CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(m_systemData.cameraDefinitions), cameras.data(), sizeof(CameraDefinition) * numCameras, m_cudaStream) );
  m_systemData.numCameras = numCameras;

  m_isDirtySystemData = true;  // Trigger full update of the device system data on the next launch.
}

void Device::initLights(const std::vector<LightGUI>& lightsGUI, const std::vector<GeometryData>& geometryData, const unsigned int stride, const unsigned int index)
{
  activateContext();
  synchronizeStream();

  MY_ASSERT((sizeof(LightDefinition) & 15) == 0); // Verify float4 alignment.

  const int numLights = static_cast<int>(lightsGUI.size()); // This is allowed to be zero.

  // The default initialization of m_systemData.numLights is 0.
  if (m_systemData.numLights != numLights)
  {
    memFree(reinterpret_cast<CUdeviceptr>(m_systemData.lightDefinitions));
    m_systemData.lightDefinitions = nullptr;

    m_systemData.lightDefinitions = (0 < numLights) ? reinterpret_cast<LightDefinition*>(memAlloc(sizeof(LightDefinition) * numLights, 16)) : nullptr;

    m_lights.resize(numLights);
  }

  for (int i = 0; i < numLights; ++i)
  {
    const LightGUI&  lightGUI = lightsGUI[i]; // LightGUI data on the host.
    LightDefinition& light    = m_lights[i];  // LightDefinition data on the host in device layout.

    light.typeLight  = lightGUI.typeLight;
    light.idMaterial = lightGUI.idMaterial;
    light.idObject   = lightGUI.idObject;

    // My device side matrices are row-major left-multiplied and 3x4 for affine transformations.
    // nvpro-pipeline matrices are row-major right-multiplied. operator~() is transpose.
    memcpy(light.matrix,    (~lightGUI.matrix).getPtr(),    sizeof(float) * 12);
    memcpy(light.matrixInv, (~lightGUI.matrixInv).getPtr(), sizeof(float) * 12);

    const dp::math::Mat33f rotation(lightGUI.orientation);
    const dp::math::Mat33f rotationInv(lightGUI.orientationInv);

    memcpy(light.ori,    (~rotation).getPtr(),    sizeof(float) * 9);
    memcpy(light.oriInv, (~rotationInv).getPtr(), sizeof(float) * 9);

    light.attributes      = 0;
    light.indices         = 0;
    light.textureEmission = 0;
    light.textureProfile  = 0;
    light.cdfU            = 0; // 2D, (width  + 1) * height float elements.
    light.cdfV            = 0; // 1D, (height + 1) float elements.
    light.emission        = lightGUI.colorEmission * lightGUI.multiplierEmission;
    light.width           = 0;
    light.height          = 0;
    light.area            = lightGUI.area;
    light.invIntegral     = 1.0f;
    light.spotAngleHalf   = dp::math::degToRad(lightGUI.spotAngle * 0.5f);
    light.spotExponent    = lightGUI.spotExponent;

    if (!lightGUI.nameEmission.empty())
    {
      std::map<std::string, Texture*>::const_iterator it = m_mapTextures.find(lightGUI.nameEmission);
      MY_ASSERT(it != m_mapTextures.end());
      
      const Texture* texture = it->second;

      light.textureEmission = texture->getTextureObject();
      light.cdfU            = texture->getCDF_U();
      light.cdfV            = texture->getCDF_V();
      light.width           = texture->getWidth();
      light.height          = texture->getHeight();
      light.invIntegral     = 1.0f / texture->getIntegral();
    }

    if (light.typeLight == TYPE_LIGHT_MESH)
    {
      const GeometryData& geom = geometryData[lightGUI.idGeometry * stride + index];

      light.attributes = geom.d_attributes;
      light.indices    = geom.d_indices;

      // Allocate and upload the areas and cdf data.
      // Reusing the cdfU field.
      // Note that mesh lights are not importance sampled over the emission texture.
      // They are uniformly sampled over the light surface.
      size_t sizeBytes = sizeof(float) * lightGUI.cdfAreas.size();
      light.cdfU = memAlloc(sizeBytes, 4); 
      CU_CHECK( cuMemcpyHtoDAsync(light.cdfU, lightGUI.cdfAreas.data(), sizeBytes, m_cudaStream) );

      light.width = static_cast<unsigned int>(lightGUI.cdfAreas.size() - 1); // The last element index in the CDF matches the number of triangles.
    }

    if (light.typeLight == TYPE_LIGHT_IES)
    {
      if (!lightGUI.nameProfile.empty())
      {
        std::map<std::string, Texture*>::const_iterator it = m_mapTextures.find(lightGUI.nameProfile);
        MY_ASSERT(it != m_mapTextures.end());
      
        const Texture* texture = it->second;

        light.textureProfile = texture->getTextureObject();
      }
    }
  }

  CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(m_systemData.lightDefinitions), m_lights.data(), sizeof(LightDefinition) * numLights, m_cudaStream) );
  m_systemData.numLights = numLights;

  m_isDirtySystemData = true; // Trigger full update of the device system data on the next launch.
}


void Device::updateCamera(const int idCamera, const CameraDefinition& camera)
{
  activateContext();
  synchronizeStream();

  MY_ASSERT(idCamera < m_systemData.numCameras);
  CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(&m_systemData.cameraDefinitions[idCamera]), &camera, sizeof(CameraDefinition), m_cudaStream) );
}

void Device::updateLight(const int idLight, const LightGUI& lightGUI)
{
  activateContext();
  synchronizeStream();
  
  LightDefinition& light = m_lights[idLight];

  // Curently only these material parameters affecting the light can be changed inside the GUI.
  light.emission        = lightGUI.colorEmission * lightGUI.multiplierEmission;
  light.spotAngleHalf   = dp::math::degToRad(lightGUI.spotAngle * 0.5f);
  light.spotExponent    = lightGUI.spotExponent;

  MY_ASSERT(idLight < m_systemData.numLights);
  CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(&m_systemData.lightDefinitions[idLight]), &light, sizeof(LightDefinition), m_cudaStream) );
}

//void Device::updateLight(const int idLight, const LightDefinition& light)
//{
//  activateContext();
//  synchronizeStream();
//
//  MY_ASSERT(idLight < m_systemData.numLights);
//  CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(&m_systemData.lightDefinitions[idLight]), &light, sizeof(LightDefinition), m_cudaStream) );
//}

void Device::updateMaterial(const int idMaterial, const MaterialMDL* materialMDL)
{
  activateContext();
  synchronizeStream();

  MY_ASSERT(idMaterial < m_materialDefinitions.size());
  
  CU_CHECK( cuMemcpyHtoDAsync(m_materialDefinitions[idMaterial].arg_block,
                              materialMDL->getArgumentBlockData(), 
                              materialMDL->getArgumentBlockSize(),
                              m_cudaStream) );
}

static int2 calculateTileShift(const int2 tileSize)
{
  int xShift = 0; 
  while (xShift < 32 && (tileSize.x & (1 << xShift)) == 0)
  {
    ++xShift;
  }

  int yShift = 0; 
  while (yShift < 32 && (tileSize.y & (1 << yShift)) == 0)
  {
    ++yShift;
  }
  
  MY_ASSERT(xShift < 32 && yShift < 32); // Can only happen for zero input.

  return make_int2(xShift, yShift);
}


void Device::setState(const DeviceState& state)
{
  activateContext();
  synchronizeStream();

  // The system can switch dynamically betweeen brute force path tracing and direct lighting (next event estimation)
  // That's used to compare direct lighting results with the normally correct brute force path tracing at runtime.
  if (m_systemData.directLighting != state.directLighting)
  {
    m_systemData.directLighting = state.directLighting;

    m_isDirtySystemData   = true;
  }

  // Special handling from the previous DeviceMultiGPULocalCopy class.
  if (m_systemData.resolution != state.resolution ||
      m_systemData.tileSize   != state.tileSize)
  {
    if (1 < m_count)
    {
      // Calculate the new launch width for the tiled rendering.
      // It must be a multiple of the tileSize width, otherwise the right-most tiles will not get filled correctly.
      const int width = (state.resolution.x + m_count - 1) / m_count;
      const int mask  = state.tileSize.x - 1;
      m_launchWidth = (width + mask) & ~mask; // == ((width + (tileSize - 1)) / tileSize.x) * tileSize.x;
    }
    else
    {
      // Single-GPU launch width is the same as the rendering resolution width.
      m_launchWidth = state.resolution.x;
    }
  }

  if (m_systemData.resolution != state.resolution)
  {
    m_systemData.resolution = state.resolution;

    m_isDirtyOutputBuffer = true;
    m_isDirtySystemData   = true;
  }

  if (m_systemData.tileSize != state.tileSize)
  {
    m_systemData.tileSize  = state.tileSize;
    m_systemData.tileShift = calculateTileShift(m_systemData.tileSize);
    m_isDirtySystemData = true;
  }

  if (m_systemData.samplesSqrt != state.samplesSqrt)
  {
    m_systemData.samplesSqrt = state.samplesSqrt;
    
    // Update the m_subFrames host index array.
    const int spp = m_systemData.samplesSqrt * m_systemData.samplesSqrt;

    m_subFrames.resize(spp);

    for (int i = 0; i < spp; ++i)
    {
      m_subFrames[i] = i;
    }

    m_isDirtySystemData = true;
  }

  if (m_systemData.typeLens != state.typeLens)
  {
    m_systemData.typeLens = state.typeLens;
    m_isDirtySystemData = true;
  }

  if (m_systemData.pathLengths != state.pathLengths)
  {
    m_systemData.pathLengths = state.pathLengths;
    m_isDirtySystemData = true;
  }

  if (m_systemData.walkLength != state.walkLength)
  {
    m_systemData.walkLength = state.walkLength;
    m_isDirtySystemData = true;
  }
  
  if (m_systemData.sceneEpsilon != state.epsilonFactor * SCENE_EPSILON_SCALE)
  {
    m_systemData.sceneEpsilon = state.epsilonFactor * SCENE_EPSILON_SCALE;
    m_isDirtySystemData = true;
  }

#if USE_TIME_VIEW
  if (m_systemData.clockScale != state.clockFactor * CLOCK_FACTOR_SCALE)
  {
    m_systemData.clockScale = state.clockFactor * CLOCK_FACTOR_SCALE;
    m_isDirtySystemData = true;
  }
#endif
}


GeometryData Device::createGeometry(std::shared_ptr<sg::Triangles> geometry)
{
  activateContext();
  synchronizeStream();

  GeometryData data;

  data.primitiveType = PT_TRIANGLES;
  data.owner         = m_index;

  const std::vector<TriangleAttributes>& attributes = geometry->getAttributes();
  const std::vector<unsigned int>&       indices    = geometry->getIndices();

  const size_t attributesSizeInBytes = sizeof(TriangleAttributes) * attributes.size();
  const size_t indicesSizeInBytes    = sizeof(unsigned int)       * indices.size();

  data.d_attributes = memAlloc(attributesSizeInBytes, 16);
  data.d_indices    = memAlloc(indicesSizeInBytes, sizeof(unsigned int));

  data.numAttributes = attributes.size();
  data.numIndices    = indices.size();
  
  CU_CHECK( cuMemcpyHtoDAsync(data.d_attributes, attributes.data(), attributesSizeInBytes, m_cudaStream) );
  CU_CHECK( cuMemcpyHtoDAsync(data.d_indices,    indices.data(),    indicesSizeInBytes,    m_cudaStream) );

  OptixBuildInput buildInput = {};

  buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

  buildInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
  buildInput.triangleArray.vertexStrideInBytes = sizeof(TriangleAttributes);
  buildInput.triangleArray.numVertices         = static_cast<unsigned int>(attributes.size());
  buildInput.triangleArray.vertexBuffers       = &data.d_attributes;

  buildInput.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  buildInput.triangleArray.indexStrideInBytes = sizeof(unsigned int) * 3;

  buildInput.triangleArray.numIndexTriplets   = static_cast<unsigned int>(indices.size()) / 3;
  buildInput.triangleArray.indexBuffer        = data.d_indices;

  unsigned int inputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

  buildInput.triangleArray.flags         = inputFlags;
  buildInput.triangleArray.numSbtRecords = 1;

  OptixAccelBuildOptions accelBuildOptions = {};

  accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  if (m_count == 1)
  {
    // PERF Enable OPTIX_BUILD_FLAG_PREFER_FAST_TRACE on single-GPU only.
    // Note that OPTIX_BUILD_FLAG_PREFER_FAST_TRACE will use more memory,
    // which performs worse when sharing across the NVLINK bridge which is much slower than VRAM accesses.
    // This means comparisons between single-GPU and multi-GPU are not doing exactly the same!
    accelBuildOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  }
  accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes accelBufferSizes;
  
  OPTIX_CHECK( m_api.optixAccelComputeMemoryUsage(m_optixContext, &accelBuildOptions, &buildInput, 1, &accelBufferSizes) );

  data.d_gas = memAlloc(accelBufferSizes.outputSizeInBytes, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT, cuda::USAGE_TEMP); // This is a temporary buffer. The Compaction will be the static one!

  CUdeviceptr d_tmp = memAlloc(accelBufferSizes.tempSizeInBytes, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT, cuda::USAGE_TEMP);

  OptixAccelEmitDesc accelEmit = {};

  accelEmit.result = memAlloc(sizeof(size_t), sizeof(size_t), cuda::USAGE_TEMP);
  accelEmit.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

  OPTIX_CHECK( m_api.optixAccelBuild(m_optixContext, m_cudaStream, 
                                     &accelBuildOptions, &buildInput, 1,
                                     d_tmp, accelBufferSizes.tempSizeInBytes,
                                     data.d_gas, accelBufferSizes.outputSizeInBytes, 
                                     &data.traversable, &accelEmit, 1) );

  size_t sizeCompact;

  CU_CHECK( cuMemcpyDtoHAsync(&sizeCompact, accelEmit.result, sizeof(size_t), m_cudaStream) );
  
  synchronizeStream();

  memFree(accelEmit.result);
  memFree(d_tmp);

  // Compact the AS only when possible. This can save more than half the memory on RTX boards.
  if (sizeCompact < accelBufferSizes.outputSizeInBytes)
  {
    CUdeviceptr d_gasCompact = memAlloc(sizeCompact, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT); // This is the static GAS allocation!

    OPTIX_CHECK( m_api.optixAccelCompact(m_optixContext, m_cudaStream, data.traversable, d_gasCompact, sizeCompact, &data.traversable) );

    synchronizeStream(); // Must finish accessing data.d_gas source before it can be freed and overridden.

    memFree(data.d_gas);

    data.d_gas = d_gasCompact;

    //std::cout << "Compaction saved " << accelBufferSizes.outputSizeInBytes - sizeCompact << '\n'; // DEBUG
    accelBufferSizes.outputSizeInBytes = sizeCompact; // DEBUG for the std::cout below.
  }

  // Return the relocation info for this GAS traversable handle from this device's OptiX context.
  // It's used to assert that the GAS is compatible across devices which means NVLINK peer-to-peer sharing is allowed.
  // (This is more meant as example code, because in NVLINK islands the GPU configuration must be homogeneous and addresses are unique with UVA.)
  OPTIX_CHECK( m_api.optixAccelGetRelocationInfo(m_optixContext, data.traversable, &data.info) );

  //std::cout << "createGeometry() device index = " << m_index << ": attributes = " << attributesSizeInBytes << ", indices = " << indicesSizeInBytes << ", GAS = " << accelBufferSizes.outputSizeInBytes << "\n"; // DEBUG

  return data;
}


GeometryData Device::createGeometry(std::shared_ptr<sg::Curves> geometry)
{
  activateContext();
  synchronizeStream();

  GeometryData data;

  data.primitiveType = PT_CURVES;
  data.owner         = m_index;

  const std::vector<CurveAttributes>& attributes = geometry->getAttributes();
  const std::vector<unsigned int>&       indices = geometry->getIndices();

  const size_t attributesSizeInBytes = sizeof(CurveAttributes) * attributes.size();

  data.d_attributes  = memAlloc(attributesSizeInBytes, 16);
  data.numAttributes = attributes.size();

  CU_CHECK( cuMemcpyHtoDAsync(data.d_attributes, attributes.data(), attributesSizeInBytes, m_cudaStream) );

  CUdeviceptr d_radii = data.d_attributes + sizeof(float3); // Pointer to the radius in the .w component of the float4 vertex attribute
  
  const size_t indicesSizeInBytes = sizeof(unsigned int) * indices.size();
   
  data.d_indices  = memAlloc(indicesSizeInBytes, sizeof(unsigned int));
  data.numIndices = indices.size();
  
  CU_CHECK( cuMemcpyHtoDAsync(data.d_indices, indices.data(), indicesSizeInBytes, m_cudaStream) );

  OptixBuildInput buildInput = {};

  buildInput.type = OPTIX_BUILD_INPUT_TYPE_CURVES;
  
  buildInput.curveArray.curveType            = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
  buildInput.curveArray.numPrimitives        = static_cast<unsigned int>(indices.size());
  buildInput.curveArray.vertexBuffers        = &data.d_attributes;
  buildInput.curveArray.numVertices          = static_cast<unsigned int>(attributes.size());
  buildInput.curveArray.vertexStrideInBytes  = sizeof(CurveAttributes);
  buildInput.curveArray.widthBuffers         = &d_radii;
  buildInput.curveArray.widthStrideInBytes   = sizeof(CurveAttributes);
  buildInput.curveArray.normalBuffers        = nullptr; // Reserved for future use
  buildInput.curveArray.normalStrideInBytes  = 0;       // Reserved for future use
  buildInput.curveArray.indexBuffer          = data.d_indices;
  buildInput.curveArray.indexStrideInBytes   = sizeof(unsigned int);
  buildInput.curveArray.flag                 = OPTIX_GEOMETRY_FLAG_NONE; // Only one flag because Curves have only one SBT entry.
  buildInput.curveArray.primitiveIndexOffset = 0;

  OptixAccelBuildOptions accelBuildOptions = {};

  accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  if (m_count == 1)
  {
    // PERF Enable OPTIX_BUILD_FLAG_PREFER_FAST_TRACE on single-GPU only.
    // Note that OPTIX_BUILD_FLAG_PREFER_FAST_TRACE will use more memory,
    // which performs worse when sharing across the NVLINK bridge which is much slower than VRAM accesses.
    // This means comparisons between single-GPU and multi-GPU are not doing exactly the same!
    accelBuildOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  }
  accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes accelBufferSizes;
  
  OPTIX_CHECK( m_api.optixAccelComputeMemoryUsage(m_optixContext, &accelBuildOptions, &buildInput, 1, &accelBufferSizes) );

  data.d_gas = memAlloc(accelBufferSizes.outputSizeInBytes, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT, cuda::USAGE_TEMP); // This is a temporary buffer. The Compaction will be the static one!

  CUdeviceptr d_tmp = memAlloc(accelBufferSizes.tempSizeInBytes, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT, cuda::USAGE_TEMP);

  OptixAccelEmitDesc accelEmit = {};

  accelEmit.result = memAlloc(sizeof(size_t), sizeof(size_t), cuda::USAGE_TEMP);
  accelEmit.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

  OPTIX_CHECK( m_api.optixAccelBuild(m_optixContext, m_cudaStream, 
                                     &accelBuildOptions, &buildInput, 1,
                                     d_tmp, accelBufferSizes.tempSizeInBytes,
                                     data.d_gas, accelBufferSizes.outputSizeInBytes, 
                                     &data.traversable, &accelEmit, 1) );

  size_t sizeCompact;

  CU_CHECK( cuMemcpyDtoHAsync(&sizeCompact, accelEmit.result, sizeof(size_t), m_cudaStream) );
  
  synchronizeStream();

  memFree(accelEmit.result);
  memFree(d_tmp);

  // Compact the AS only when possible. This can save more than half the memory on RTX boards.
  if (sizeCompact < accelBufferSizes.outputSizeInBytes)
  {
    CUdeviceptr d_gasCompact = memAlloc(sizeCompact, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT); // This is the static GAS allocation!

    OPTIX_CHECK( m_api.optixAccelCompact(m_optixContext, m_cudaStream, data.traversable, d_gasCompact, sizeCompact, &data.traversable) );

    synchronizeStream(); // Must finish accessing data.d_gas source before it can be freed and overridden.

    memFree(data.d_gas);

    data.d_gas = d_gasCompact;

    //std::cout << "Compaction saved " << accelBufferSizes.outputSizeInBytes - sizeCompact << '\n'; // DEBUG
    accelBufferSizes.outputSizeInBytes = sizeCompact; // DEBUG for the std::cout below.
  }

  // Return the relocation info for this GAS traversable handle from this device's OptiX context.
  // It's used to assert that the GAS is compatible across devices which means NVLINK peer-to-peer sharing is allowed.
  // (This is more meant as example code, because in NVLINK islands the GPU configuration must be homogeneous and addresses are unique with UVA.)
  OPTIX_CHECK( m_api.optixAccelGetRelocationInfo(m_optixContext, data.traversable, &data.info) );

  //std::cout << "createGeometry() device index = " << m_index << ": attributes = " << attributesSizeInBytes << ", indices = " << indicesSizeInBytes << ", GAS = " << accelBufferSizes.outputSizeInBytes << "\n"; // DEBUG

  return data;
}


void Device::destroyGeometry(GeometryData& data)
{
  memFree(data.d_gas);
  memFree(data.d_indices);
  memFree(data.d_attributes);
}

void Device::createInstance(const GeometryData& geometryData, const InstanceData& instanceData, const float matrix[12])
{
  activateContext();
  synchronizeStream();

  // If the GeometryData is owned by a different device, that means it has been created in a different OptiX context.
  // Then check if the data is compatible with the OptiX context on this device. 
  // If yes, it can be shared via peer-to-peer as well because the device pointers are all unique with UVA. It's not actually relocated.
  // If not, no instance with this geometry data is created and it'll be missing from the rendering.
  // Same when there is no valid material or shader index assigned.
  if (m_index != geometryData.owner) // No need to check compatibility on the same device.
  {
    int compatible = 0;

#if (OPTIX_VERSION >= 70600)
    OPTIX_CHECK( m_api.optixCheckRelocationCompatibility(m_optixContext, &geometryData.info, &compatible) );
#else
    OPTIX_CHECK( m_api.optixAccelCheckRelocationCompatibility(m_optixContext, &geometryData.info, &compatible) );
#endif

    if (compatible == 0)
    {
      std::cerr << "ERROR: createInstance() device index " << m_index << " is not AS-compatible with the GeometryData owner " << geometryData.owner << ". Instance ignored!\n";
      MY_ASSERT(!"createInstance() AS incompatible");
      return; // This means this geometry will not actually be present in the OptiX render graph of this device!
    }
  }

  // First check if there is a valid material assigned to this instance.
  const int idMaterial = instanceData.idMaterial;

  if (idMaterial < 0 || static_cast<int>(m_materialDefinitions.size()) < idMaterial)
  {
    std::cerr << "ERROR: createInstance() idMaterial " << idMaterial << " is invalid. Instance ignored!\n";
    MY_ASSERT(!"createInstance() idMaterial invalid.");
    return;
  }

  // Then check if we actually have a valid shader compiled for this material.
  const int indexShader = m_materialDefinitions[idMaterial].indexShader;

  if (indexShader < 0 || static_cast<int>(m_deviceShaderConfigurations.size()) < indexShader)
  {
    std::cerr << "ERROR: createInstance() indexShader " << indexShader << " is invalid. Instance ignored!\n";
    MY_ASSERT(!"createInstance() indexShader invalid.");
    return;
  }

  OptixInstance instance = {};
      
  const unsigned int id = static_cast<unsigned int>(m_instances.size());
  memcpy(instance.transform, matrix, sizeof(float) * 12);
  instance.instanceId        = id; // User defined instance index, queried with optixGetInstanceId().
  instance.visibilityMask    = 255;
  
  // PERF Determine which hit record to use.
  // Triangles:       Hit records 0 to 3. (no emission, emission) x (no cutout, cutout)
  // Cubic B-splines: Hit record 4.

  unsigned int hitRecord = 0;
  if (geometryData.primitiveType == PT_CURVES)
  {
    hitRecord = 4; // Cubic B-spline curves.
  }
  else
  {
    hitRecord |= ((m_deviceShaderConfigurations[indexShader].flags & USE_EMISSION      ) == 0) ? 0 : 1; // no emission, emission
    hitRecord |= ((m_deviceShaderConfigurations[indexShader].flags & USE_CUTOUT_OPACITY) == 0) ? 0 : 2; // no cutout  , cutout
  }

  instance.sbtOffset         = NUM_RAY_TYPES * hitRecord;
  instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
  instance.traversableHandle = geometryData.traversable;
    
  m_instances.push_back(instance); // OptixInstance data

  m_instanceData.push_back(instanceData); // Per instance data, indexed with instanceId: idGeometry, idMaterial, idLight, idObject.
}


void Device::createTLAS()
{
  activateContext();
  synchronizeStream();

  // Construct the TLAS by attaching all flattened instances.
  const size_t instancesSizeInBytes = sizeof(OptixInstance) * m_instances.size();

  CUdeviceptr d_instances = memAlloc(instancesSizeInBytes, OPTIX_INSTANCE_BYTE_ALIGNMENT, cuda::USAGE_TEMP);
  CU_CHECK( cuMemcpyHtoDAsync(d_instances, m_instances.data(), instancesSizeInBytes, m_cudaStream) );

  OptixBuildInput instanceInput = {};

  instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  instanceInput.instanceArray.instances    = d_instances;
  instanceInput.instanceArray.numInstances = static_cast<unsigned int>(m_instances.size());

  OptixAccelBuildOptions accelBuildOptions = {};

  accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
  if (m_count == 1)
  {
    accelBuildOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  }
  accelBuildOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;
  
  OptixAccelBufferSizes accelBufferSizes;

  OPTIX_CHECK( m_api.optixAccelComputeMemoryUsage(m_optixContext, &accelBuildOptions, &instanceInput, 1, &accelBufferSizes ) );

  m_d_ias = memAlloc(accelBufferSizes.outputSizeInBytes, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
  
  CUdeviceptr d_tmp = memAlloc(accelBufferSizes.tempSizeInBytes, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT, cuda::USAGE_TEMP);

  OPTIX_CHECK( m_api.optixAccelBuild(m_optixContext, m_cudaStream,
                                     &accelBuildOptions, &instanceInput, 1,
                                     d_tmp,   accelBufferSizes.tempSizeInBytes,
                                     m_d_ias, accelBufferSizes.outputSizeInBytes,
                                     &m_systemData.topObject, nullptr, 0));

  CU_CHECK( cuStreamSynchronize(m_cudaStream) );

  memFree(d_tmp);
  memFree(d_instances);
}


void Device::createGeometryInstanceData(const std::vector<GeometryData>& geometryData, const unsigned int stride, const unsigned int index)
{
  activateContext();
  synchronizeStream();

  const unsigned int numInstances = static_cast<unsigned int>(m_instances.size());

  m_geometryInstanceData.resize(numInstances);

  for (unsigned int i = 0; i < numInstances; ++i)
  {
    const InstanceData& inst = m_instanceData[i];
    const GeometryData& geom = geometryData[inst.idGeometry * stride + index]; // This addressing supports both peer-to-peer shared and non-shared GAS.

    GeometryInstanceData& gid = m_geometryInstanceData[i];

    gid.ids        = make_int4(inst.idMaterial, inst.idLight, inst.idObject, 0);
    gid.attributes = geom.d_attributes;
    gid.indices    = geom.d_indices;
  }

  m_d_geometryInstanceData = reinterpret_cast<GeometryInstanceData*>(memAlloc(sizeof(GeometryInstanceData) * numInstances, 16) ); // int4 requires 16 byte alignment.
  CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(m_d_geometryInstanceData), m_geometryInstanceData.data(), sizeof(GeometryInstanceData) * numInstances, m_cudaStream) );

  m_systemData.geometryInstanceData = m_d_geometryInstanceData;
}


// Given an OpenGL UUID find the matching CUDA device.
bool Device::matchUUID(const char* uuid)
{
  for (size_t i = 0; i < 16; ++i)
  {
    if (m_deviceUUID.bytes[i] != uuid[i])
    {
      return false;
    }
  }
  return true;
}

// Given an OpenGL LUID find the matching CUDA device.
bool Device::matchLUID(const char* luid, const unsigned int nodeMask)
{
  if ((m_nodeMask & nodeMask) == 0)
  {
    return false;
  }
  for (size_t i = 0; i < 8; ++i)
  {
    if (m_deviceLUID[i] != luid[i])
    {
      return false;
    }
  }
  return true;
}


void Device::activateContext() const
{
  CU_CHECK( cuCtxSetCurrent(m_cudaContext) ); 
}

void Device::synchronizeStream() const
{
  CU_CHECK( cuStreamSynchronize(m_cudaStream) );
}

void Device::render(const unsigned int iterationIndex, void** buffer, const int mode)
{
  activateContext();

  m_systemData.iterationIndex = iterationIndex;

  if (m_isDirtyOutputBuffer)
  {
    MY_ASSERT(buffer != nullptr);
    if (*buffer == nullptr) // The buffer is nullptr for the device which should allocate the full resolution buffers. This device is called first!
    {
      // Only allocate the host buffer once, not per each device.
      m_bufferHost.resize(m_systemData.resolution.x * m_systemData.resolution.y);

      // Note that this requires that all other devices have finished accessing this buffer, but that is automatically the case
      // after calling Device::setState() which is the only place which can change the resolution.
      memFree(m_systemData.outputBuffer); // This is asynchronous and the pointer can be 0.
      m_systemData.outputBuffer = memAlloc(sizeof(float4) * m_systemData.resolution.x * m_systemData.resolution.y, sizeof(float4));

      *buffer = reinterpret_cast<void*>(m_systemData.outputBuffer); // Set the pointer, so that other devices don't allocate it. It's not shared!

      if (1 < m_count)
      {
        // This is a temporary buffer on the primary board which is used by the compositor. The texelBuffer needs to stay intact for the accumulation.
        memFree(m_systemData.tileBuffer);
        m_systemData.tileBuffer = memAlloc(sizeof(float4) * m_launchWidth * m_systemData.resolution.y, sizeof(float4));

        m_d_compositorData = memAlloc(sizeof(CompositorData), 16);
      }

      m_ownsSharedBuffer = true; // Indicate which device owns the m_systemData.outputBuffer and m_bufferHost so that display routines can assert.

      if (m_cudaGraphicsResource != nullptr) // Need to unregister texture or PBO before resizing it.
      {
        CU_CHECK( cuGraphicsUnregisterResource(m_cudaGraphicsResource) );
      }

      switch (m_interop)
      {
        case INTEROP_MODE_OFF:
          break;

        case INTEROP_MODE_TEX:
          // Let the device which is called first resize the OpenGL texture.
          glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei) m_systemData.resolution.x, (GLsizei) m_systemData.resolution.y, 0, GL_RGBA, GL_FLOAT, (GLvoid*) m_bufferHost.data()); // RGBA32F
          glFinish(); // Synchronize with following CUDA operations.

          CU_CHECK( cuGraphicsGLRegisterImage(&m_cudaGraphicsResource, m_tex, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD) );
          break;

        case INTEROP_MODE_PBO:
          glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
          glBufferData(GL_PIXEL_UNPACK_BUFFER, m_systemData.resolution.x * m_systemData.resolution.y * sizeof(float4), nullptr, GL_DYNAMIC_DRAW);
          glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

          CU_CHECK( cuGraphicsGLRegisterBuffer(&m_cudaGraphicsResource, m_pbo, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD) ); 
          break;
      }
    }

    if (1 < m_count)
    {
      // Allocate a GPU local buffer in the per-device launch size. This is where the accumulation happens.
      memFree(m_systemData.texelBuffer);
      m_systemData.texelBuffer = memAlloc(sizeof(float4) * m_launchWidth * m_systemData.resolution.y, sizeof(float4));
    }

    m_isDirtyOutputBuffer = false; // Buffer is allocated with new size.
    m_isDirtySystemData   = true;  // Now the sysData on the device needs to be updated, and that needs a sync!
  }

  if (m_isDirtySystemData) // Update the whole SystemData block because more than the iterationIndex changed. This normally means a GUI interaction. Just sync.
  {
    synchronizeStream();

    CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(m_d_systemData), &m_systemData, sizeof(SystemData), m_cudaStream) );
    m_isDirtySystemData = false;
  }
  else // Just copy the new iterationIndex.
  {
    if (mode == 0) // Fully asynchronous launches ruin the interactivity. Synchronize in interactive mode.
    {
      synchronizeStream();
    }
    // PERF For really asynchronous copies of the iteration indices, multiple source pointers are required. Good that I know the number of iterations upfront!
    // Using the m_subFrames array as source pointers. Just contains the identity of the index. Updating the device side sysData.iterationIndex from there.
    CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(&m_d_systemData->iterationIndex), &m_subFrames[m_systemData.iterationIndex], sizeof(unsigned int), m_cudaStream) );
  }

  // Note the launch width per device to render in tiles.
  OPTIX_CHECK( m_api.optixLaunch(m_pipeline, m_cudaStream, reinterpret_cast<CUdeviceptr>(m_d_systemData), sizeof(SystemData), &m_sbt, m_launchWidth, m_systemData.resolution.y, /* depth */ 1) );
}


void Device::updateDisplayTexture()
{
  activateContext();

  // Only allow this on the device which owns the shared peer-to-peer buffer which also resized the host buffer to copy this to the host.
  MY_ASSERT(!m_isDirtyOutputBuffer && m_ownsSharedBuffer && m_tex != 0);

  switch (m_interop)
  {
    case INTEROP_MODE_OFF:
      // Copy the GPU local render buffer into host and update the HDR texture image from there.
      CU_CHECK( cuMemcpyDtoHAsync(m_bufferHost.data(), m_systemData.outputBuffer, sizeof(float4) * m_systemData.resolution.x * m_systemData.resolution.y, m_cudaStream) );
      synchronizeStream(); // Wait for the buffer to arrive on the host.

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, m_tex);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei) m_systemData.resolution.x, (GLsizei) m_systemData.resolution.y, 0, GL_RGBA, GL_FLOAT, m_bufferHost.data()); // RGBA32F from host buffer data.
      break;
      
    case INTEROP_MODE_TEX:
      {
        // Map the Texture object directly and copy the output buffer. 
        CU_CHECK( cuGraphicsMapResources(1, &m_cudaGraphicsResource, m_cudaStream )); // This is an implicit cuSynchronizeStream().

        CUarray dstArray = nullptr;

        CU_CHECK( cuGraphicsSubResourceGetMappedArray(&dstArray, m_cudaGraphicsResource, 0, 0) ); // arrayIndex = 0, mipLevel = 0

        CUDA_MEMCPY3D params = {};

        params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        params.srcDevice     = m_systemData.outputBuffer;
        params.srcPitch      = m_systemData.resolution.x * sizeof(float4);
        params.srcHeight     = m_systemData.resolution.y;

        params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        params.dstArray      = dstArray;
        params.WidthInBytes  = m_systemData.resolution.x * sizeof(float4);
        params.Height        = m_systemData.resolution.y;
        params.Depth         = 1;

        CU_CHECK( cuMemcpy3D(&params) ); // Copy from linear to array layout.

        CU_CHECK( cuGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream) ); // This is an implicit cuSynchronizeStream().
      }
      break;

    case INTEROP_MODE_PBO: // This contains two device-to-device copies and is just for demonstration. Use INTEROP_MODE_TEX when possible.
      {
        size_t size = 0;
        CUdeviceptr d_ptr;
  
        CU_CHECK( cuGraphicsMapResources(1, &m_cudaGraphicsResource, m_cudaStream) ); // This is an implicit cuSynchronizeStream().
        CU_CHECK( cuGraphicsResourceGetMappedPointer(&d_ptr, &size, m_cudaGraphicsResource) ); // The pointer can change on every map!
        MY_ASSERT(m_systemData.resolution.x * m_systemData.resolution.y * sizeof(float4) <= size);
        CU_CHECK( cuMemcpyDtoDAsync(d_ptr, m_systemData.outputBuffer, m_systemData.resolution.x * m_systemData.resolution.y * sizeof(float4), m_cudaStream) ); // PERF PBO interop is kind of moot with a direct texture access.
        CU_CHECK( cuGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream) ); // This is an implicit cuSynchronizeStream().

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_tex);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei) m_systemData.resolution.x, (GLsizei) m_systemData.resolution.y, 0, GL_RGBA, GL_FLOAT, (GLvoid*) 0); // RGBA32F from byte offset 0 in the pixel unpack buffer.
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      }
      break;
  }
}


const void* Device::getOutputBufferHost()
{
  activateContext();

  MY_ASSERT(!m_isDirtyOutputBuffer && m_ownsSharedBuffer); // Only allow this on the device which owns the shared peer-to-peer buffer and resized the host buffer to copy this to the host.
  
  // Note that the caller takes care to sync the other devices before calling into here or this image might not be complete!
  CU_CHECK( cuMemcpyDtoHAsync(m_bufferHost.data(), m_systemData.outputBuffer, sizeof(float4) * m_systemData.resolution.x * m_systemData.resolution.y, m_cudaStream) );
    
  synchronizeStream(); // Wait for the buffer to arrive on the host.

  return m_bufferHost.data();
}

// PERF This is NOT called when there is only one active device!
// That is using a different ray generation program instead which accumulates directly into the output buffer.
void Device::compositor(Device* other)
{
  MY_ASSERT(!m_isDirtyOutputBuffer && m_ownsSharedBuffer);

  // The compositor sources the tileBuffer, which is only allocated on the primary device. 
  // The texelBuffer is a GPU local buffer on all devices and contains the accumulation.
  if (this == other)
  {
    activateContext();

    CU_CHECK( cuMemcpyDtoDAsync(m_systemData.tileBuffer, m_systemData.texelBuffer,
                                sizeof(float4) * m_launchWidth * m_systemData.resolution.y, m_cudaStream) );
  }
  else
  {
    // Make sure the other device has finished rendering! Otherwise there can be checkerboard corruption visible.
    other->activateContext();
    other->synchronizeStream();
  
    activateContext();

    CU_CHECK( cuMemcpyPeerAsync(m_systemData.tileBuffer, m_cudaContext, other->m_systemData.texelBuffer, other->m_cudaContext,
                                sizeof(float4) * m_launchWidth * m_systemData.resolution.y, m_cudaStream) );
  }

  CompositorData compositorData; // FIXME This would need to be persistent per Device to allow async copies!

  compositorData.outputBuffer = m_systemData.outputBuffer;
  compositorData.tileBuffer   = m_systemData.tileBuffer;
  compositorData.resolution   = m_systemData.resolution;
  compositorData.tileSize     = m_systemData.tileSize;
  compositorData.tileShift    = m_systemData.tileShift;
  compositorData.launchWidth  = m_launchWidth;
  compositorData.deviceCount  = m_systemData.deviceCount;
  compositorData.deviceIndex  = other->m_systemData.deviceIndex; // This is the only value which changes per device. 

  // Need a synchronous copy here to not overwrite or delete the compositorData above.
  CU_CHECK( cuMemcpyHtoD(m_d_compositorData, &compositorData, sizeof(CompositorData)) );
 
  void* args[1] = { &m_d_compositorData };

  const int blockDimX = std::min(compositorData.tileSize.x, 16);
  const int blockDimY = std::min(compositorData.tileSize.y, 16);

  const int gridDimX  = (m_launchWidth               + blockDimX - 1) / blockDimX;
  const int gridDimY  = (compositorData.resolution.y + blockDimY - 1) / blockDimY;

  MY_ASSERT(gridDimX <= m_deviceAttribute.maxGridDimX && 
            gridDimY <= m_deviceAttribute.maxGridDimY);

  // Reduction kernel with launch dimension of height blocks with 32 threads.
  CU_CHECK( cuLaunchKernel(m_functionCompositor,    // CUfunction f,
                                       gridDimX,    // unsigned int gridDimX,
                                       gridDimY,    // unsigned int gridDimY,
                                              1,    // unsigned int gridDimZ,
                                      blockDimX,    // unsigned int blockDimX,
                                      blockDimY,    // unsigned int blockDimY,
                                              1,    // unsigned int blockDimZ,
                                              0,    // unsigned int sharedMemBytes,
                                   m_cudaStream,    // CUstream hStream,
                                           args,    // void **kernelParams,
                                        nullptr) ); // void **extra

  synchronizeStream();
}


// Arena version of cuMemAlloc(), but asynchronous!
CUdeviceptr Device::memAlloc(const size_t size, const size_t alignment, const cuda::Usage usage)
{
  return m_allocator->alloc(size, alignment, usage);
}

// Arena version of cuMemFree(), but asynchronous!
void Device::memFree(const CUdeviceptr ptr)
{
  m_allocator->free(ptr);
}

// This is getting the current VRAM situation on the device.
// Means this includes everything running on the GPU and all allocations done for textures and the ArenaAllocator.
// Currently not used for picking the home device for the next shared allocation, because with the ArenaAllocator that isn't fine grained.
// Instead getMemoryAllocated() is used to return the sum of all allocated blocks inside arenas and the texture sizes in bytes.
size_t Device::getMemoryFree() const
{
  activateContext();

  size_t sizeFree  = 0;
  size_t sizeTotal = 0;

  CU_CHECK( cuMemGetInfo(&sizeFree, &sizeTotal) );

  return sizeFree;
}

// getMemoryAllocated() returns the sum of all allocated blocks inside arenas and the texture sizes in bytes (without GPU alignment and padding).
// Using this in Raytracer::getDeviceHome() assumes the free VRAM amount is about equal on the devices in an island.
size_t Device::getMemoryAllocated() const
{
  return m_allocator->getSizeMemoryAllocated() + m_sizeMemoryTextureArrays;
}

Texture* Device::initTexture(const std::string& name, const Picture* picture, const unsigned int flags)
{
  activateContext();
  synchronizeStream(); // PERF Required here?

  Texture* texture;

  // FIXME Only using the filename as key and not the load flags. This will not support the same image with different flags!
  std::map<std::string, Texture*>::const_iterator it = m_mapTextures.find(name); 
  if (it == m_mapTextures.end())
  {
    texture = new Texture(this); // This device is the owner of the CUarray or CUmipmappedArray data.
    texture->create(picture, flags); 

    m_sizeMemoryTextureArrays += texture->getSizeBytes(); // Texture memory tracking.

    m_mapTextures[name] = texture;

    std::cout << "initTexture() device index = " << m_index << ": name = " << name << '\n'; // DEBUG
  }
  else
  {
    texture = it->second; // Return the existing texture under this name.
    
    std::cout << "initTexture() Texture " << name << " reused\n"; // DEBUG
  }

  return texture; // Not used when not sharing.
}


void Device::shareTexture(const std::string& name, const Texture* shared)
{
  activateContext();
  synchronizeStream(); // PERF Required here?

  std::map<std::string, Texture*>::const_iterator it = m_mapTextures.find(name);

  if (it == m_mapTextures.end())
  {
    Texture* texture = new Texture(shared->getOwner());
    
    texture->create(shared); // No texture memory tracking in this case. Arrays are reused.

    m_mapTextures[name] = texture;
  }
}


unsigned int Device::appendProgramGroupMDL(const int indexModule, const std::string& nameFunction)
{
  OptixProgramGroupDesc pgd = {};

  pgd.kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd.callables.moduleDC            = m_modulesMDL[indexModule];
  pgd.callables.entryFunctionNameDC = nameFunction.c_str();
 
  OptixProgramGroup pg = {};

  OPTIX_CHECK( m_api.optixProgramGroupCreate(m_optixContext, &pgd, 1u, &m_pgo, nullptr, nullptr, &pg) );

  // Add the call offset skipping the lens shader and light sample callables on the host.
  const unsigned int idx = static_cast<unsigned int>(m_programGroupsMDL.size()) + CALL_OFFSET;

  m_programGroupsMDL.push_back(pg);

  return idx;
}


// Compile_result and MaterialMDL is per reference, ShaderConfiguration is per shader (code).
void Device::compileMaterial(mi::neuraylib::ITransaction* transaction,
                             MaterialMDL* material,
                             const Compile_result& res,
                             const ShaderConfiguration& config)
{
  activateContext();
  synchronizeStream(); // PERF Required here?

  // This function is called per reference because it needs to allocate and store the parameter argument block per reference.
  // Though the shader code, resp. the callable program indices need to be reused.
  const int indexShader = material->getShaderIndex();

  const std::string suffix = std::to_string(indexShader);

  DeviceShaderConfiguration dsc = {};

  // Set all callable indices to the invalid value -1.
  // The MDL code generator will generate all functions (sample, evaluate, pdf).
  // This is only containing the direct callables which are required inside the pipeline of this unidirectional path tracer.
  
  dsc.idxCallInit = -1;

  dsc.idxCallThinWalled = -1;

  dsc.idxCallSurfaceScatteringSample = -1;
  dsc.idxCallSurfaceScatteringEval   = -1;

  dsc.idxCallBackfaceScatteringSample = -1;
  dsc.idxCallBackfaceScatteringEval   = -1;

  dsc.idxCallSurfaceEmissionEval           = -1;
  dsc.idxCallSurfaceEmissionIntensity      = -1;
  dsc.idxCallSurfaceEmissionIntensityMode  = -1;

  dsc.idxCallBackfaceEmissionEval           = -1;
  dsc.idxCallBackfaceEmissionIntensity      = -1;
  dsc.idxCallBackfaceEmissionIntensityMode  = -1;

  dsc.idxCallIor = -1;
  
  // No direct callables for VDFs itself. The MDL SDK is not generating code for VDFs.

  dsc.idxCallVolumeAbsorptionCoefficient = -1;
  dsc.idxCallVolumeScatteringCoefficient = -1;
  dsc.idxCallVolumeDirectionalBias       = -1;

  dsc.idxCallGeometryCutoutOpacity = -1;

  dsc.idxCallHairSample = -1;
  dsc.idxCallHairEval   = -1;

  // Simplify the conditions by translating all constants unconditionally.
  if (config.thin_walled)
  {
    dsc.flags |= IS_THIN_WALLED;
  }
  dsc.surface_intensity       = make_float3(config.surface_intensity[0], config.surface_intensity[1], config.surface_intensity[2]);
  dsc.surface_intensity_mode  = config.surface_intensity_mode;
  dsc.backface_intensity      = make_float3(config.backface_intensity[0], config.backface_intensity[1], config.backface_intensity[2]);
  dsc.backface_intensity_mode = config.backface_intensity_mode;
  dsc.ior                     = make_float3(config.ior[0], config.ior[1], config.ior[2]);
  dsc.absorption_coefficient  = make_float3(config.absorption_coefficient[0], config.absorption_coefficient[1], config.absorption_coefficient[2]);
  dsc.scattering_coefficient  = make_float3(config.scattering_coefficient[0], config.scattering_coefficient[1], config.scattering_coefficient[2]);
  dsc.cutout_opacity          = config.cutout_opacity;

  MY_ASSERT(indexShader <= m_modulesMDL.size());

  // If the shader index hasn't been seen before, we need to create a new OptixModule and a device side shader configuration.
  // Otherwise the indexShader is already indexing the correct shader configuration and 
  // only the per reference parameter argument block and texture_handler need to be setup per reference.
  if (indexShader == m_modulesMDL.size())
  {
    OptixModule moduleMDL = {};

#if (OPTIX_VERSION >= 70700)
    OPTIX_CHECK( m_api.optixModuleCreate(m_optixContext, &m_mco, &m_pco, res.target_code->get_code(), res.target_code->get_code_size(), nullptr, nullptr, &moduleMDL) );
#else
    OPTIX_CHECK( m_api.optixModuleCreateFromPTX(m_optixContext, &m_mco, &m_pco, res.target_code->get_code(), res.target_code->get_code_size(), nullptr, nullptr, &moduleMDL) );
#endif
    
    m_modulesMDL.push_back(moduleMDL);
  
    dsc.idxCallInit = appendProgramGroupMDL(indexShader, std::string("__direct_callable__init") + suffix); // The material init function.

    if (!config.is_thin_walled_constant)
    {
      dsc.idxCallThinWalled = appendProgramGroupMDL(indexShader, std::string("__direct_callable__thin_walled") + suffix);
    }

    if (config.is_surface_bsdf_valid)
    {
      const std::string name = std::string("__direct_callable__surface_scattering") + suffix;

      dsc.idxCallSurfaceScatteringSample = appendProgramGroupMDL(indexShader, name + std::string("_sample"));
      dsc.idxCallSurfaceScatteringEval   = appendProgramGroupMDL(indexShader, name + std::string("_evaluate"));
    }

    if (config.is_backface_bsdf_valid)
    {
      const std::string name = std::string("__direct_callable__backface_scattering") + suffix;

      dsc.idxCallBackfaceScatteringSample = appendProgramGroupMDL(indexShader, name + std::string("_sample"));
      dsc.idxCallBackfaceScatteringEval   = appendProgramGroupMDL(indexShader, name + std::string("_evaluate"));
    }

    if (config.is_surface_edf_valid)
    {
      const std::string name = std::string("__direct_callable__surface_emission_emission") + suffix;

      dsc.idxCallSurfaceEmissionEval = appendProgramGroupMDL(indexShader, name + std::string("_evaluate"));

      if (!config.is_surface_intensity_constant)
      {
        dsc.idxCallSurfaceEmissionIntensity = appendProgramGroupMDL(indexShader, std::string("__direct_callable__surface_emission_intensity") + suffix);
      }

      if (!config.is_surface_intensity_mode_constant)
      {
        dsc.idxCallSurfaceEmissionIntensityMode = appendProgramGroupMDL(indexShader, std::string("__direct_callable__surface_emission_mode") + suffix);
      }
    }

    if (config.is_backface_edf_valid)
    {
      if (config.use_backface_edf)
      {
        const std::string name = std::string("__direct_callable__backface_emission_emission") + suffix;

        dsc.idxCallBackfaceEmissionEval = appendProgramGroupMDL(indexShader, name + std::string("_evaluate"));
      }
      else // Surface and backface expressions were identical. Reuse the code of the surface expression.
      {
        dsc.idxCallBackfaceEmissionEval = dsc.idxCallSurfaceEmissionEval;
      }

      if (config.use_backface_intensity)
      {
        if (!config.is_backface_intensity_constant)
        {
          dsc.idxCallBackfaceEmissionIntensity = appendProgramGroupMDL(indexShader, std::string("__direct_callable__backface_emission_intensity") + suffix);
        }
      }
      else // Surface and backface expressions were identical. Reuse the code of the surface expression.
      {
        dsc.idxCallBackfaceEmissionIntensity = dsc.idxCallSurfaceEmissionIntensity;
      }

      if (config.use_backface_intensity_mode)
      {
        if (!config.is_backface_intensity_mode_constant)
        {
          dsc.idxCallBackfaceEmissionIntensityMode = appendProgramGroupMDL(indexShader, std::string("__direct_callable__backface_emission_mode") + suffix);
        }
      }
      else // Surface and backface expressions were identical. Reuse the code of the surface expression.
      {
        dsc.idxCallBackfaceEmissionIntensityMode = dsc.idxCallSurfaceEmissionIntensityMode;
      }
    }

    if (config.isEmissive())
    {
      dsc.flags |= USE_EMISSION;
    }

    if (!config.is_ior_constant)
    {
      dsc.idxCallIor = appendProgramGroupMDL(indexShader, std::string("__direct_callable__ior") + suffix);
    }

    if (!config.is_absorption_coefficient_constant)
    {
      dsc.idxCallVolumeAbsorptionCoefficient = appendProgramGroupMDL(indexShader, std::string("__direct_callable__volume_absorption_coefficient") + suffix);
    }

    if (config.is_vdf_valid)
    {
      // The MDL SDK doesn't generate code for the volume.scattering expression.
      // Means volume scattering must be implemented by the renderer and only the parameter expresssions can be generated.

      // The volume scattering coefficient and direction bias are only used when there is a valid VDF. 
      if (!config.is_scattering_coefficient_constant)
      {
        dsc.idxCallVolumeScatteringCoefficient = appendProgramGroupMDL(indexShader, std::string("__direct_callable__volume_scattering_coefficient") + suffix);
      }

      if (!config.is_directional_bias_constant)
      {
        dsc.idxCallVolumeDirectionalBias = appendProgramGroupMDL(indexShader, std::string("__direct_callable__volume_directional_bias") + suffix);
      }

      // volume.scattering.emission_intensity not implemented.
    }

    if (config.use_cutout_opacity)
    {
      dsc.flags |= USE_CUTOUT_OPACITY;
      
      if (!config.is_cutout_opacity_constant)
      {
        dsc.idxCallGeometryCutoutOpacity = appendProgramGroupMDL(indexShader, std::string("__direct_callable__geometry_cutout_opacity") + suffix);
      }
    }

    if (config.is_hair_bsdf_valid)
    {
        const std::string name = std::string("__direct_callable__hair") + suffix;

        dsc.idxCallHairSample = appendProgramGroupMDL(indexShader, name + std::string("_sample"));
        dsc.idxCallHairEval   = appendProgramGroupMDL(indexShader, name + std::string("_evaluate"));
    }

     m_deviceShaderConfigurations.push_back(dsc);
  
    MY_ASSERT(m_modulesMDL.size() == m_deviceShaderConfigurations.size());
  }
}


const TextureMDLHost* Device::prepareTextureMDL(mi::neuraylib::ITransaction* transaction,
                                                mi::base::Handle<mi::neuraylib::IImage_api> image_api,
                                                char const* texture_db_name,
                                                mi::neuraylib::ITarget_code::Texture_shape texture_shape)
{
  activateContext();
  synchronizeStream(); // PERF Required here?
  
  // Get access to the texture data by the texture database name from the target code.
  mi::base::Handle<const mi::neuraylib::ITexture> texture(transaction->access<mi::neuraylib::ITexture>(texture_db_name));

  // First check the texture cache.
  std::string entry_name = std::string(texture_db_name) + "_" + std::to_string(unsigned(texture_shape));

  const auto& it = m_mapTextureNameToIndex.find(entry_name);
  if (it != m_mapTextureNameToIndex.end())
  {
    return &m_textureMDLHosts[it->second]; // The texture already exists inside the texture cache on this device.
  }

  // This is the structure which will hold the newly created texture data.
  TextureMDLHost host;
  memset(&host, 0, sizeof(TextureMDLHost));

  host.m_owner = this; // Track the device which owns the CUarray data.

  // std::cout << "DEBUG: prepareTextureMDL() loading " << entry_name << '\n';

  // Access image and canvas via the texture object
  mi::base::Handle<const mi::neuraylib::IImage> image(transaction->access<mi::neuraylib::IImage>(texture->get_image()));
  
  mi::base::Handle<const mi::neuraylib::ICanvas> canvas(image->get_canvas(0, 0, 0));
  
  mi::Uint32 tex_width  = canvas->get_resolution_x();
  mi::Uint32 tex_height = canvas->get_resolution_y();
  mi::Uint32 tex_layers = canvas->get_layers_size();
  
  if (image->is_uvtile() || image->is_animated())
  {
    std::cerr << "ERROR: prepareTextureMDL() uvtile and/or animated textures not supported!\n";
    return nullptr;
  }

  char const* image_type = image->get_type(0, 0);

  // Determine the image type.

  // MDL pixel types.
  //"Sint8"      // Signed 8-bit integer
  //"Sint32"     // Signed 32-bit integer
  //"Float32"    // 32-bit IEEE-754 single-precision floating-point number
  //"Float32<2>" // 2 x Float32
  //"Float32<3>" // 3 x Float32
  //"Float32<4>" // 4 x Float32
  //"Rgb"        // 3 x Uint8 representing RGB color
  //"Rgba"       // 4 x Uint8 representing RGBA color
  //"Rgbe"       // 4 x Uint8 representing RGBE color
  //"Rgbea"      // 5 x Uint8 representing RGBEA color
  //"Rgb_16"     // 3 x Uint16 representing RGB color
  //"Rgba_16"    // 4 x Uint16 representing RGBA color
  //"Rgb_fp"     // 3 x Float32 representing RGB color
  //"Color"      // 4 x Float32 representing RGBA color

  size_t sizeBytesPerElement = 1;

  const mi::Float32 effectiveGamma = texture->get_effective_gamma(0, 0);

  // Handle RGB8 and RGBA8 images natively.
  if (strcmp(image_type, "Rgb") == 0)
  {
    canvas = image_api->convert(canvas.get(), "Rgba"); // Append an alpha channel with 0xFF.

    host.m_descArray3D.Format      = CU_AD_FORMAT_UNSIGNED_INT8;
    host.m_descArray3D.NumChannels = 4;

    sizeBytesPerElement = sizeof(mi::Uint8);
  }
  else if (strcmp(image_type, "Rgba") == 0)
  {
    host.m_descArray3D.Format      = CU_AD_FORMAT_UNSIGNED_INT8;
    host.m_descArray3D.NumChannels = 4;

    sizeBytesPerElement = sizeof(mi::Uint8);
  }
  else // FIXME PERF All other formats are currently converted to linear float4 colors.
  {
    if (effectiveGamma != 1.0f) // Convert image to linear color space if necessary.
    {
      // Copy/convert to float4 canvas and adjust gamma from "effective gamma" to 1.0.
      mi::base::Handle<mi::neuraylib::ICanvas> gamma_canvas(image_api->convert(canvas.get(), "Color"));

      gamma_canvas->set_gamma(texture->get_effective_gamma(0, 0));
      image_api->adjust_gamma(gamma_canvas.get(), 1.0f);
    
      canvas = gamma_canvas;
    }
    else if (strcmp(image_type, "Color")      != 0 && 
             strcmp(image_type, "Float32<4>") != 0) 
    {
      // All other formats which aren't float4 already get converted to linear float4 color. Gamma is 1.0 here.
      canvas = image_api->convert(canvas.get(), "Color");
    }

    host.m_descArray3D.Format      = CU_AD_FORMAT_FLOAT;
    host.m_descArray3D.NumChannels = 4;

    sizeBytesPerElement = sizeof(mi::Float32);
  }

  host.m_resourceDescription = {};

  // Copy image data to GPU array depending on texture shape
  if (texture_shape == mi::neuraylib::ITarget_code::Texture_shape_cube ||
      texture_shape == mi::neuraylib::ITarget_code::Texture_shape_3d ||
      texture_shape == mi::neuraylib::ITarget_code::Texture_shape_bsdf_data) // DEBUG MBSDF data should not reach this code path.
  {
    // Cubemap and 3D texture objects require 3D CUDA arrays.
    if (texture_shape == mi::neuraylib::ITarget_code::Texture_shape_cube && tex_layers != 6)
    {
      std::cerr << "ERROR: prepareTextureMDL() Invalid number of layers (" << tex_layers << "), cubemaps must have 6 layers!\n";
      return nullptr;
    }

    // Allocate a 3D array on the GPU
    host.m_descArray3D.Width  = tex_width;
    host.m_descArray3D.Height = tex_height;
    host.m_descArray3D.Depth  = tex_layers; // Not a 2D texture if this is != 0.

    // Track the current texture allocation size on this device.
    host.m_sizeBytesArray = host.m_descArray3D.Width *
                            host.m_descArray3D.Height *
                            host.m_descArray3D.Depth *
                            host.m_descArray3D.NumChannels *
                            ((host.m_descArray3D.Format == CU_AD_FORMAT_UNSIGNED_INT8) ? 1 : 4);
    m_sizeMemoryTextureArrays += host.m_sizeBytesArray;

    CU_CHECK( cuArray3DCreate(&host.m_d_array, &host.m_descArray3D) );

    // Prepare the memcpy parameter structure

    // Copy the image data of all layers (the layers are not consecutive in memory)
    for (mi::Uint32 layer = 0; layer < tex_layers; ++layer)
    {
      mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile(layer));
      
      CUDA_MEMCPY3D params = {};

      params.srcMemoryType = CU_MEMORYTYPE_HOST;
      params.srcHost       = tile->get_data();
      params.srcPitch      = tex_width * sizeBytesPerElement * host.m_descArray3D.NumChannels;
      params.srcHeight     = tex_height;

      params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
      params.dstArray      = host.m_d_array;

      params.dstXInBytes   = 0;
      params.dstY          = 0;
      params.dstZ          = layer;

      params.WidthInBytes  = params.srcPitch;
      params.Height        = tex_height;
      params.Depth         = 1;

      CU_CHECK( cuMemcpy3D(&params) );
    }

    host.m_resourceDescription.resType = CU_RESOURCE_TYPE_ARRAY;
    host.m_resourceDescription.res.array.hArray = host.m_d_array;
 }
  else
  {
    // 2D texture objects use CUDA arrays
    host.m_descArray3D.Width  = tex_width;
    host.m_descArray3D.Height = tex_height;
    host.m_descArray3D.Depth  = 0; // A 2D array is allocated if only Depth extent is zero.
    
    // Track the current texture allocation size on this device.
    host.m_sizeBytesArray = host.m_descArray3D.Width *
                            host.m_descArray3D.Height *
                            host.m_descArray3D.NumChannels *
                            ((host.m_descArray3D.Format == CU_AD_FORMAT_UNSIGNED_INT8) ? 1 : 4);
    m_sizeMemoryTextureArrays += host.m_sizeBytesArray;

    CU_CHECK( cuArray3DCreate(&host.m_d_array, &host.m_descArray3D) );

    mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile());
   
    CUDA_MEMCPY3D params = {};

    params.srcMemoryType = CU_MEMORYTYPE_HOST;
    params.srcHost       = tile->get_data();
    params.srcPitch      = tex_width * sizeBytesPerElement * host.m_descArray3D.NumChannels;
    params.srcHeight     = tex_height;

    params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    params.dstArray      = host.m_d_array;

    params.WidthInBytes  = params.srcPitch;
    params.Height        = tex_height;
    params.Depth         = 1;

    CU_CHECK( cuMemcpy3D(&params) );

    host.m_resourceDescription.resType = CU_RESOURCE_TYPE_ARRAY;
    host.m_resourceDescription.res.array.hArray = host.m_d_array;
  }

  // For cube maps we need clamped address mode to avoid artifacts in the corners.
  CUaddress_mode addr_mode = (texture_shape == mi::neuraylib::ITarget_code::Texture_shape_cube) ? CU_TR_ADDRESS_MODE_CLAMP : CU_TR_ADDRESS_MODE_WRAP;

  // Create filtered texture object
  host.m_textureDescription = {};

  // If the flag CU_TRSF_NORMALIZED_COORDINATES is not set, the only supported address mode is CU_TR_ADDRESS_MODE_CLAMP.
  host.m_textureDescription.addressMode[0] = addr_mode;
  host.m_textureDescription.addressMode[1] = addr_mode;
  host.m_textureDescription.addressMode[2] = addr_mode;

  host.m_textureDescription.filterMode = CU_TR_FILTER_MODE_LINEAR; // Bilinear filtering by default.

  // Possible flags: CU_TRSF_READ_AS_INTEGER, CU_TRSF_NORMALIZED_COORDINATES, CU_TRSF_SRGB
  host.m_textureDescription.flags = CU_TRSF_NORMALIZED_COORDINATES;
  if (effectiveGamma != 1.0f && sizeBytesPerElement == 1)
  {
    host.m_textureDescription.flags |= CU_TRSF_SRGB;
  }

  host.m_textureDescription.maxAnisotropy = 1;

  // LOD 0 only by default.
  // This means when using mipmaps it's the developer's responsibility to set at least 
  // maxMipmapLevelClamp > 0.0f before calling Texture::create() to make sure mipmaps can be sampled!
  host.m_textureDescription.mipmapFilterMode    = CU_TR_FILTER_MODE_POINT;
  host.m_textureDescription.mipmapLevelBias     = 0.0f;
  host.m_textureDescription.minMipmapLevelClamp = 0.0f;
  host.m_textureDescription.maxMipmapLevelClamp = 0.0f; // This should be set to Picture::getNumberOfLevels() when using mipmaps.

  host.m_textureDescription.borderColor[0] = 0.0f;
  host.m_textureDescription.borderColor[1] = 0.0f;
  host.m_textureDescription.borderColor[2] = 0.0f;
  host.m_textureDescription.borderColor[3] = 0.0f;

  CUtexObject tex_obj = 0; // This type is interchangeable with cudaTextureObject_t.
  
  CU_CHECK( cuTexObjectCreate(&tex_obj, &host.m_resourceDescription, &host.m_textureDescription, nullptr) ); 

  // Create unfiltered texture object if necessary (cube textures have no texel functions)
  CUtexObject tex_obj_unfilt = 0;

  if (texture_shape != mi::neuraylib::ITarget_code::Texture_shape_cube)
  {
    // Use a black border for access outside of the texture
    host.m_textureDescription.addressMode[0] = CU_TR_ADDRESS_MODE_BORDER;
    host.m_textureDescription.addressMode[1] = CU_TR_ADDRESS_MODE_BORDER;
    host.m_textureDescription.addressMode[2] = CU_TR_ADDRESS_MODE_BORDER;
    
    host.m_textureDescription.filterMode = CU_TR_FILTER_MODE_POINT;

    CU_CHECK( cuTexObjectCreate(&tex_obj_unfilt, &host.m_resourceDescription, &host.m_textureDescription, nullptr) ); 
  }

  // Add the device-side information for the Texture_handler.
  host.m_texture = TextureMDL(tex_obj, tex_obj_unfilt, make_uint3(tex_width, tex_height, tex_layers));

  // Get the new texture entry index for the cache.
  const int indexCache = static_cast<int>(m_textureMDLHosts.size());
  // Track the cache index of this texture. This is needed to build the Texture_handler.
  host.m_index = indexCache;
  // Track the index inside the texture cache.
  m_mapTextureNameToIndex[entry_name] = indexCache;
  // Store the texture array and object handles inside the vector of all textures on this device.
  m_textureMDLHosts.push_back(host); 

  // Return the pointer this newly created TextureMDLHost.
  return &m_textureMDLHosts[indexCache];
}


void Device::shareTextureMDL(const TextureMDLHost* shared,
                             char const* texture_db_name,
                             mi::neuraylib::ITarget_code::Texture_shape texture_shape)
{
  activateContext();
  synchronizeStream(); // PERF Required here?

  // First check the texture cache.
  std::string entry_name = std::string(texture_db_name) + "_" + std::to_string(unsigned(texture_shape));

  const auto& it = m_mapTextureNameToIndex.find(entry_name);
  if (it != m_mapTextureNameToIndex.end())
  {
    // The texture already exists inside the per device cache,
    // which also means the MaterialMDL indices point to that already.
    return;
  }

  TextureMDLHost host = *shared; // Copy everything from the shared texture.

  host.m_texture.filtered_object   = 0; // Except for the texture objects.
  host.m_texture.unfiltered_object = 0;

  // Create filtered texture object
  CUaddress_mode addr_mode = (texture_shape == mi::neuraylib::ITarget_code::Texture_shape_cube) ? CU_TR_ADDRESS_MODE_CLAMP : CU_TR_ADDRESS_MODE_WRAP;

  // If the flag CU_TRSF_NORMALIZED_COORDINATES is not set, the only supported address mode is CU_TR_ADDRESS_MODE_CLAMP.
  host.m_textureDescription.addressMode[0] = addr_mode;
  host.m_textureDescription.addressMode[1] = addr_mode;
  host.m_textureDescription.addressMode[2] = addr_mode;

  host.m_textureDescription.filterMode = CU_TR_FILTER_MODE_LINEAR; // Bilinear filtering.

  CU_CHECK( cuTexObjectCreate(&host.m_texture.filtered_object, &host.m_resourceDescription, &host.m_textureDescription, nullptr) ); 

  if (texture_shape != mi::neuraylib::ITarget_code::Texture_shape_cube)
  {
    // Use a black border for access outside of the texture
    host.m_textureDescription.addressMode[0] = CU_TR_ADDRESS_MODE_BORDER;
    host.m_textureDescription.addressMode[1] = CU_TR_ADDRESS_MODE_BORDER;
    host.m_textureDescription.addressMode[2] = CU_TR_ADDRESS_MODE_BORDER;
    
    host.m_textureDescription.filterMode = CU_TR_FILTER_MODE_POINT;

    CU_CHECK( cuTexObjectCreate(&host.m_texture.unfiltered_object, &host.m_resourceDescription, &host.m_textureDescription, nullptr) ); 
  }

  // Get the new texture entry index for the cache.
  const int indexTexture = static_cast<int>(m_textureMDLHosts.size());
  MY_ASSERT(host.m_index == indexTexture); // Make sure the index of the shared texture is the same as the newly appended reused texture.
  // Store the texture array and object handles inside the vector of all textures on this device.
  m_textureMDLHosts.push_back(host); // This array is the same size on all devices!
  // Track the index inside the texture cache of this device.
  m_mapTextureNameToIndex[entry_name] = indexTexture;
}


bool Device::prepare_mbsdfs_part(mi::neuraylib::Mbsdf_part part,
                                 MbsdfHost& host,
                                 const mi::neuraylib::IBsdf_measurement* bsdf_measurement)
{
  mi::base::Handle<const mi::neuraylib::Bsdf_isotropic_data> dataset;

  switch (part)
  {
    case mi::neuraylib::MBSDF_DATA_REFLECTION:
      dataset = bsdf_measurement->get_reflection<mi::neuraylib::Bsdf_isotropic_data>();
      break;

    case mi::neuraylib::MBSDF_DATA_TRANSMISSION:
      dataset = bsdf_measurement->get_transmission<mi::neuraylib::Bsdf_isotropic_data>();
      break;
  }

  // No data, fine.
  if (!dataset)
  {
    return true;
  }

  // get dimensions
  uint2 res;

  res.x = dataset->get_resolution_theta();
  res.y = dataset->get_resolution_phi();

  unsigned int num_channels = (dataset->get_type() == mi::neuraylib::BSDF_SCALAR) ? 1 : 3;

  Mbsdf& mbsdf = host.m_mbsdf;

  mbsdf.Add(part, res, num_channels);

  // Get data.
  mi::base::Handle<const mi::neuraylib::IBsdf_buffer> buffer(dataset->get_bsdf_buffer());

  // {1, 3} * (index_theta_in * (res_phi * res_theta) + index_theta_out * res_phi + index_phi)
  const mi::Float32* src_data = buffer->get_data();

  // Prepare importance sampling data:
  // - For theta_in we will be able to perform a two stage CDF, 
  //   first to select theta_out, and second to select phi_out.
  // - Maximum component is used to "probability" in case of colored measurements.

  // CDF of the probability to select a certain theta_out for a given theta_in.
  const unsigned int cdf_theta_size = res.x * res.x;

  // For each of theta_in x theta_out combination, a CDF of the probabilities to select a certain theta_out is stored.
  const unsigned sample_data_size = cdf_theta_size + cdf_theta_size * res.y;
  
  float* sample_data = new float[sample_data_size];

  float* albedo_data = new float[res.x]; // albedo for sampling reflection and transmission

  float* sample_data_theta = sample_data;                  // begin of the first (theta) CDF
  float* sample_data_phi   = sample_data + cdf_theta_size; // begin of the second (phi) CDFs

  const float s_theta = (float) (M_PI * 0.5) / float(res.x); // step size
  const float s_phi   = (float) (M_PI)       / float(res.y); // step size

  float max_albedo = 0.0f;

  for (unsigned int t_in = 0; t_in < res.x; ++t_in)
  {
    float sum_theta     = 0.0f;
    float sintheta0_sqd = 0.0f;

    for (unsigned int t_out = 0; t_out < res.x; ++t_out)
    {
      const float sintheta1     = sinf(float(t_out + 1) * s_theta);
      const float sintheta1_sqd = sintheta1 * sintheta1;

      // BSDFs are symmetric: f(w_in, w_out) = f(w_out, w_in)
      // Take the average of both measurements.

      // Area of two the surface elements (the ones we are averaging).
      const float mu = (sintheta1_sqd - sintheta0_sqd) * s_phi * 0.5f;

      sintheta0_sqd = sintheta1_sqd;

      // Offset for both the thetas into the measurement data (select row in the volume).
      const unsigned int offset_phi  = (t_in  * res.x + t_out) * res.y;
      const unsigned int offset_phi2 = (t_out * res.x + t_in ) * res.y;

      // Build CDF for phi
      float sum_phi = 0.0f;

      for (unsigned int p_out = 0; p_out < res.y; ++p_out)
      {
        const unsigned int idx  = offset_phi  + p_out;
        const unsigned int idx2 = offset_phi2 + p_out;

        float value = 0.0f;

        if (num_channels == 3)
        {
          value = fmax(fmaxf(src_data[3 * idx  + 0], src_data[3 * idx  + 1]), fmaxf(src_data[3 * idx  + 2], 0.0f))
                + fmax(fmaxf(src_data[3 * idx2 + 0], src_data[3 * idx2 + 1]), fmaxf(src_data[3 * idx2 + 2], 0.0f));
        }
        else /* num_channels == 1 */
        {
          value = fmaxf(src_data[idx], 0.0f) + fmaxf(src_data[idx2], 0.0f);
        }

        sum_phi += value * mu;

        sample_data_phi[idx] = sum_phi;
      }

      // Normalize CDF for phi.
      for (unsigned int p_out = 0; p_out < res.y; ++p_out)
      {
        const unsigned int idx = offset_phi + p_out;

        sample_data_phi[idx] = sample_data_phi[idx] / sum_phi;
      }

      // Build CDF for theta.
      sum_theta += sum_phi;
      sample_data_theta[t_in * res.x + t_out] = sum_theta;
    }

    if (sum_theta > max_albedo)
    {
      max_albedo = sum_theta;
    }

    albedo_data[t_in] = sum_theta;

    // normalize CDF for theta
    for (unsigned int t_out = 0; t_out < res.x; ++t_out)
    {
      const unsigned int idx = t_in * res.x + t_out;

      sample_data_theta[idx] = sample_data_theta[idx] / sum_theta;
    }
  }

  // Copy entire CDF data buffer to GPU
  CUdeviceptr d_sample_obj = memAlloc(sample_data_size * sizeof(float), 4);
  CU_CHECK( cuMemcpyHtoD(d_sample_obj, sample_data, sample_data_size * sizeof(float)) );
  delete[] sample_data; // Copy is synchronous so this can be deleted now.

  CUdeviceptr d_albedo_obj = memAlloc(res.x * sizeof(float), 4);
  CU_CHECK( cuMemcpyHtoD(d_albedo_obj, albedo_data, res.x * sizeof(float)) );
  delete[] albedo_data; // Copy is synchronous so this can be deleted now.

  mbsdf.sample_data[part] = reinterpret_cast<float*>(d_sample_obj);
  mbsdf.albedo_data[part] = reinterpret_cast<float*>(d_albedo_obj);
  mbsdf.max_albedo[part]  = max_albedo;

  // Prepare evaluation data:
  // - Simply store the measured data in a volume texture.
  // - In case of color data, we store each sample in a vector4 to get texture support.
  unsigned int lookup_channels = (num_channels == 3) ? 4 : 1;

  // Make lookup data symmetric
  float* lookup_data = new float[lookup_channels * res.y * res.x * res.x];

  for (unsigned int t_in = 0; t_in < res.x; ++t_in)
  {
    for (unsigned int t_out = 0; t_out < res.x; ++t_out)
    {
      const unsigned int offset_phi  = (t_in  * res.x + t_out) * res.y;
      const unsigned int offset_phi2 = (t_out * res.x + t_in ) * res.y;

      for (unsigned int p_out = 0; p_out < res.y; ++p_out)
      {
        const unsigned int idx  = offset_phi  + p_out;
        const unsigned int idx2 = offset_phi2 + p_out;

        if (num_channels == 3)
        {
          lookup_data[4 * idx + 0] = (src_data[3 * idx + 0] + src_data[3 * idx2 + 0]) * 0.5f;
          lookup_data[4 * idx + 1] = (src_data[3 * idx + 1] + src_data[3 * idx2 + 1]) * 0.5f;
          lookup_data[4 * idx + 2] = (src_data[3 * idx + 2] + src_data[3 * idx2 + 2]) * 0.5f;
          lookup_data[4 * idx + 3] = 1.0f;
        }
        else
        {
          lookup_data[idx] = (src_data[idx] + src_data[idx2]) * 0.5f;
        }
      }
    }
  }

  // Allocate a 3D array on the GPU (phi_delta x theta_out x theta_in)
  CUDA_ARRAY3D_DESCRIPTOR& descArray3D = host.m_descArray3D[part];

  descArray3D = {};

  descArray3D.Width       = res.y;
  descArray3D.Height      = res.x;
  descArray3D.Depth       = res.x;
  descArray3D.Format      = CU_AD_FORMAT_FLOAT;
  descArray3D.NumChannels = (num_channels == 3) ? 4 : 1;
  descArray3D.Flags       = 0;

  // Track the current texture allocation size on this device.
  host.m_sizeBytesArray[part] += descArray3D.Width *
                                 descArray3D.Height *
                                 descArray3D.Depth * 
                                 descArray3D.NumChannels * sizeof(float);
  m_sizeMemoryTextureArrays += host.m_sizeBytesArray[part];

  CU_CHECK( cuArray3DCreate(&host.m_d_array[part], &descArray3D) );

  // Prepare and copy
  CUDA_MEMCPY3D params = {};

  params.srcMemoryType = CU_MEMORYTYPE_HOST;
  params.srcHost       = lookup_data;
  params.srcPitch      = res.y * sizeof(float) * descArray3D.NumChannels;
  params.srcHeight     = res.x;

  params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  params.dstArray      = host.m_d_array[part];

  params.WidthInBytes  = params.srcPitch;
  params.Height        = res.x;
  params.Depth         = res.x;

  CU_CHECK( cuMemcpy3D(&params) );

  delete[] lookup_data;

  CUDA_RESOURCE_DESC& resourceDesc = host.m_resourceDescription[part];

  resourceDesc = {};

  resourceDesc.resType = CU_RESOURCE_TYPE_ARRAY;
  resourceDesc.res.array.hArray = host.m_d_array[part];

  CUDA_TEXTURE_DESC& textureDesc = host.m_textureDescription; // Same settings for both parts.

  textureDesc = {};

  // Possible flags: CU_TRSF_READ_AS_INTEGER, CU_TRSF_NORMALIZED_COORDINATES, CU_TRSF_SRGB
  textureDesc.flags = CU_TRSF_NORMALIZED_COORDINATES;
  textureDesc.filterMode = CU_TR_FILTER_MODE_LINEAR; // Bilinear filtering by default.
  textureDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
  textureDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
  textureDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
  textureDesc.maxAnisotropy = 1;
  // DAR The default initialization handled all these. Just for code clarity.
  textureDesc.mipmapFilterMode    = CU_TR_FILTER_MODE_POINT;
  textureDesc.mipmapLevelBias     = 0.0f;
  textureDesc.minMipmapLevelClamp = 0.0f;
  textureDesc.maxMipmapLevelClamp = 0.0f;
  textureDesc.borderColor[0] = 0.0f;
  textureDesc.borderColor[1] = 0.0f;
  textureDesc.borderColor[2] = 0.0f;
  textureDesc.borderColor[3] = 0.0f;

  CUtexObject eval_tex_obj = 0; // This type is interchangeable with cudaTextureObject_t.
  
  CU_CHECK( cuTexObjectCreate(&eval_tex_obj, &resourceDesc, &textureDesc, nullptr) ); 
  
  mbsdf.eval_data[part] = eval_tex_obj;

  return true;
}


MbsdfHost* Device::prepareMBSDF(mi::neuraylib::ITransaction* transaction,
                                const mi::neuraylib::ITarget_code* code,
                                const int index)
{
  activateContext();
  synchronizeStream(); // PERF Required here?

  // Get access to the MBSDF data by the texture database name from the target code.
  mi::base::Handle<const mi::neuraylib::IBsdf_measurement> bsdf_measurement(transaction->access<mi::neuraylib::IBsdf_measurement>(code->get_bsdf_measurement(index)));

  MbsdfHost host;
  memset(&host, 0, sizeof(MbsdfHost));

  host.m_owner = this; // Track the device which owns the CUarray data.

  // Handle reflection part.
  if (!prepare_mbsdfs_part(mi::neuraylib::MBSDF_DATA_REFLECTION, host, bsdf_measurement.get()))
  {
    return nullptr;
  }

  // Handle transmission part.
  if (!prepare_mbsdfs_part(mi::neuraylib::MBSDF_DATA_TRANSMISSION, host, bsdf_measurement.get()))
  {
    return nullptr;
  }

  // Get the index of this Mbsdf inside the per Device's m_mbsdf vector.
  const int indexCache = static_cast<int>(m_mbsdfHosts.size());

  host.m_index = indexCache;
  // FIXME Implement a cache. (It's unlikely that multiple different materials use the same measured BSDF though.)

  // These are all MBSDFs inside the scene.
  m_mbsdfHosts.push_back(host);

  return &m_mbsdfHosts[indexCache];
}


void Device::shareMBSDF(const MbsdfHost* shared)
{
  activateContext();
  synchronizeStream(); // PERF Required here?

  MbsdfHost host = *shared;  // Copy everything from the shared MbsdfHost.

  for (int part = 0; part < 2; ++part)
  {
    host.m_mbsdf.eval_data[part] = 0; // Texture objects will be generated per-device.

    if (host.m_d_array[part]) // If there is CUarray data for the part, create a texture object from that.
    {
      CU_CHECK( cuTexObjectCreate(&host.m_mbsdf.eval_data[part], &host.m_resourceDescription[part], &host.m_textureDescription, nullptr) ); 
    }
  }

  // Get the index of this MbsdfHost inside the per Device's m_mbsdfHosts vector.
  const int indexCache = static_cast<int>(m_mbsdfHosts.size());
  MY_ASSERT(indexCache == host.m_index); // Make sure the index is the same on all devices.

  // These are all MBSDFs inside the scene.
  m_mbsdfHosts.push_back(host);
}


LightprofileHost* Device::prepareLightprofile(mi::neuraylib::ITransaction* transaction,
                                              const mi::neuraylib::ITarget_code* code,
                                              int index)
{
  activateContext();
  synchronizeStream(); // PERF Required here?

  // Get access to the light_profile data.
  mi::base::Handle<const mi::neuraylib::ILightprofile> lprof_nr(transaction->access<mi::neuraylib::ILightprofile>(code->get_light_profile(index)));

  uint2  res   = make_uint2(lprof_nr->get_resolution_theta(), lprof_nr->get_resolution_phi());
  float2 start = make_float2(lprof_nr->get_theta(0), lprof_nr->get_phi(0));
  float2 delta = make_float2(lprof_nr->get_theta(1) - start.x, lprof_nr->get_phi(1) - start.y);

  // phi-mayor: [res.x x res.y]
  const float* data = lprof_nr->get_data();

  // Compute total power.
  // Compute inverse CDF data for sampling.
  // Sampling will work on cells rather than grid nodes (used for evaluation).

  // First (res.x-1) for the cdf for sampling theta.
  // Rest (rex.x-1) * (res.y-1) for the individual cdfs for sampling phi (after theta).
  size_t cdf_data_size = (res.x - 1) + (res.x - 1) * (res.y - 1);
  
  float* cdf_data = new float[cdf_data_size];

  float debug_total_area = 0.0f;
  float sum_theta        = 0.0f;
  float total_power      = 0.0f;

  float cos_theta0 = cosf(start.x);

  for (unsigned int t = 0; t < res.x - 1; ++t)
  {
    const float cos_theta1 = cosf(start.x + float(t + 1) * delta.x);

    // Area of the patch (grid cell)
    // \mu = int_{theta0}^{theta1} sin{theta} \delta theta
    const float mu = cos_theta0 - cos_theta1;
    cos_theta0 = cos_theta1;

    // Build CDF for phi.
    float* cdf_data_phi = cdf_data + (res.x - 1) + t * (res.y - 1);
    
    float sum_phi = 0.0f;
    for (unsigned int p = 0; p < res.y - 1; ++p)
    {
      // The probability to select a patch corresponds to the value times area.
      // The value of a cell is the average of the corners.
      // Omit the *1/4 as we normalize in the end.
      float value = data[ p      * res.x + t    ]
                  + data[ p      * res.x + t + 1]
                  + data[(p + 1) * res.x + t    ]
                  + data[(p + 1) * res.x + t + 1];

      sum_phi        += value * mu;
      cdf_data_phi[p] = sum_phi;
      
      debug_total_area += mu;
    }

    // Normalize CDF for phi.
    for (unsigned int p = 0; p < res.y - 2; ++p)
    {
      cdf_data_phi[p] = (0.0f < sum_phi) ? (cdf_data_phi[p] / sum_phi) : 0.0f;
    }

    cdf_data_phi[res.y - 2] = 1.0f;

    // Build CDF for theta
    sum_theta  += sum_phi;
    cdf_data[t] = sum_theta;
  }

  total_power = sum_theta * 0.25f * delta.y;

  // Normalize CDF for theta.
  for (unsigned int t = 0; t < res.x - 2; ++t)
  {
    cdf_data[t] = (0.0f < sum_theta) ? (cdf_data[t] / sum_theta) : cdf_data[t];
  }

  cdf_data[res.x - 2] = 1.0f;

  // Copy entire CDF data buffer to GPU
  CUdeviceptr d_cdf_data_obj = memAlloc(cdf_data_size * sizeof(float), 4);
  CU_CHECK( cuMemcpyHtoD(d_cdf_data_obj, cdf_data, cdf_data_size * sizeof(float)) ); // Synchronous.
  delete[] cdf_data;

  // --------------------------------------------------------------------------------------------
  // Prepare evaluation data.
  //  - Use a 2d texture that allows bilinear interpolation.

  LightprofileHost host;
  memset(&host, 0, sizeof(LightprofileHost));

  // Allocate a 3D array on the GPU (phi_delta x theta_out x theta_in)
  CUDA_ARRAY3D_DESCRIPTOR& descArray3D = host.m_descArray3D;

  descArray3D = {};

  descArray3D.Width       = res.x;
  descArray3D.Height      = res.y;
  descArray3D.Depth       = 0; // A 2D array is allocated if only Depth extent is zero.
  descArray3D.Format      = CU_AD_FORMAT_FLOAT;
  descArray3D.NumChannels = 1;
  descArray3D.Flags       = 0;

  CU_CHECK( cuArray3DCreate(&host.m_d_array, &descArray3D) );

  // Copy data to GPU array
  CUDA_MEMCPY3D params = {};

  params.srcMemoryType = CU_MEMORYTYPE_HOST;
  params.srcHost       = data;
  params.srcPitch      = res.x * sizeof(float) * descArray3D.NumChannels;
  params.srcHeight     = res.y;

  params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  params.dstArray      = host.m_d_array;

  params.WidthInBytes  = params.srcPitch;
  params.Height        = res.y;
  params.Depth         = 1;

  CU_CHECK( cuMemcpy3D(&params) );
  
  // Create filtered texture object
  CUDA_RESOURCE_DESC& resourceDesc = host.m_resourceDescription;
  
  resourceDesc = {};

  resourceDesc.resType = CU_RESOURCE_TYPE_ARRAY;
  resourceDesc.res.array.hArray = host.m_d_array;

  CUDA_TEXTURE_DESC& textureDesc = host.m_textureDescription;
  
  textureDesc = {};

  // Possible flags: CU_TRSF_READ_AS_INTEGER, CU_TRSF_NORMALIZED_COORDINATES, CU_TRSF_SRGB
  textureDesc.flags = CU_TRSF_NORMALIZED_COORDINATES;
  textureDesc.filterMode = CU_TR_FILTER_MODE_LINEAR; // Bilinear filtering by default.
  textureDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP; // FIXME Shouldn't phi use wrap?
  textureDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
  textureDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
  textureDesc.maxAnisotropy = 1;
  // DAR The default initialization handled all these. Just for code clarity.
  textureDesc.mipmapFilterMode    = CU_TR_FILTER_MODE_POINT;
  textureDesc.mipmapLevelBias     = 0.0f;
  textureDesc.minMipmapLevelClamp = 0.0f;
  textureDesc.maxMipmapLevelClamp = 0.0f;
  textureDesc.borderColor[0] = 1.0f; // DAR DEBUG Why 1.0f? Shouldn't matter with clamp.
  textureDesc.borderColor[1] = 1.0f;
  textureDesc.borderColor[2] = 1.0f;
  textureDesc.borderColor[3] = 1.0f;

  CUtexObject tex_obj = 0; // This type is interchangeable with cudaTextureObject_t.
  
  CU_CHECK( cuTexObjectCreate(&tex_obj, &resourceDesc, &textureDesc, nullptr) ); 
  
  double multiplier = lprof_nr->get_candela_multiplier();

  host.m_profile = Lightprofile(tex_obj,
                                reinterpret_cast<float*>(d_cdf_data_obj),
                                res,
                                start,
                                delta,
                                float(multiplier),
                                float(total_power * multiplier));

  const int indexCache = static_cast<int>(m_lightprofileHosts.size());

  host.m_index = indexCache;
  // FIXME Implement a cache. (It's unlikely that multiple different materials use the same light profile though.)

  m_lightprofileHosts.push_back(host); // These are all light profiles in the scene.

  return &m_lightprofileHosts[indexCache];
}


void Device::shareLightprofile(const LightprofileHost* shared)
{
  activateContext();
  synchronizeStream(); // PERF Required here?

  LightprofileHost host = *shared;  // Copy everything from the shared MbsdfHost.

  host.m_profile.eval_data = 0; // Texture objects will be generated per-device.

  if (host.m_d_array)
  {
    CU_CHECK( cuTexObjectCreate(&host.m_profile.eval_data, &host.m_resourceDescription, &host.m_textureDescription, nullptr) ); 
  }

  const int indexCache = static_cast<int>(m_lightprofileHosts.size());
  MY_ASSERT(indexCache == host.m_index); // Make sure the index is the same on all devices.

  m_lightprofileHosts.push_back(host); // These are all Lightprofiles inside the scene.
}


void Device::initTextureHandler(std::vector<MaterialMDL*>& materialsMDL)
{
  activateContext();
  synchronizeStream(); // PERF Required here?

  const size_t numMaterialReferences = materialsMDL.size();

  for (MaterialMDL* material : materialsMDL)
  {
    MaterialDefinitionMDL materialDefinition = {}; // Set everything to zero.

    const size_t sizeArgumentBlock = material->getArgumentBlockSize();
    // If the material has an argument block, allocate and upload it.
    if (0 < sizeArgumentBlock)
    {
      materialDefinition.arg_block = memAlloc(sizeArgumentBlock, 16);

      CU_CHECK( cuMemcpyHtoD(materialDefinition.arg_block, material->getArgumentBlockData(), sizeArgumentBlock) );
    }

    // The MaterialMDL (per reference) only stores indices into the texture, MBSDF, and light profile caches (per device).
    // The current MDL runtime functions expect expanded arrays with the actual TextureMDL, Mbsdf and Lightprofile structures.
    // Expand them here.
    // Note that all three Texture_handler arrays are indexed zero-based and do not contain the invalid resource at index 0!

    Texture_handler handler = {};

    if (!material->m_indicesToTextures.empty())
    {
      std::vector<TextureMDL> textures;

      for (int i : material->m_indicesToTextures)
      {
        textures.push_back(m_textureMDLHosts[i].m_texture);
      }

      handler.num_textures = static_cast<unsigned int>(textures.size());
      handler.textures     = reinterpret_cast<TextureMDL*>(memAlloc(sizeof(TextureMDL) * textures.size(), 16));

      CU_CHECK( cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(handler.textures), textures.data(), sizeof(TextureMDL) * textures.size()) ); // Synchronous to not overwrite the host data in the loop.
    }

    if (!material->m_indicesToMBSDFs.empty())
    {
      std::vector<Mbsdf> mbsdfs;

      for (int i : material->m_indicesToMBSDFs)
      {
        mbsdfs.push_back(m_mbsdfHosts[i].m_mbsdf);
      }

      handler.num_mbsdfs = static_cast<unsigned int>(mbsdfs.size());
      handler.mbsdfs     = reinterpret_cast<Mbsdf*>(memAlloc(sizeof(Mbsdf) * mbsdfs.size(), 16));

      CU_CHECK( cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(handler.mbsdfs), mbsdfs.data(), sizeof(Mbsdf) * mbsdfs.size()) ); // Synchronous to not overwrite the host data in the loop.
    }

    if (!material->m_indicesToLightprofiles.empty())
    {
      std::vector<Lightprofile> profiles;

      for (int i : material->m_indicesToLightprofiles)
      {
        profiles.push_back(m_lightprofileHosts[i].m_profile);
      }

      handler.num_lightprofiles = static_cast<unsigned int>(profiles.size());
      handler.lightprofiles     = reinterpret_cast<Lightprofile*>(memAlloc(sizeof(Lightprofile) * profiles.size(), 16));

      CU_CHECK( cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(handler.lightprofiles), profiles.data(), sizeof(Lightprofile) * profiles.size()) ); // Synchronous to not overwrite the host data in the loop.
    }

    materialDefinition.texture_handler = reinterpret_cast<Texture_handler*>(memAlloc(sizeof(Texture_handler), 16));
      
    CU_CHECK( cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(materialDefinition.texture_handler), &handler, sizeof(Texture_handler)) );

    // Set the index to the shader cache. This defines which shader configuration is used.
    // If this indexShader is invalid, the instance with that material will not be put into the rendergraph!
    materialDefinition.indexShader = material->getShaderIndex();

    m_materialDefinitions.push_back(materialDefinition);
  }

  // Allocate the material definition device array, one entry per MDL material reference.
  m_systemData.materialDefinitionsMDL = 0;  // The MDL material parameter argument block, texture handler and index into the shader.

  if (!m_materialDefinitions.empty())
  {
    m_systemData.materialDefinitionsMDL = reinterpret_cast<MaterialDefinitionMDL*>(memAlloc(sizeof(MaterialDefinitionMDL) * m_materialDefinitions.size(), 16));
    
    CU_CHECK( cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(m_systemData.materialDefinitionsMDL), m_materialDefinitions.data(), sizeof(MaterialDefinitionMDL) * m_materialDefinitions.size()) );

    m_systemData.numMaterials = static_cast<int>(m_materialDefinitions.size());
  }

  m_systemData.shaderConfigurations = 0;

  if (!m_deviceShaderConfigurations.empty())
  {
    m_systemData.shaderConfigurations = reinterpret_cast<DeviceShaderConfiguration*>(memAlloc(sizeof(DeviceShaderConfiguration) * m_deviceShaderConfigurations.size(), 16));
    
    CU_CHECK( cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(m_systemData.shaderConfigurations), m_deviceShaderConfigurations.data(), sizeof(DeviceShaderConfiguration) * m_deviceShaderConfigurations.size()) );
  }

  m_isDirtySystemData = true;

  // Only when all MDL materials have been created, all information about the direct callable programs are available and the pipeline can be built.
  initPipeline(); 
}

