/* 
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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


Device::Device(const RendererStrategy strategy,
               const int ordinal,
               const int index,
               const int count,
               const int miss,
               const int interop,
               const unsigned int tex,
               const unsigned int pbo)
: m_strategy(strategy)
, m_ordinal(ordinal)
, m_index(index)
, m_count(count)
, m_miss(miss)
, m_interop(interop)
, m_tex(tex)
, m_pbo(pbo)
, m_nodeMask(0)
, m_launchWidth(0)
, m_ownsSharedBuffer(false)
, m_textureAlbedo(nullptr)
, m_textureCutout(nullptr)
, m_textureEnv(nullptr)
{
  initDeviceAttributes(); // CUDA

  OPTIX_CHECK( initFunctionTable() );

  // Create a CUDA Context and make it current to this thread.
  // PERF What is the best CU_CTX_SCHED_* setting here?
  // CU_CTX_MAP_HOST host to allow pinned memory.
  CU_CHECK( cuCtxCreate(&m_cudaContext, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, ordinal) ); 

  // PERF To make use of asynchronous copies. Currently not really anything happening in parallel due to synchronize calls.
  CU_CHECK( cuStreamCreate(&m_cudaStream, CU_STREAM_NON_BLOCKING) ); 

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

  OptixDeviceContextOptions options = {};

  options.logCallbackFunction = &callbackLogger;
  options.logCallbackData     = this; // This allows per device logs. It's currently printing the device ordinal.
  options.logCallbackLevel    = 3;    // Keep at warning level to suppress the disk cache messages.

  OPTIX_CHECK( m_api.optixDeviceContextCreate(m_cudaContext, &options, &m_optixContext) );

  initDeviceProperties(); // OptiX

  CU_CHECK( cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&m_d_systemData), sizeof(SystemData)) );
  m_isDirtySystemData = true; // Trigger SystemData update before the next launch.

  // Initialize all renderer system data.
  //m_systemData.rect                = make_int4(0, 0, 1, 1); // Currently unused.
  m_systemData.topObject           = 0;
  m_systemData.outputBuffer        = 0; // Deferred allocation. Only done in render() of the derived Device classes to allow for different memory spaces!
  m_systemData.tileBuffer          = 0; // For the final frame tiled renderer the intermediate buffer is only tileSize.
  m_systemData.texelBuffer         = 0; // For the final frame tiled renderer. Contains the accumulated result of the current tile.
  m_systemData.cameraDefinitions   = nullptr;
  m_systemData.lightDefinitions    = nullptr;
  m_systemData.materialDefinitions = nullptr;
  m_systemData.envTexture          = 0;
  m_systemData.envCDF_U            = nullptr;
  m_systemData.envCDF_V            = nullptr;
  m_systemData.resolution          = make_int2(1, 1); // Deferred allocation after setResolution() when m_isDirtyOutputBuffer == true.
  m_systemData.tileSize            = make_int2(8, 8); // Default value for multi-GPU tiling. Must be power-of-two values. (8x8 covers either 8x4 or 4x8 internal 2D warp shapes.)
  m_systemData.tileShift           = make_int2(3, 3); // The right-shift for the division by tileSize. 
  m_systemData.pathLengths         = make_int2(2, 5); // min, max
  m_systemData.deviceCount         = m_count; // The number of active devices.
  m_systemData.deviceIndex         = m_index; // This allows to distinguish multiple devices.
  m_systemData.iterationIndex      = 0;
  m_systemData.samplesSqrt         = 0; // Invalid value! Enforces that there is at least one setState() call before rendering.
  m_systemData.sceneEpsilon        = 500.0f * SCENE_EPSILON_SCALE;
  m_systemData.clockScale          = 1000.0f * CLOCK_FACTOR_SCALE;
  m_systemData.lensShader          = 0;
  m_systemData.numCameras          = 0;
  m_systemData.numLights           = 0;
  m_systemData.numMaterials        = 0;
  m_systemData.envWidth            = 0;
  m_systemData.envHeight           = 0;
  m_systemData.envIntegral         = 1.0f;
  m_systemData.envRotation         = 0.0f;

  m_isDirtyOutputBuffer = true; // First render call initializes it. This is done in the derived render() functions.

  m_moduleFilenames.resize(NUM_MODULE_IDENTIFIERS);

  // Starting with OptiX SDK 7.5.0 and CUDA 11.7 either PTX or OptiX IR input can be used to create modules.
  // Just initialize the m_moduleFilenames depending on the definition of USE_OPTIX_IR.
  // That is added to the project definitions inside the CMake script when OptiX SDK 7.5.0 and CUDA 11.7 or newer are found.
#if defined(USE_OPTIX_IR)
  m_moduleFilenames[MODULE_ID_RAYGENERATION]  = std::string("./rtigo3_core/raygeneration.optixir");
  m_moduleFilenames[MODULE_ID_EXCEPTION]      = std::string("./rtigo3_core/exception.optixir");
  m_moduleFilenames[MODULE_ID_MISS]           = std::string("./rtigo3_core/miss.optixir");
  m_moduleFilenames[MODULE_ID_CLOSESTHIT]     = std::string("./rtigo3_core/closesthit.optixir");
  m_moduleFilenames[MODULE_ID_ANYHIT]         = std::string("./rtigo3_core/anyhit.optixir");
  m_moduleFilenames[MODULE_ID_LENS_SHADER]    = std::string("./rtigo3_core/lens_shader.optixir");
  m_moduleFilenames[MODULE_ID_LIGHT_SAMPLE]   = std::string("./rtigo3_core/light_sample.optixir");
  m_moduleFilenames[MODULE_ID_BXDF_DIFFUSE]   = std::string("./rtigo3_core/bxdf_diffuse.optixir");
  m_moduleFilenames[MODULE_ID_BXDF_SPECULAR]  = std::string("./rtigo3_core/bxdf_specular.optixir");
  m_moduleFilenames[MODULE_ID_BXDF_GGX_SMITH] = std::string("./rtigo3_core/bxdf_ggx_smith.optixir");
#else
  m_moduleFilenames[MODULE_ID_RAYGENERATION]  = std::string("./rtigo3_core/raygeneration.ptx");
  m_moduleFilenames[MODULE_ID_EXCEPTION]      = std::string("./rtigo3_core/exception.ptx");
  m_moduleFilenames[MODULE_ID_MISS]           = std::string("./rtigo3_core/miss.ptx");
  m_moduleFilenames[MODULE_ID_CLOSESTHIT]     = std::string("./rtigo3_core/closesthit.ptx");
  m_moduleFilenames[MODULE_ID_ANYHIT]         = std::string("./rtigo3_core/anyhit.ptx");
  m_moduleFilenames[MODULE_ID_LENS_SHADER]    = std::string("./rtigo3_core/lens_shader.ptx");
  m_moduleFilenames[MODULE_ID_LIGHT_SAMPLE]   = std::string("./rtigo3_core/light_sample.ptx");
  m_moduleFilenames[MODULE_ID_BXDF_DIFFUSE]   = std::string("./rtigo3_core/bxdf_diffuse.ptx");
  m_moduleFilenames[MODULE_ID_BXDF_SPECULAR]  = std::string("./rtigo3_core/bxdf_specular.ptx");
  m_moduleFilenames[MODULE_ID_BXDF_GGX_SMITH] = std::string("./rtigo3_core/bxdf_ggx_smith.ptx");
#endif

  initPipeline();
}


Device::~Device()
{
  CU_CHECK_NO_THROW( cuCtxSetCurrent(m_cudaContext) ); // Activate this CUDA context. Not using activate() because this needs a no-throw check.
  CU_CHECK_NO_THROW( cuCtxSynchronize() );             // Make sure everthing running on this CUDA context has finished.

  delete m_textureEnv; // Allowed to be nullptr.
  delete m_textureCutout;
  delete m_textureAlbedo;

  CU_CHECK_NO_THROW( cuMemFree(reinterpret_cast<CUdeviceptr>(m_d_systemData)) );

  // The m_systemData.outputBuffer is allocated in different ways in the derived Device classes. Must be destroyed in their destructors.
  //CU_CHECK_NO_THROW( cuMemFree(m_systemData.outputBuffer) );

  CU_CHECK_NO_THROW( cuMemFree(m_systemData.tileBuffer) );
  CU_CHECK_NO_THROW( cuMemFree(m_systemData.texelBuffer) );

  CU_CHECK_NO_THROW( cuMemFree(reinterpret_cast<CUdeviceptr>(m_systemData.cameraDefinitions)) );
  CU_CHECK_NO_THROW( cuMemFree(reinterpret_cast<CUdeviceptr>(m_systemData.lightDefinitions)) );
  CU_CHECK_NO_THROW( cuMemFree(reinterpret_cast<CUdeviceptr>(m_systemData.materialDefinitions)) );

  for (size_t i = 0; i < m_geometryData.size(); ++i)
  {
    CU_CHECK_NO_THROW( cuMemFree(m_geometryData[i].d_attributes) ); // DAR FIXME Move these into an arena allocator.
    CU_CHECK_NO_THROW( cuMemFree(m_geometryData[i].d_indices) );
    CU_CHECK_NO_THROW( cuMemFree(m_geometryData[i].d_gas) );
  }

  CU_CHECK_NO_THROW( cuMemFree(m_d_ias) );

  CU_CHECK_NO_THROW( cuMemFree(reinterpret_cast<CUdeviceptr>(m_d_sbtRecordGeometryInstanceData)) ); // This holds all SBT records with istance data (hitgroup).
  CU_CHECK_NO_THROW( cuMemFree(m_d_sbtRecordHeaders) );                                             // This holds all SBT records without instance data.

  OPTIX_CHECK_NO_THROW( m_api.optixPipelineDestroy(m_pipeline) );
  OPTIX_CHECK_NO_THROW(m_api.optixDeviceContextDestroy(m_optixContext) );

  CU_CHECK_NO_THROW( cuStreamDestroy(m_cudaStream) );
  CU_CHECK_NO_THROW( cuCtxDestroy(m_cudaContext) );
}


void Device::initDeviceAttributes()
{
  char text[1024];
  text[1023] = 0;

  CU_CHECK( cuDeviceGetName(text, 1023, m_ordinal) );
  m_deviceName = std::string(text);

  CU_CHECK( cuDeviceGetPCIBusId(text, 1023, m_ordinal) );
  m_devicePciBusId = std::string(text);
  //std::cout << "domain:bus:device.function = " << m_devicePciBusId << '\n'; // DEBUG

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
  MY_ASSERT(NUM_RAYTYPES == 2); // The following code only works for two raytypes.

  OptixModuleCompileOptions mco = {};

  mco.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#if USE_DEBUG_EXCEPTIONS
  mco.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0; // No optimizations.
  mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;     // Full debug. Never profile kernels with this setting!
#else
  mco.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3; // All optimizations, is the default.
  // Keep generated line info for Nsight Compute profiling. (NVCC_OPTIONS use --generate-line-info in CMakeLists.txt)
#if (OPTIX_VERSION >= 70400)
  mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL; 
#else
  mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif
#endif

  OptixPipelineCompileOptions pco = {};

  pco.usesMotionBlur        = 0;
  pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
  pco.numPayloadValues      = 2;  // I need two to encode a 64-bit pointer to the per ray payload structure.
  pco.numAttributeValues    = 2;  // The minimum is two, for the barycentrics.
#if USE_DEBUG_EXCEPTIONS
  pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
                       OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                       OPTIX_EXCEPTION_FLAG_USER |
                       OPTIX_EXCEPTION_FLAG_DEBUG;
#else
  pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
  pco.pipelineLaunchParamsVariableName = "sysData";
#if (OPTIX_VERSION != 70000)
  // Only using built-in Triangles in this renderer. 
  // This is the recommended setting for best performance then.
  pco.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE; // New in OptiX 7.1.0.
#endif

  // Each source file results in one OptixModule.
  std::vector<OptixModule> modules(NUM_MODULE_IDENTIFIERS);

  // Create all modules:
  for (size_t i = 0; i < m_moduleFilenames.size(); ++i)
  {
    // Since OptiX 7.5.0 the program input can either be *.ptx source code or *.optixir binary code.
    // The module filenames are automatically switched between *.ptx or *.optixir extension based on the definition of USE_OPTIX_IR
    std::vector<char> programData = readData(m_moduleFilenames[i]);

#if (OPTIX_VERSION >= 70700)
    OPTIX_CHECK( m_api.optixModuleCreate(m_optixContext, &mco, &pco, programData.data(), programData.size(), nullptr, nullptr, &modules[i]) );
#else
    OPTIX_CHECK( m_api.optixModuleCreateFromPTX(m_optixContext, &mco, &pco, programData.data(), programData.size(), nullptr, nullptr, &modules[i]) );
#endif
  }

  std::vector<OptixProgramGroupDesc> programGroupDescriptions(NUM_PROGRAM_GROUP_IDS);
  memset(programGroupDescriptions.data(), 0, sizeof(OptixProgramGroupDesc) * programGroupDescriptions.size());
  
  OptixProgramGroupDesc* pgd;

  // All of these first because they are SbtRecordHeader and put into a single CUDA memory block.
  pgd = &programGroupDescriptions[PGID_RAYGENERATION];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->raygen.module = modules[MODULE_ID_RAYGENERATION];
  switch (m_strategy)
  {
    case RS_INTERACTIVE_SINGLE_GPU:
    case RS_INTERACTIVE_MULTI_GPU_ZERO_COPY:
    case RS_INTERACTIVE_MULTI_GPU_PEER_ACCESS:
      pgd->raygen.entryFunctionName = "__raygen__path_tracer";
      break;
    case RS_INTERACTIVE_MULTI_GPU_LOCAL_COPY:
      pgd->raygen.entryFunctionName = "__raygen__path_tracer_local_copy";
      break;
    default:
      std::cerr << "ERROR: initPipeline() unexpected RendererStrategy.\n";
      pgd->raygen.entryFunctionName = "__raygen__path_tracer";
      break;
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
  switch (m_miss)
  {
    case 0: // Black, not a light.
      pgd->miss.entryFunctionName = "__miss__env_null";
      break;
    case 1: // Constant white environment.
    default:
      pgd->miss.entryFunctionName = "__miss__env_constant";
      break;
    case 2: // Spherical HDR environment light.
      pgd->miss.entryFunctionName = "__miss__env_sphere";
      break;
  }

  pgd = &programGroupDescriptions[PGID_MISS_SHADOW];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->miss.module            = nullptr;
  pgd->miss.entryFunctionName = nullptr; // No miss program for shadow rays. 

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
  pgd = &programGroupDescriptions[PGID_LIGHT_ENV];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = (m_miss == 2) ? "__direct_callable__light_env_sphere" : "__direct_callable__light_env_constant"; // miss == 0 is not a light, use constant program.

  pgd = &programGroupDescriptions[PGID_LIGHT_AREA];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_parallelogram";

  // BxDF sample and eval
  pgd = &programGroupDescriptions[PGID_BRDF_DIFFUSE_SAMPLE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = modules[MODULE_ID_BXDF_DIFFUSE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__sample_brdf_diffuse";

  pgd = &programGroupDescriptions[PGID_BRDF_DIFFUSE_EVAL];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = modules[MODULE_ID_BXDF_DIFFUSE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__eval_brdf_diffuse";

  pgd = &programGroupDescriptions[PGID_BRDF_SPECULAR_SAMPLE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = modules[MODULE_ID_BXDF_SPECULAR];
  pgd->callables.entryFunctionNameDC = "__direct_callable__sample_brdf_specular";

  pgd = &programGroupDescriptions[PGID_BRDF_SPECULAR_EVAL];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = modules[MODULE_ID_BXDF_SPECULAR];
  pgd->callables.entryFunctionNameDC = "__direct_callable__eval_brdf_specular"; // black

  pgd = &programGroupDescriptions[PGID_BSDF_SPECULAR_SAMPLE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = modules[MODULE_ID_BXDF_SPECULAR];
  pgd->callables.entryFunctionNameDC = "__direct_callable__sample_bsdf_specular";

  pgd = &programGroupDescriptions[PGID_BSDF_SPECULAR_EVAL];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  // No implementation for __direct_callable__eval_bsdf_specular, it's specular.
  pgd->callables.moduleDC            = modules[MODULE_ID_BXDF_SPECULAR];
  pgd->callables.entryFunctionNameDC = "__direct_callable__eval_brdf_specular"; // black

  pgd = &programGroupDescriptions[PGID_BRDF_GGX_SMITH_SAMPLE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = modules[MODULE_ID_BXDF_GGX_SMITH];
  pgd->callables.entryFunctionNameDC = "__direct_callable__sample_brdf_ggx_smith";

  pgd = &programGroupDescriptions[PGID_BRDF_GGX_SMITH_EVAL];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = modules[MODULE_ID_BXDF_GGX_SMITH];
  pgd->callables.entryFunctionNameDC = "__direct_callable__eval_brdf_ggx_smith";

  pgd = &programGroupDescriptions[PGID_BSDF_GGX_SMITH_SAMPLE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = modules[MODULE_ID_BXDF_GGX_SMITH];
  pgd->callables.entryFunctionNameDC = "__direct_callable__sample_bsdf_ggx_smith";

  pgd = &programGroupDescriptions[PGID_BSDF_GGX_SMITH_EVAL];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  // No implementation for __direct_callable__eval_ggx_smith, it's specular.
  pgd->callables.moduleDC            = modules[MODULE_ID_BXDF_SPECULAR];
  pgd->callables.entryFunctionNameDC = "__direct_callable__eval_brdf_specular"; // black

  // HitGroups are using SbtRecordGeometryInstanceData and will be put into a separate CUDA memory block.
  pgd = &programGroupDescriptions[PGID_HIT_RADIANCE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleCH            = modules[MODULE_ID_CLOSESTHIT];
  pgd->hitgroup.entryFunctionNameCH = "__closesthit__radiance";

  pgd = &programGroupDescriptions[PGID_HIT_SHADOW];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleAH            = modules[MODULE_ID_ANYHIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__shadow";

  pgd = &programGroupDescriptions[PGID_HIT_RADIANCE_CUTOUT];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleCH            = modules[MODULE_ID_CLOSESTHIT];
  pgd->hitgroup.entryFunctionNameCH = "__closesthit__radiance";
  pgd->hitgroup.moduleAH            = modules[MODULE_ID_ANYHIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__radiance_cutout";

  pgd = &programGroupDescriptions[PGID_HIT_SHADOW_CUTOUT];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleAH            = modules[MODULE_ID_ANYHIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__shadow_cutout";

  OptixProgramGroupOptions pgo = {}; // This is a just placeholder.

  std::vector<OptixProgramGroup> programGroups(programGroupDescriptions.size());
  
  OPTIX_CHECK( m_api.optixProgramGroupCreate(m_optixContext, programGroupDescriptions.data(), (unsigned int) programGroupDescriptions.size(), &pgo, nullptr, nullptr, programGroups.data()) );

  OptixPipelineLinkOptions plo = {};

  plo.maxTraceDepth = 2;

#if (OPTIX_VERSION < 70700)
  // OptixPipelineLinkOptions debugLevel is only present in OptiX SDK versions before 7.7.0.
  #if USE_DEBUG_EXCEPTIONS
    plo.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL; // Full debug. Never profile kernels with this setting!
  #else
    // Keep generated line info for Nsight Compute profiling. (NVCC_OPTIONS use --generate-line-info in CMakeLists.txt)
    #if (OPTIX_VERSION >= 70400)
      plo.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL; 
    #else
      plo.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
    #endif
  #endif // USE_DEBUG_EXCEPTIONS
#endif // 70700
#if (OPTIX_VERSION == 70000)
  plo.overrideUsesMotionBlur = 0; // Does not exist in OptiX 7.1.0.
#endif

  OPTIX_CHECK( m_api.optixPipelineCreate(m_optixContext, &pco, &plo, programGroups.data(), (unsigned int) programGroups.size(), nullptr, nullptr, &m_pipeline) );

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

  // Arguments
  unsigned int directCallableStackSizeFromTraversal = ssp.dssDC; // maxDCDepth == 1 // FromTraversal: DC is invoked from IS or AH.      // Possible stack size optimizations.
  unsigned int directCallableStackSizeFromState     = ssp.dssDC; // maxDCDepth == 1 // FromState:     DC is invoked from RG, MS, or CH. // Possible stack size optimizations.
  unsigned int continuationStackSize = ssp.cssRG + cssCCTree + cssCHOrMSPlusCCTree * (std::max(1u, plo.maxTraceDepth) - 1u) +
                                       std::min(1u, plo.maxTraceDepth) * std::max(cssCHOrMSPlusCCTree, ssp.cssAH + ssp.cssIS);
  unsigned int maxTraversableGraphDepth = 2;

  OPTIX_CHECK( m_api.optixPipelineSetStackSize(m_pipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState, continuationStackSize, maxTraversableGraphDepth) );

  // Set up the fixed portion of the Shader Binding Table (SBT)

  // Put all SbtRecordHeader types in one CUdeviceptr.
  const int numHeaders = LAST_DIRECT_CALLABLE_ID - PGID_RAYGENERATION + 1;

  std::vector<SbtRecordHeader> sbtRecordHeaders(numHeaders);

  for (int i = 0; i < numHeaders; ++i)
  {
    OPTIX_CHECK( m_api.optixSbtRecordPackHeader(programGroups[PGID_RAYGENERATION + i], &sbtRecordHeaders[i]) );
  }

  CU_CHECK( cuMemAlloc(&m_d_sbtRecordHeaders, sizeof(SbtRecordHeader) * numHeaders) );
  CU_CHECK( cuMemcpyHtoDAsync(m_d_sbtRecordHeaders, sbtRecordHeaders.data(), sizeof(SbtRecordHeader) * numHeaders, m_cudaStream) );

  // Hit groups for radiance and shadow rays. These will be initialized later per instance.
  // This just provides the headers with the program group indices.

  // Note that the SBT record data field is uninitialized after these!
  // These are stored to be able to initialize the SBT hitGroup with the respective opaque and cutout shaders.
  OPTIX_CHECK( m_api.optixSbtRecordPackHeader(programGroups[PGID_HIT_RADIANCE],        &m_sbtRecordHitRadiance) );
  OPTIX_CHECK( m_api.optixSbtRecordPackHeader(programGroups[PGID_HIT_SHADOW],          &m_sbtRecordHitShadow) );

  OPTIX_CHECK( m_api.optixSbtRecordPackHeader(programGroups[PGID_HIT_RADIANCE_CUTOUT], &m_sbtRecordHitRadianceCutout) );
  OPTIX_CHECK( m_api.optixSbtRecordPackHeader(programGroups[PGID_HIT_SHADOW_CUTOUT],   &m_sbtRecordHitShadowCutout) );

  // Setup the OptixShaderBindingTable.

  m_sbt.raygenRecord            = m_d_sbtRecordHeaders + sizeof(SbtRecordHeader) * PGID_RAYGENERATION;

  m_sbt.exceptionRecord         = m_d_sbtRecordHeaders + sizeof(SbtRecordHeader) * PGID_EXCEPTION;

  m_sbt.missRecordBase          = m_d_sbtRecordHeaders + sizeof(SbtRecordHeader) * PGID_MISS_RADIANCE;
  m_sbt.missRecordStrideInBytes = (unsigned int) sizeof(SbtRecordHeader);
  m_sbt.missRecordCount         = NUM_RAYTYPES;

  // These are going to be setup after the RenderGraph has been built!
  //m_sbt.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>(m_d_sbtRecordGeometryInstanceData);
  //m_sbt.hitgroupRecordStrideInBytes = (unsigned int) sizeof(SbtRecordGeometryInstanceData);
  //m_sbt.hitgroupRecordCount         = NUM_RAYTYPES * numInstances;

  m_sbt.callablesRecordBase          = m_d_sbtRecordHeaders + sizeof(SbtRecordHeader) * FIRST_DIRECT_CALLABLE_ID;
  m_sbt.callablesRecordStrideInBytes = (unsigned int) sizeof(SbtRecordHeader);
  m_sbt.callablesRecordCount         = LAST_DIRECT_CALLABLE_ID - FIRST_DIRECT_CALLABLE_ID + 1;

  // After all required optixSbtRecordPackHeader, optixProgramGroupGetStackSize, and optixPipelineCreate
  // calls have been done, the OptixProgramGroup and OptixModule objects can be destroyed.
  for (auto pg: programGroups)
  {
    OPTIX_CHECK( m_api.optixProgramGroupDestroy(pg) );
  }

  for (auto m : modules)
  {
    OPTIX_CHECK(m_api.optixModuleDestroy(m));
  }
}


// FIXME Hardcocded textures. => See nvlink_shared which supports textures per material.
void Device::initTextures(std::map<std::string, Picture*> const& mapOfPictures)
{
  activateContext();
  synchronizeStream();

  std::map<std::string, Picture*>::const_iterator itAlbedo = mapOfPictures.find(std::string("albedo"));
  MY_ASSERT(itAlbedo != mapOfPictures.end());

  std::map<std::string, Picture*>::const_iterator itCutout = mapOfPictures.find(std::string("cutout"));
  MY_ASSERT(itCutout != mapOfPictures.end());

  std::map<std::string, Picture*>::const_iterator itEnv = mapOfPictures.find(std::string("environment"));

  m_textureAlbedo = new Texture();
  m_textureAlbedo->create(itAlbedo->second, IMAGE_FLAG_2D);

  m_textureCutout = new Texture();
  m_textureCutout->create(itCutout->second, IMAGE_FLAG_2D);

  if (itEnv != mapOfPictures.end())
  {
    m_textureEnv = new Texture();
    m_textureEnv->create(itEnv->second, IMAGE_FLAG_2D | IMAGE_FLAG_ENV);

    m_systemData.envTexture  = m_textureEnv->getTextureObject();
    m_systemData.envCDF_U    = reinterpret_cast<float*>(m_textureEnv->getCDF_U());
    m_systemData.envCDF_V    = reinterpret_cast<float*>(m_textureEnv->getCDF_V());
    m_systemData.envWidth    = m_textureEnv->getWidth();
    m_systemData.envHeight   = m_textureEnv->getHeight();
    m_systemData.envIntegral = m_textureEnv->getIntegral();
  }
}

void Device::initCameras(std::vector<CameraDefinition> const& cameras)
{
  activateContext();
  synchronizeStream();
    
  const int numCameras = static_cast<int>(cameras.size());
  MY_ASSERT(0 < numCameras); // There must be at least one camera defintion or the lens shaders won't work.

  // The default initialization of numCameras is 0.
  if (m_systemData.numCameras != numCameras)
  {
    if (m_systemData.cameraDefinitions != nullptr) // No need to call free on first time initialization.
    {
      CU_CHECK( cuMemFree(reinterpret_cast<CUdeviceptr>(m_systemData.cameraDefinitions)) );
    }
    CU_CHECK( cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&m_systemData.cameraDefinitions), sizeof(CameraDefinition) * numCameras) );
  }

  // Update the camera data.
  CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(m_systemData.cameraDefinitions), cameras.data(), sizeof(CameraDefinition) * numCameras, m_cudaStream) );
  m_systemData.numCameras = numCameras;

  m_isDirtySystemData = true;  // Trigger full update of the device system data on the next launch.
}

void Device::initLights(std::vector<LightDefinition> const& lights)
{
  activateContext();
  synchronizeStream();

  MY_ASSERT((sizeof(LightDefinition) & 15) == 0); // Verify float4 alignment.

  const int numLights = static_cast<int>(lights.size()); // This is allowed to be zero.

  // The default initialization of numLights is 0.
  if (m_systemData.numLights != numLights)
  {
    if (m_systemData.lightDefinitions != nullptr) // No need to call free on first time initialization.
    {
      CU_CHECK( cuMemFree(reinterpret_cast<CUdeviceptr>(m_systemData.lightDefinitions)) );
      m_systemData.lightDefinitions = nullptr; // It's valid to have zero lights.
    }
    if (0 < numLights)
    {
      CU_CHECK( cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&m_systemData.lightDefinitions), sizeof(LightDefinition) * numLights) );
    }
  }

  if (0 < numLights)
  {
    CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(m_systemData.lightDefinitions), lights.data(), sizeof(LightDefinition) * numLights, m_cudaStream) );
    m_systemData.numLights = numLights;
  }

  m_isDirtySystemData = true;  // Trigger full update of the device system data on the next launch.
}

void Device::initMaterials(std::vector<MaterialGUI> const& materialsGUI)
{
  activateContext();
  synchronizeStream();

  MY_ASSERT((sizeof(MaterialDefinition) & 15) == 0); // Verify float4 alignment.

  const int numMaterials = static_cast<int>(materialsGUI.size());
  MY_ASSERT(0 < numMaterials); // There must be at least one material or the hit shaders won't work.

  // The default initialization of numMaterials is 0.
  if (m_systemData.numMaterials != numMaterials) // FIXME Could grow only with additional capacity tracker.
  {
    if (m_systemData.materialDefinitions != nullptr) // No need to call free on first time initialization.
    {
      CU_CHECK( cuMemFree(reinterpret_cast<CUdeviceptr>(m_systemData.materialDefinitions)) );
    }
    CU_CHECK( cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&m_systemData.materialDefinitions), sizeof(MaterialDefinition) * numMaterials) );

    m_materials.resize(numMaterials);
  }

  // FIXME This could be made faster on GUI interactions on scenes with very many materials when really only copying the changed values.
  for (int i = 0; i < numMaterials; ++i)
  {
    MaterialGUI const&  materialGUI = materialsGUI[i];      // Material UI data in the host.
    MaterialDefinition& material    = m_materials[i]; // MaterialDefinition data on the host in device layout.

    MY_ASSERT(m_textureAlbedo != nullptr);
    MY_ASSERT(m_textureCutout != nullptr);

    material.textureAlbedo = (materialGUI.useAlbedoTexture) ? m_textureAlbedo->getTextureObject() : 0;
    material.textureCutout = (materialGUI.useCutoutTexture) ? m_textureCutout->getTextureObject() : 0;
    material.roughness     = materialGUI.roughness;
    material.indexBSDF     = materialGUI.indexBSDF;
    material.albedo        = materialGUI.albedo;
    material.absorption    = make_float3(0.0f); // Null coefficient means no absorption active.
    if (0.0f < materialGUI.absorptionScale)
    {
      // Calculate the effective absorption coefficient from the GUI parameters.
      // The absorption coefficient components must all be > 0.0f if absorptionScale > 0.0f.
      // Prevent logf(0.0f) which results in infinity.
      const float x = -logf(fmax(0.0001f, materialGUI.absorptionColor.x));
      const float y = -logf(fmax(0.0001f, materialGUI.absorptionColor.y));
      const float z = -logf(fmax(0.0001f, materialGUI.absorptionColor.z));
      material.absorption = make_float3(x, y, z) * materialGUI.absorptionScale;
      //std::cout << "absorption = (" << material.absorption.x << ", " << material.absorption.y << ", " << material.absorption.z << ")\n"; // DEBUG
    }
    material.ior   = materialGUI.ior;
    material.flags = (materialGUI.thinwalled) ? FLAG_THINWALLED : 0;
  }

  CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(m_systemData.materialDefinitions), m_materials.data(), sizeof(MaterialDefinition) * numMaterials, m_cudaStream) );

  m_isDirtySystemData = true;  // Trigger full update of the device system data on the next launch.
}

void Device::initScene(std::shared_ptr<sg::Group> root, const unsigned int numGeometries)
{
  activateContext();
  synchronizeStream();

  m_geometryData.resize(numGeometries);

  float matrix[12];

  // Set the affine matrix to identity by default.
  memset(matrix, 0, sizeof(float) * 12);
  matrix[ 0] = 1.0f;
  matrix[ 5] = 1.0f;
  matrix[10] = 1.0f;

  InstanceData data(~0u, -1, -1);

  traverseNode(root, matrix, data);

  createTLAS();

  createHitGroupRecords();
}


void Device::updateCamera(const int idCamera, CameraDefinition const& camera)
{
  activateContext();
  synchronizeStream();

  MY_ASSERT(idCamera < m_systemData.numCameras);
  CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(&m_systemData.cameraDefinitions[idCamera]), &camera, sizeof(CameraDefinition), m_cudaStream) );
}

void Device::updateLight(const int idLight, LightDefinition const& light)
{
  activateContext();
  synchronizeStream();

  MY_ASSERT(idLight < m_systemData.numLights);
  CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(&m_systemData.lightDefinitions[idLight]), &light, sizeof(LightDefinition), m_cudaStream) );
}

void Device::updateMaterial(const int idMaterial, MaterialGUI const& materialGUI)
{
  activateContext();
  synchronizeStream();

  MY_ASSERT(idMaterial < m_materials.size());
  MaterialDefinition& material = m_materials[idMaterial];  // MaterialDefinition on the host in device layout.

  MY_ASSERT(m_textureAlbedo != nullptr);
  MY_ASSERT(m_textureCutout != nullptr);

  const bool changeShader = (material.textureCutout != 0) != materialGUI.useCutoutTexture; // Cutout state wil be toggled?

  material.textureAlbedo = (materialGUI.useAlbedoTexture) ? m_textureAlbedo->getTextureObject() : 0;
  material.textureCutout = (materialGUI.useCutoutTexture) ? m_textureCutout->getTextureObject() : 0;
  material.roughness     = materialGUI.roughness;
  material.indexBSDF     = materialGUI.indexBSDF;
  material.albedo        = materialGUI.albedo;
  material.absorption = make_float3(0.0f); // Null means no absorption active.
  if (0.0f < materialGUI.absorptionScale)
  {
    // Calculate the effective absorption coefficient from the GUI parameters.
    // The absoption coefficient components must all be > 0.0f if absoprionScale > 0.0f.
    // Prevent logf(0.0f) which results in infinity.
    const float x = -logf(fmax(0.0001f, materialGUI.absorptionColor.x));
    const float y = -logf(fmax(0.0001f, materialGUI.absorptionColor.y));
    const float z = -logf(fmax(0.0001f, materialGUI.absorptionColor.z));
    material.absorption = make_float3(x, y, z) * materialGUI.absorptionScale;
    //std::cout << "absorption = (" << material.absorption.x << ", " << material.absorption.y << ", " << material.absorption.z << ")\n";
  }
  material.ior   = materialGUI.ior;
  material.flags = (materialGUI.thinwalled) ? FLAG_THINWALLED : 0;

  // Copy only the one changed material. No need to trigger an update of the system data, because the m_systemData.materialDefinitions pointer itself didn't change.
  CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(&m_systemData.materialDefinitions[idMaterial]), &material, sizeof(MaterialDefinition), m_cudaStream) );

  if (changeShader)
  {
    const unsigned int numInstances = static_cast<unsigned int>(m_instances.size());

    // PERF Maintain a list of instances per material ID.
    // Or better completely change the SBT to be per material shader and use the instance ID to index the material parameters which defines the SBT material shader offset.
    for (unsigned int inst = 0; inst < numInstances; ++inst)
    {
      if (idMaterial == m_instanceData[inst].idMaterial)
      {
        const unsigned int idx = inst * NUM_RAYTYPES;

        if (!materialGUI.useCutoutTexture)
        {
          // Only update the header to switch the program hit group. The SBT record data field doesn't change. 
          memcpy(m_sbtRecordGeometryInstanceData[idx    ].header, m_sbtRecordHitRadiance.header, OPTIX_SBT_RECORD_HEADER_SIZE);
          memcpy(m_sbtRecordGeometryInstanceData[idx + 1].header, m_sbtRecordHitShadow.header,   OPTIX_SBT_RECORD_HEADER_SIZE);
        }
        else
        {
          memcpy(m_sbtRecordGeometryInstanceData[idx    ].header, m_sbtRecordHitRadianceCutout.header, OPTIX_SBT_RECORD_HEADER_SIZE);
          memcpy(m_sbtRecordGeometryInstanceData[idx + 1].header, m_sbtRecordHitShadowCutout.header,   OPTIX_SBT_RECORD_HEADER_SIZE);
        }
        // PERF If the scene has many instances with few using the same material, this is faster. Otherwise the SBT can also be uploaded completely. See below.
        // Only copy the two SBT entries which changed. 
        CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(&m_d_sbtRecordGeometryInstanceData[idx]), &m_sbtRecordGeometryInstanceData[idx], sizeof(SbtRecordGeometryInstanceData) * NUM_RAYTYPES, m_cudaStream) );
      }
    }
    // Upload the whole SBT.
    //CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(m_d_sbtRecordGeometryInstanceData), m_sbtRecordGeometryInstanceData.data(), sizeof(SbtRecordGeometryInstanceData) * NUM_RAYTYPES * numInstances, m_cudaStream) );
  }
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


void Device::setState(DeviceState const& state)
{
  activateContext();
  synchronizeStream();

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
    m_isDirtySystemData = true;
  }

  if (m_systemData.lensShader != state.lensShader)
  {
    m_systemData.lensShader = state.lensShader;
    m_isDirtySystemData = true;
  }

  if (m_systemData.pathLengths != state.pathLengths)
  {
    m_systemData.pathLengths = state.pathLengths;
    m_isDirtySystemData = true;
  }
  
  if (m_systemData.sceneEpsilon != state.epsilonFactor * SCENE_EPSILON_SCALE)
  {
    m_systemData.sceneEpsilon = state.epsilonFactor * SCENE_EPSILON_SCALE;
    m_isDirtySystemData = true;
  }

  if (m_systemData.envRotation != state.envRotation)
  {
    // FIXME Implement free rotation with a rotation matrix.
    m_systemData.envRotation = state.envRotation;
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

// This is only overloaded by the derived DeviceMultiGPULocalCopy class.
void Device::compositor(Device* /* other */)
{
}


// m = a * b;
static void multiplyMatrix(float* m, const float* a, const float* b)
{
  m[ 0] = a[0] * b[0] + a[1] * b[4] + a[ 2] * b[ 8]; // + a[3] * 0
  m[ 1] = a[0] * b[1] + a[1] * b[5] + a[ 2] * b[ 9]; // + a[3] * 0
  m[ 2] = a[0] * b[2] + a[1] * b[6] + a[ 2] * b[10]; // + a[3] * 0
  m[ 3] = a[0] * b[3] + a[1] * b[7] + a[ 2] * b[11] + a[3]; // * 1
  
  m[ 4] = a[4] * b[0] + a[5] * b[4] + a[ 6] * b[ 8]; // + a[7] * 0
  m[ 5] = a[4] * b[1] + a[5] * b[5] + a[ 6] * b[ 9]; // + a[7] * 0
  m[ 6] = a[4] * b[2] + a[5] * b[6] + a[ 6] * b[10]; // + a[7] * 0
  m[ 7] = a[4] * b[3] + a[5] * b[7] + a[ 6] * b[11] + a[7]; // * 1

  m[ 8] = a[8] * b[0] + a[9] * b[4] + a[10] * b[ 8]; // + a[11] * 0
  m[ 9] = a[8] * b[1] + a[9] * b[5] + a[10] * b[ 9]; // + a[11] * 0
  m[10] = a[8] * b[2] + a[9] * b[6] + a[10] * b[10]; // + a[11] * 0
  m[11] = a[8] * b[3] + a[9] * b[7] + a[10] * b[11] + a[11]; // * 1
}

void Device::traverseNode(std::shared_ptr<sg::Node> node, float matrix[12], InstanceData data)
{
  switch (node->getType())
  {
    case sg::NodeType::NT_GROUP:
    {
      std::shared_ptr<sg::Group> group = std::dynamic_pointer_cast<sg::Group>(node);

      for (size_t i = 0; i < group->getNumChildren(); ++i)
      {
        traverseNode(group->getChild(i), matrix, data);
      }
    }
    break;

    case sg::NodeType::NT_INSTANCE:
    {
      std::shared_ptr<sg::Instance> instance = std::dynamic_pointer_cast<sg::Instance>(node);

      // Concatenate the transformations along the path.
      float trafo[12];
      multiplyMatrix(trafo, matrix, instance->getTransform());

      int idMaterial = instance->getMaterial();
      if (0 <= idMaterial)
      {
        data.idMaterial = idMaterial;  
      }

      int idLight = instance->getLight();
      if (0 <= idLight)
      {
        data.idLight = idLight;  
      }

      traverseNode(instance->getChild(), trafo, data);
    }
    break;

    case sg::NodeType::NT_TRIANGLES:
    {
      std::shared_ptr<sg::Triangles> geometry = std::dynamic_pointer_cast<sg::Triangles>(node);
      data.idGeometry = createGeometry(geometry);

      createInstance(m_geometryData[data.idGeometry].traversable, matrix, data);
    }
    break;
  }
}

unsigned int Device::createGeometry(std::shared_ptr<sg::Triangles> geometry)
{
  const unsigned int idGeometry = geometry->getId();
  MY_ASSERT(idGeometry < m_geometryData.size());

  // Did we create a geometry acceleration structure (GAS) for this Triangles node already?
  if (m_geometryData[idGeometry].traversable != 0)
  {
    return idGeometry; // Yes, reuse the GAS traversable.
  }
  
  std::vector<TriangleAttributes> const& attributes = geometry->getAttributes();
  std::vector<unsigned int>       const& indices    = geometry->getIndices();

  const size_t attributesSizeInBytes = sizeof(TriangleAttributes) * attributes.size();

  CUdeviceptr d_attributes;

  // DAR FIXME This all needs some Buffer class which maintains CUdeviceptr per Device, supporting separate allocations and peer-to-peer on multiple islands.
  CU_CHECK( cuMemAlloc(&d_attributes, attributesSizeInBytes) );
  CU_CHECK( cuMemcpyHtoDAsync(d_attributes, attributes.data(), attributesSizeInBytes, m_cudaStream) );

  const size_t indicesSizeInBytes = sizeof(int) * indices.size();

  CUdeviceptr d_indices;

  CU_CHECK( cuMemAlloc(&d_indices, indicesSizeInBytes) );
  CU_CHECK( cuMemcpyHtoDAsync(d_indices, indices.data(), indicesSizeInBytes, m_cudaStream) );

  OptixBuildInput buildInput = {};

  buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

  buildInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
  buildInput.triangleArray.vertexStrideInBytes = sizeof(TriangleAttributes);
  buildInput.triangleArray.numVertices         = static_cast<unsigned int>(attributes.size());
  buildInput.triangleArray.vertexBuffers       = &d_attributes;

  buildInput.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  buildInput.triangleArray.indexStrideInBytes = sizeof(unsigned int) * 3;

  buildInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(indices.size()) / 3;
  buildInput.triangleArray.indexBuffer      = d_indices;

  unsigned int inputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

  buildInput.triangleArray.flags         = inputFlags;
  buildInput.triangleArray.numSbtRecords = 1;

  OptixAccelBuildOptions accelBuildOptions = {};

  accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
  accelBuildOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes accelBufferSizes;
  
  OPTIX_CHECK( m_api.optixAccelComputeMemoryUsage(m_optixContext, &accelBuildOptions, &buildInput, 1, &accelBufferSizes) );

  CUdeviceptr d_gas; // This holds the acceleration structure.

  CU_CHECK( cuMemAlloc(&d_gas, accelBufferSizes.outputSizeInBytes) );

  CUdeviceptr d_tmp;

  CU_CHECK( cuMemAlloc(&d_tmp, accelBufferSizes.tempSizeInBytes) ); // Allocate the temp buffer last to reduce VRAM fragmentation.

  OptixTraversableHandle traversableHandle = 0; // This is the handle which gets returned.

  OPTIX_CHECK( m_api.optixAccelBuild(m_optixContext, m_cudaStream, 
                                     &accelBuildOptions, &buildInput, 1,
                                     d_tmp, accelBufferSizes.tempSizeInBytes,
                                     d_gas, accelBufferSizes.outputSizeInBytes, 
                                     &traversableHandle, nullptr, 0) );

  CU_CHECK( cuStreamSynchronize(m_cudaStream) );

  CU_CHECK( cuMemFree(d_tmp) );
  
  // Track the GeometryData to be able to set them in the SBT record GeometryInstanceData and free them on exit.
  // FIXME Move this to the top and use the fields directly.
  GeometryData geometryData;

  geometryData.traversable   = traversableHandle;
  geometryData.d_attributes  = d_attributes;
  geometryData.d_indices     = d_indices;
  geometryData.numAttributes = attributes.size();
  geometryData.numIndices    = indices.size();
  geometryData.d_gas         = d_gas;

  m_geometryData[idGeometry] = geometryData;
    
  return idGeometry;
}

void Device::createInstance(const OptixTraversableHandle traversable, float matrix[12], InstanceData const& data)
{
  MY_ASSERT(0 <= data.idMaterial);

  OptixInstance instance = {};
      
  const unsigned int id = static_cast<unsigned int>(m_instances.size());
  memcpy(instance.transform, matrix, sizeof(float) * 12);
  instance.instanceId        = id; // User defined instance index, queried with optixGetInstanceId().
  instance.visibilityMask    = 255;
  instance.sbtOffset         = id * NUM_RAYTYPES; // This controls the SBT instance offset! This must be set explicitly when each instance is using a separate BLAS.
  instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
  instance.traversableHandle = traversable;
    
  m_instances.push_back(instance); // OptiX instance data
  m_instanceData.push_back(data);  // SBT record data: idGeometry, idMaterial, idLight
}


void Device::createTLAS()
{
  // Construct the TLAS by attaching all flattened instances.
  CUdeviceptr d_instances;
  
  const size_t instancesSizeInBytes = sizeof(OptixInstance) * m_instances.size();

  CU_CHECK( cuMemAlloc(&d_instances, instancesSizeInBytes) ); // This will fail with CUDA_ERROR_INVALID_VALUE if m_instances is empty.
  CU_CHECK( cuMemcpyHtoDAsync(d_instances, m_instances.data(), instancesSizeInBytes, m_cudaStream) );

  OptixBuildInput instanceInput = {};

  instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  instanceInput.instanceArray.instances    = d_instances;
  instanceInput.instanceArray.numInstances = static_cast<unsigned int>(m_instances.size());

  OptixAccelBuildOptions accelBuildOptions = {};

  accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
  accelBuildOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;
  
  OptixAccelBufferSizes accelBufferSizes;

  OPTIX_CHECK( m_api.optixAccelComputeMemoryUsage(m_optixContext, &accelBuildOptions, &instanceInput, 1, &accelBufferSizes) );

  CU_CHECK( cuMemAlloc(&m_d_ias, accelBufferSizes.outputSizeInBytes) ); // This contains the top-level acceleration structure.

  CUdeviceptr d_tmp;
  
  CU_CHECK( cuMemAlloc(&d_tmp, accelBufferSizes.tempSizeInBytes) ); // Allocate the temp buffer last to reduce VRAM fragmentation.

  OPTIX_CHECK( m_api.optixAccelBuild(m_optixContext, m_cudaStream,
                                     &accelBuildOptions, &instanceInput, 1,
                                     d_tmp,   accelBufferSizes.tempSizeInBytes,
                                     m_d_ias, accelBufferSizes.outputSizeInBytes,
                                     &m_systemData.topObject, nullptr, 0));

  CU_CHECK( cuStreamSynchronize(m_cudaStream) );

  CU_CHECK( cuMemFree(d_tmp) );

  CU_CHECK( cuMemFree(d_instances) ); // Don't need the instances anymore.
}


void Device::createHitGroupRecords()
{
  const unsigned int numInstances = static_cast<unsigned int>(m_instances.size());

  m_sbtRecordGeometryInstanceData.resize(NUM_RAYTYPES * numInstances);

  for (unsigned int i = 0; i < numInstances; ++i)
  {
    InstanceData const& data = m_instanceData[i];
    const int idx = i * NUM_RAYTYPES; // idx == radiance ray, idx + 1 == shadow ray

    if (m_materials[data.idMaterial].textureCutout == 0)
    {
      // Only update the header to switch the program hit group. The SBT record data field doesn't change. 
      memcpy(m_sbtRecordGeometryInstanceData[idx    ].header, m_sbtRecordHitRadiance.header, OPTIX_SBT_RECORD_HEADER_SIZE);
      memcpy(m_sbtRecordGeometryInstanceData[idx + 1].header, m_sbtRecordHitShadow.header,   OPTIX_SBT_RECORD_HEADER_SIZE);
    }
    else
    {
      memcpy(m_sbtRecordGeometryInstanceData[idx    ].header, m_sbtRecordHitRadianceCutout.header, OPTIX_SBT_RECORD_HEADER_SIZE);
      memcpy(m_sbtRecordGeometryInstanceData[idx + 1].header, m_sbtRecordHitShadowCutout.header,   OPTIX_SBT_RECORD_HEADER_SIZE);
    }

    m_sbtRecordGeometryInstanceData[idx    ].data.attributes    = m_geometryData[data.idGeometry].d_attributes;
    m_sbtRecordGeometryInstanceData[idx    ].data.indices       = m_geometryData[data.idGeometry].d_indices;
    m_sbtRecordGeometryInstanceData[idx    ].data.materialIndex = data.idMaterial;
    m_sbtRecordGeometryInstanceData[idx    ].data.lightIndex    = data.idLight;

    m_sbtRecordGeometryInstanceData[idx + 1].data.attributes    = m_geometryData[data.idGeometry].d_attributes;
    m_sbtRecordGeometryInstanceData[idx + 1].data.indices       = m_geometryData[data.idGeometry].d_indices;
    m_sbtRecordGeometryInstanceData[idx + 1].data.materialIndex = data.idMaterial;
    m_sbtRecordGeometryInstanceData[idx + 1].data.lightIndex    = data.idLight;
  }

  CU_CHECK( cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&m_d_sbtRecordGeometryInstanceData), sizeof(SbtRecordGeometryInstanceData) * NUM_RAYTYPES * numInstances) );
  CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(m_d_sbtRecordGeometryInstanceData), m_sbtRecordGeometryInstanceData.data(), sizeof(SbtRecordGeometryInstanceData) * NUM_RAYTYPES * numInstances, m_cudaStream) );

  m_sbt.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>(m_d_sbtRecordGeometryInstanceData);
  m_sbt.hitgroupRecordStrideInBytes = (unsigned int) sizeof(SbtRecordGeometryInstanceData);
  m_sbt.hitgroupRecordCount         = NUM_RAYTYPES * numInstances;
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
