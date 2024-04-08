/* 
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "inc/NVMLImpl.h"

#include <cstring>
#include <iostream>

#ifdef _WIN32

static void *nvmlLoadFromDriverStore(const char* nvmlDllName)
{
  void* handle = NULL;

  // We are going to look for the OpenGL driver which lives next to nvoptix.dll and nvml.dll. 
  // 0 (null) will be returned if any errors occured.

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

    size_t newPathSize = strlen(regValue) + strlen(nvmlDllName) + 1;
    char*  dllPath = (char*) malloc(newPathSize);
    strcpy(dllPath, regValue);
    strcat(dllPath, nvmlDllName);

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

static void *nvmlLoadFromSystemDirectory(const char* nvmlDllName)
{
  // Get the size of the path first, then allocate.
  const unsigned int size = GetSystemDirectoryA(NULL, 0);
  if (size == 0)
  {
    // Couldn't get the system path size, so bail.
    return NULL;
  }
  
  // Alloc enough memory to concatenate with "\\nvml.dll".
  const size_t pathSize = size + 1 + strlen(nvmlDllName);

  char* systemPath = (char*) malloc(pathSize);

  if (GetSystemDirectoryA(systemPath, size) != size - 1)
  {
    // Something went wrong.
    free(systemPath);
    return NULL;
  }

  strcat(systemPath, "\\");
  strcat(systemPath, nvmlDllName);

  void* handle = LoadLibraryA(systemPath);

  free(systemPath);

  return handle;
}

static void* nvmlLoadWindowsDll(void)
{
  const char* nvmlDllName = "nvml.dll";

  void* handle = nvmlLoadFromDriverStore(nvmlDllName);

  if (!handle)
  {
    handle = nvmlLoadFromSystemDirectory(nvmlDllName);
    // If the nvml.dll is still not found here, something is wrong with the display driver installation.
  }

  return handle;
}
#endif


NVMLImpl::NVMLImpl()
  : m_handle(0)
{
  // Fill all existing NVML function pointers with nullptr.
  memset(&m_api, 0, sizeof(NVMLFunctionTable));
}

//NVMLImpl::~NVMLImpl()
//{
//}


// Helper function to get the entry point address in a loaded library just to abstract the platform in GET_FUNC macro.
static void* getFunc(void* handle, const char* name)
{
#ifdef _WIN32
  return GetProcAddress((HMODULE) handle, name);
#else
  return dlsym(handle, name);
#endif
}

bool NVMLImpl::initFunctionTable()
{
#ifdef _WIN32
  void* handle = nvmlLoadWindowsDll();
  if (!handle)
  {
    std::cerr << "nvml.dll not found\n";
    return false;
  }
#else
  void* handle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (!handle)
  {
    std::cerr << "libnvidia-ml.so.1 not found\n";
    return false;
  }
#endif

// Local macro to get the NVML entry point addresses and assign them to the NVMLFunctionTable members with the right type.
// Some of the NVML functions are versioned by a #define adding a version suffix (_v2, _v3) to the name,
// which requires a set of two macros to resolve the unversioned function name to the versioned one.

#define GET_FUNC_V(name) \
{ \
  const void* func = getFunc(handle, #name); \
  if (func) { \
    m_api.name = reinterpret_cast<name##_t>(func); \
  } else { \
    std::cerr << "ERROR: " << #name << " is nullptr\n"; \
    success = false; \
  } \
}

#define GET_FUNC(name) GET_FUNC_V(name)


  bool success = true;

  GET_FUNC(nvmlInit);
  //GET_FUNC(nvmlInitWithFlags);
  GET_FUNC(nvmlShutdown);
  //GET_FUNC(nvmlErrorString);
  //GET_FUNC(nvmlSystemGetDriverVersion);
  //GET_FUNC(nvmlSystemGetNVMLVersion);
  //GET_FUNC(nvmlSystemGetCudaDriverVersion);
  //GET_FUNC(nvmlSystemGetCudaDriverVersion_v2);
  //GET_FUNC(nvmlSystemGetProcessName);
  //GET_FUNC(nvmlUnitGetCount);
  //GET_FUNC(nvmlUnitGetHandleByIndex);
  //GET_FUNC(nvmlUnitGetUnitInfo);
  //GET_FUNC(nvmlUnitGetLedState);
  //GET_FUNC(nvmlUnitGetPsuInfo);
  //GET_FUNC(nvmlUnitGetTemperature);
  //GET_FUNC(nvmlUnitGetFanSpeedInfo);
  //GET_FUNC(nvmlUnitGetDevices);
  //GET_FUNC(nvmlSystemGetHicVersion);
  //GET_FUNC(nvmlDeviceGetCount);
  //GET_FUNC(nvmlDeviceGetAttributes);
  //GET_FUNC(nvmlDeviceGetHandleByIndex);
  //GET_FUNC(nvmlDeviceGetHandleBySerial);
  //GET_FUNC(nvmlDeviceGetHandleByUUID);
  GET_FUNC(nvmlDeviceGetHandleByPciBusId);
  //GET_FUNC(nvmlDeviceGetName);
  //GET_FUNC(nvmlDeviceGetBrand);
  //GET_FUNC(nvmlDeviceGetIndex);
  //GET_FUNC(nvmlDeviceGetSerial);
  //GET_FUNC(nvmlDeviceGetMemoryAffinity);
  //GET_FUNC(nvmlDeviceGetCpuAffinityWithinScope);
  //GET_FUNC(nvmlDeviceGetCpuAffinity);
  //GET_FUNC(nvmlDeviceSetCpuAffinity);
  //GET_FUNC(nvmlDeviceClearCpuAffinity);
  //GET_FUNC(nvmlDeviceGetTopologyCommonAncestor);
  //GET_FUNC(nvmlDeviceGetTopologyNearestGpus);
  //GET_FUNC(nvmlSystemGetTopologyGpuSet);
  //GET_FUNC(nvmlDeviceGetP2PStatus);
  //GET_FUNC(nvmlDeviceGetUUID);
  //GET_FUNC(nvmlVgpuInstanceGetMdevUUID);
  //GET_FUNC(nvmlDeviceGetMinorNumber);
  //GET_FUNC(nvmlDeviceGetBoardPartNumber);
  //GET_FUNC(nvmlDeviceGetInforomVersion);
  //GET_FUNC(nvmlDeviceGetInforomImageVersion);
  //GET_FUNC(nvmlDeviceGetInforomConfigurationChecksum);
  //GET_FUNC(nvmlDeviceValidateInforom);
  //GET_FUNC(nvmlDeviceGetDisplayMode);
  //GET_FUNC(nvmlDeviceGetDisplayActive);
  //GET_FUNC(nvmlDeviceGetPersistenceMode);
  //GET_FUNC(nvmlDeviceGetPciInfo);
  //GET_FUNC(nvmlDeviceGetMaxPcieLinkGeneration);
  //GET_FUNC(nvmlDeviceGetMaxPcieLinkWidth);
  //GET_FUNC(nvmlDeviceGetCurrPcieLinkGeneration);
  //GET_FUNC(nvmlDeviceGetCurrPcieLinkWidth);
  //GET_FUNC(nvmlDeviceGetPcieThroughput);
  //GET_FUNC(nvmlDeviceGetPcieReplayCounter);
  //GET_FUNC(nvmlDeviceGetClockInfo);
  //GET_FUNC(nvmlDeviceGetMaxClockInfo);
  //GET_FUNC(nvmlDeviceGetApplicationsClock);
  //GET_FUNC(nvmlDeviceGetDefaultApplicationsClock);
  //GET_FUNC(nvmlDeviceResetApplicationsClocks);
  //GET_FUNC(nvmlDeviceGetClock);
  //GET_FUNC(nvmlDeviceGetMaxCustomerBoostClock);
  //GET_FUNC(nvmlDeviceGetSupportedMemoryClocks);
  //GET_FUNC(nvmlDeviceGetSupportedGraphicsClocks);
  //GET_FUNC(nvmlDeviceGetAutoBoostedClocksEnabled);
  //GET_FUNC(nvmlDeviceSetAutoBoostedClocksEnabled);
  //GET_FUNC(nvmlDeviceSetDefaultAutoBoostedClocksEnabled);
  //GET_FUNC(nvmlDeviceGetFanSpeed);
  //GET_FUNC(nvmlDeviceGetFanSpeed_v2);
  //GET_FUNC(nvmlDeviceGetTemperature);
  //GET_FUNC(nvmlDeviceGetTemperatureThreshold);
  //GET_FUNC(nvmlDeviceGetPerformanceState);
  //GET_FUNC(nvmlDeviceGetCurrentClocksThrottleReasons);
  //GET_FUNC(nvmlDeviceGetSupportedClocksThrottleReasons);
  //GET_FUNC(nvmlDeviceGetPowerState);
  //GET_FUNC(nvmlDeviceGetPowerManagementMode);
  //GET_FUNC(nvmlDeviceGetPowerManagementLimit);
  //GET_FUNC(nvmlDeviceGetPowerManagementLimitConstraints);
  //GET_FUNC(nvmlDeviceGetPowerManagementDefaultLimit);
  //GET_FUNC(nvmlDeviceGetPowerUsage);
  //GET_FUNC(nvmlDeviceGetTotalEnergyConsumption);
  //GET_FUNC(nvmlDeviceGetEnforcedPowerLimit);
  //GET_FUNC(nvmlDeviceGetGpuOperationMode);
  //GET_FUNC(nvmlDeviceGetMemoryInfo);
  //GET_FUNC(nvmlDeviceGetComputeMode);
  //GET_FUNC(nvmlDeviceGetCudaComputeCapability);
  //GET_FUNC(nvmlDeviceGetEccMode);
  //GET_FUNC(nvmlDeviceGetBoardId);
  //GET_FUNC(nvmlDeviceGetMultiGpuBoard);
  //GET_FUNC(nvmlDeviceGetTotalEccErrors);
  //GET_FUNC(nvmlDeviceGetDetailedEccErrors);
  //GET_FUNC(nvmlDeviceGetMemoryErrorCounter);
  //GET_FUNC(nvmlDeviceGetUtilizationRates);
  //GET_FUNC(nvmlDeviceGetEncoderUtilization);
  //GET_FUNC(nvmlDeviceGetEncoderCapacity);
  //GET_FUNC(nvmlDeviceGetEncoderStats);
  //GET_FUNC(nvmlDeviceGetEncoderSessions);
  //GET_FUNC(nvmlDeviceGetDecoderUtilization);
  //GET_FUNC(nvmlDeviceGetFBCStats);
  //GET_FUNC(nvmlDeviceGetFBCSessions);
  //GET_FUNC(nvmlDeviceGetDriverModel);
  //GET_FUNC(nvmlDeviceGetVbiosVersion);
  //GET_FUNC(nvmlDeviceGetBridgeChipInfo);
  //GET_FUNC(nvmlDeviceGetComputeRunningProcesses);
  //GET_FUNC(nvmlDeviceGetGraphicsRunningProcesses);
  //GET_FUNC(nvmlDeviceOnSameBoard);
  //GET_FUNC(nvmlDeviceGetAPIRestriction);
  //GET_FUNC(nvmlDeviceGetSamples);
  //GET_FUNC(nvmlDeviceGetBAR1MemoryInfo);
  //GET_FUNC(nvmlDeviceGetViolationStatus);
  //GET_FUNC(nvmlDeviceGetAccountingMode);
  //GET_FUNC(nvmlDeviceGetAccountingStats);
  //GET_FUNC(nvmlDeviceGetAccountingPids);
  //GET_FUNC(nvmlDeviceGetAccountingBufferSize);
  //GET_FUNC(nvmlDeviceGetRetiredPages);
  //GET_FUNC(nvmlDeviceGetRetiredPages_v2);
  //GET_FUNC(nvmlDeviceGetRetiredPagesPendingStatus);
  //GET_FUNC(nvmlDeviceGetRemappedRows);
  //GET_FUNC(nvmlDeviceGetArchitecture);
  //GET_FUNC(nvmlUnitSetLedState);
  //GET_FUNC(nvmlDeviceSetPersistenceMode);
  //GET_FUNC(nvmlDeviceSetComputeMode);
  //GET_FUNC(nvmlDeviceSetEccMode);
  //GET_FUNC(nvmlDeviceClearEccErrorCounts);
  //GET_FUNC(nvmlDeviceSetDriverModel);
  //GET_FUNC(nvmlDeviceSetGpuLockedClocks);
  //GET_FUNC(nvmlDeviceResetGpuLockedClocks);
  //GET_FUNC(nvmlDeviceSetApplicationsClocks);
  //GET_FUNC(nvmlDeviceSetPowerManagementLimit);
  //GET_FUNC(nvmlDeviceSetGpuOperationMode);
  //GET_FUNC(nvmlDeviceSetAPIRestriction);
  //GET_FUNC(nvmlDeviceSetAccountingMode);
  //GET_FUNC(nvmlDeviceClearAccountingPids);
  GET_FUNC(nvmlDeviceGetNvLinkState);
  //GET_FUNC(nvmlDeviceGetNvLinkVersion);
  GET_FUNC(nvmlDeviceGetNvLinkCapability);
  GET_FUNC(nvmlDeviceGetNvLinkRemotePciInfo);
  //GET_FUNC(nvmlDeviceGetNvLinkErrorCounter);
  //GET_FUNC(nvmlDeviceResetNvLinkErrorCounters);
  //GET_FUNC(nvmlDeviceSetNvLinkUtilizationControl);
  //GET_FUNC(nvmlDeviceGetNvLinkUtilizationControl);
  //GET_FUNC(nvmlDeviceGetNvLinkUtilizationCounter);
  //GET_FUNC(nvmlDeviceFreezeNvLinkUtilizationCounter);
  //GET_FUNC(nvmlDeviceResetNvLinkUtilizationCounter);
  //GET_FUNC(nvmlEventSetCreate);
  //GET_FUNC(nvmlDeviceRegisterEvents);
  //GET_FUNC(nvmlDeviceGetSupportedEventTypes);
  //GET_FUNC(nvmlEventSetWait);
  //GET_FUNC(nvmlEventSetFree);
  //GET_FUNC(nvmlDeviceModifyDrainState);
  //GET_FUNC(nvmlDeviceQueryDrainState);
  //GET_FUNC(nvmlDeviceRemoveGpu);
  //GET_FUNC(nvmlDeviceDiscoverGpus);
  //GET_FUNC(nvmlDeviceGetFieldValues);
  //GET_FUNC(nvmlDeviceGetVirtualizationMode);
  //GET_FUNC(nvmlDeviceGetHostVgpuMode);
  //GET_FUNC(nvmlDeviceSetVirtualizationMode);
  //GET_FUNC(nvmlDeviceGetGridLicensableFeatures);
  //GET_FUNC(nvmlDeviceGetProcessUtilization);
  //GET_FUNC(nvmlDeviceGetSupportedVgpus);
  //GET_FUNC(nvmlDeviceGetCreatableVgpus);
  //GET_FUNC(nvmlVgpuTypeGetClass);
  //GET_FUNC(nvmlVgpuTypeGetName);
  //GET_FUNC(nvmlVgpuTypeGetDeviceID);
  //GET_FUNC(nvmlVgpuTypeGetFramebufferSize);
  //GET_FUNC(nvmlVgpuTypeGetNumDisplayHeads);
  //GET_FUNC(nvmlVgpuTypeGetResolution);
  //GET_FUNC(nvmlVgpuTypeGetLicense);
  //GET_FUNC(nvmlVgpuTypeGetFrameRateLimit);
  //GET_FUNC(nvmlVgpuTypeGetMaxInstances);
  //GET_FUNC(nvmlVgpuTypeGetMaxInstancesPerVm);
  //GET_FUNC(nvmlDeviceGetActiveVgpus);
  //GET_FUNC(nvmlVgpuInstanceGetVmID);
  //GET_FUNC(nvmlVgpuInstanceGetUUID);
  //GET_FUNC(nvmlVgpuInstanceGetVmDriverVersion);
  //GET_FUNC(nvmlVgpuInstanceGetFbUsage);
  //GET_FUNC(nvmlVgpuInstanceGetLicenseStatus);
  //GET_FUNC(nvmlVgpuInstanceGetType);
  //GET_FUNC(nvmlVgpuInstanceGetFrameRateLimit);
  //GET_FUNC(nvmlVgpuInstanceGetEccMode);
  //GET_FUNC(nvmlVgpuInstanceGetEncoderCapacity);
  //GET_FUNC(nvmlVgpuInstanceSetEncoderCapacity);
  //GET_FUNC(nvmlVgpuInstanceGetEncoderStats);
  //GET_FUNC(nvmlVgpuInstanceGetEncoderSessions);
  //GET_FUNC(nvmlVgpuInstanceGetFBCStats);
  //GET_FUNC(nvmlVgpuInstanceGetFBCSessions);
  //GET_FUNC(nvmlVgpuInstanceGetMetadata);
  //GET_FUNC(nvmlDeviceGetVgpuMetadata);
  //GET_FUNC(nvmlGetVgpuCompatibility);
  //GET_FUNC(nvmlDeviceGetPgpuMetadataString);
  //GET_FUNC(nvmlGetVgpuVersion);
  //GET_FUNC(nvmlSetVgpuVersion);
  //GET_FUNC(nvmlDeviceGetVgpuUtilization);
  //GET_FUNC(nvmlDeviceGetVgpuProcessUtilization);
  //GET_FUNC(nvmlVgpuInstanceGetAccountingMode);
  //GET_FUNC(nvmlVgpuInstanceGetAccountingPids);
  //GET_FUNC(nvmlVgpuInstanceGetAccountingStats);
  //GET_FUNC(nvmlVgpuInstanceClearAccountingPids);
  //GET_FUNC(nvmlGetBlacklistDeviceCount);
  //GET_FUNC(nvmlGetBlacklistDeviceInfoByIndex);
  //GET_FUNC(nvmlDeviceSetMigMode);
  //GET_FUNC(nvmlDeviceGetMigMode);
  //GET_FUNC(nvmlDeviceGetGpuInstanceProfileInfo);
  //GET_FUNC(nvmlDeviceGetGpuInstancePossiblePlacements);
  //GET_FUNC(nvmlDeviceGetGpuInstanceRemainingCapacity);
  //GET_FUNC(nvmlDeviceCreateGpuInstance);
  //GET_FUNC(nvmlGpuInstanceDestroy);
  //GET_FUNC(nvmlDeviceGetGpuInstances);
  //GET_FUNC(nvmlDeviceGetGpuInstanceById);
  //GET_FUNC(nvmlGpuInstanceGetInfo);
  //GET_FUNC(nvmlGpuInstanceGetComputeInstanceProfileInfo);
  //GET_FUNC(nvmlGpuInstanceGetComputeInstanceRemainingCapacity);
  //GET_FUNC(nvmlGpuInstanceCreateComputeInstance);
  //GET_FUNC(nvmlComputeInstanceDestroy);
  //GET_FUNC(nvmlGpuInstanceGetComputeInstances);
  //GET_FUNC(nvmlGpuInstanceGetComputeInstanceById);
  //GET_FUNC(nvmlComputeInstanceGetInfo);
  //GET_FUNC(nvmlDeviceIsMigDeviceHandle);
  //GET_FUNC(nvmlDeviceGetGpuInstanceId);
  //GET_FUNC(nvmlDeviceGetComputeInstanceId);
  //GET_FUNC(nvmlDeviceGetMaxMigDeviceCount);
  //GET_FUNC(nvmlDeviceGetMigDeviceHandleByIndex);
  //GET_FUNC(nvmlDeviceGetDeviceHandleFromMigDeviceHandle);

  return success;
}
