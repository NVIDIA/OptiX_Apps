/* 
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NVML_IMPL_H
#define NVML_IMPL_H

// This header ships with the CUDA Toolkit.
#include <nvml.h>


#define NVML_CHECK(call) \
{ \
  const nvmlReturn_t result = call; \
  if (result != NVML_SUCCESS) \
  { \
    std::ostringstream message; \
    message << "ERROR: " << __FILE__ << "(" << __LINE__ << "): " << #call << " (" << result << ")"; \
    MY_ASSERT(!"NVML_CHECK"); \
    throw std::runtime_error(message.str()); \
  } \
}

// These function entry point types and names support the automatic NVML API versioning inside nvml.h.
// Some of the NVML functions are versioned by a #define adding a version suffix (_v2, _v3) to the name,
// which requires a set of two macros to resolve the unversioned function name to the versioned one.

#define FUNC_T_V(name) name##_t
#define FUNC_T(name) FUNC_T_V(name)

#define FUNC_P_V(name) name##_t name
#define FUNC_P(name) FUNC_P_V(name)

typedef nvmlReturn_t (*FUNC_T(nvmlInit))(void);
//typedef nvmlReturn_t (*FUNC_T(nvmlInitWithFlags))(unsigned int flags);
typedef nvmlReturn_t (*FUNC_T(nvmlShutdown))(void);
//typedef const char*  (*FUNC_T(nvmlErrorString))(nvmlReturn_t result);
//typedef nvmlReturn_t (*FUNC_T(nvmlSystemGetDriverVersion))(char *version, unsigned int length);
//typedef nvmlReturn_t (*FUNC_T(nvmlSystemGetNVMLVersion))(char *version, unsigned int length);
//typedef nvmlReturn_t (*FUNC_T(nvmlSystemGetCudaDriverVersion))(int *cudaDriverVersion);
//typedef nvmlReturn_t (*FUNC_T(nvmlSystemGetCudaDriverVersion_v2))(int *cudaDriverVersion);
//typedef nvmlReturn_t (*FUNC_T(nvmlSystemGetProcessName))(unsigned int pid, char *name, unsigned int length);
//typedef nvmlReturn_t (*FUNC_T(nvmlUnitGetCount))(unsigned int *unitCount);
//typedef nvmlReturn_t (*FUNC_T(nvmlUnitGetHandleByIndex))(unsigned int index, nvmlUnit_t *unit);
//typedef nvmlReturn_t (*FUNC_T(nvmlUnitGetUnitInfo))(nvmlUnit_t unit, nvmlUnitInfo_t *info);
//typedef nvmlReturn_t (*FUNC_T(nvmlUnitGetLedState))(nvmlUnit_t unit, nvmlLedState_t *state);
//typedef nvmlReturn_t (*FUNC_T(nvmlUnitGetPsuInfo))(nvmlUnit_t unit, nvmlPSUInfo_t *psu);
//typedef nvmlReturn_t (*FUNC_T(nvmlUnitGetTemperature))(nvmlUnit_t unit, unsigned int type, unsigned int *temp);
//typedef nvmlReturn_t (*FUNC_T(nvmlUnitGetFanSpeedInfo))(nvmlUnit_t unit, nvmlUnitFanSpeeds_t *fanSpeeds);
//typedef nvmlReturn_t (*FUNC_T(nvmlUnitGetDevices))(nvmlUnit_t unit, unsigned int *deviceCount, nvmlDevice_t *devices);
//typedef nvmlReturn_t (*FUNC_T(nvmlSystemGetHicVersion))(unsigned int *hwbcCount, nvmlHwbcEntry_t *hwbcEntries);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetCount))(unsigned int *deviceCount);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetAttributes))(nvmlDevice_t device, nvmlDeviceAttributes_t *attributes);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetHandleByIndex))(unsigned int index, nvmlDevice_t *device);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetHandleBySerial))(const char *serial, nvmlDevice_t *device);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetHandleByUUID))(const char *uuid, nvmlDevice_t *device);
typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetHandleByPciBusId))(const char *pciBusId, nvmlDevice_t *device);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetName))(nvmlDevice_t device, char *name, unsigned int length);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetBrand))(nvmlDevice_t device, nvmlBrandType_t *type);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetIndex))(nvmlDevice_t device, unsigned int *index);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetSerial))(nvmlDevice_t device, char *serial, unsigned int length);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetMemoryAffinity))(nvmlDevice_t device, unsigned int nodeSetSize, unsigned long *nodeSet, nvmlAffinityScope_t scope);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetCpuAffinityWithinScope))(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet, nvmlAffinityScope_t scope);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetCpuAffinity))(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceSetCpuAffinity))(nvmlDevice_t device);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceClearCpuAffinity))(nvmlDevice_t device);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetTopologyCommonAncestor))(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t *pathInfo);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetTopologyNearestGpus))(nvmlDevice_t device, nvmlGpuTopologyLevel_t level, unsigned int *count, nvmlDevice_t *deviceArray);
//typedef nvmlReturn_t (*FUNC_T(nvmlSystemGetTopologyGpuSet))(unsigned int cpuNumber, unsigned int *count, nvmlDevice_t *deviceArray);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetP2PStatus))(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex,nvmlGpuP2PStatus_t *p2pStatus);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetUUID))(nvmlDevice_t device, char *uuid, unsigned int length);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuInstanceGetMdevUUID))(nvmlVgpuInstance_t vgpuInstance, char *mdevUuid, unsigned int size);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetMinorNumber))(nvmlDevice_t device, unsigned int *minorNumber);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetBoardPartNumber))(nvmlDevice_t device, char* partNumber, unsigned int length);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetInforomVersion))(nvmlDevice_t device, nvmlInforomObject_t object, char *version, unsigned int length);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetInforomImageVersion))(nvmlDevice_t device, char *version, unsigned int length);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetInforomConfigurationChecksum))(nvmlDevice_t device, unsigned int *checksum);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceValidateInforom))(nvmlDevice_t device);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetDisplayMode))(nvmlDevice_t device, nvmlEnableState_t *display);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetDisplayActive))(nvmlDevice_t device, nvmlEnableState_t *isActive);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetPersistenceMode))(nvmlDevice_t device, nvmlEnableState_t *mode);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetPciInfo))(nvmlDevice_t device, nvmlPciInfo_t *pci);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetMaxPcieLinkGeneration))(nvmlDevice_t device, unsigned int *maxLinkGen);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetMaxPcieLinkWidth))(nvmlDevice_t device, unsigned int *maxLinkWidth);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetCurrPcieLinkGeneration))(nvmlDevice_t device, unsigned int *currLinkGen);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetCurrPcieLinkWidth))(nvmlDevice_t device, unsigned int *currLinkWidth);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetPcieThroughput))(nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int *value);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetPcieReplayCounter))(nvmlDevice_t device, unsigned int *value);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetClockInfo))(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetMaxClockInfo))(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetApplicationsClock))(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetDefaultApplicationsClock))(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceResetApplicationsClocks))(nvmlDevice_t device);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetClock))(nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int *clockMHz);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetMaxCustomerBoostClock))(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetSupportedMemoryClocks))(nvmlDevice_t device, unsigned int *count, unsigned int *clocksMHz);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetSupportedGraphicsClocks))(nvmlDevice_t device, unsigned int memoryClockMHz, unsigned int *count, unsigned int *clocksMHz);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetAutoBoostedClocksEnabled))(nvmlDevice_t device, nvmlEnableState_t *isEnabled, nvmlEnableState_t *defaultIsEnabled);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceSetAutoBoostedClocksEnabled))(nvmlDevice_t device, nvmlEnableState_t enabled);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceSetDefaultAutoBoostedClocksEnabled))(nvmlDevice_t device, nvmlEnableState_t enabled, unsigned int flags);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetFanSpeed))(nvmlDevice_t device, unsigned int *speed);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetFanSpeed_v2))(nvmlDevice_t device, unsigned int fan, unsigned int * speed);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetTemperature))(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int *temp);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetTemperatureThreshold))(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, unsigned int *temp);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetPerformanceState))(nvmlDevice_t device, nvmlPstates_t *pState);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetCurrentClocksThrottleReasons))(nvmlDevice_t device, unsigned long long *clocksThrottleReasons);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetSupportedClocksThrottleReasons))(nvmlDevice_t device, unsigned long long *supportedClocksThrottleReasons);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetPowerState))(nvmlDevice_t device, nvmlPstates_t *pState);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetPowerManagementMode))(nvmlDevice_t device, nvmlEnableState_t *mode);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetPowerManagementLimit))(nvmlDevice_t device, unsigned int *limit);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetPowerManagementLimitConstraints))(nvmlDevice_t device, unsigned int *minLimit, unsigned int *maxLimit);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetPowerManagementDefaultLimit))(nvmlDevice_t device, unsigned int *defaultLimit);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetPowerUsage))(nvmlDevice_t device, unsigned int *power);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetTotalEnergyConsumption))(nvmlDevice_t device, unsigned long long *energy);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetEnforcedPowerLimit))(nvmlDevice_t device, unsigned int *limit);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetGpuOperationMode))(nvmlDevice_t device, nvmlGpuOperationMode_t *current, nvmlGpuOperationMode_t *pending);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetMemoryInfo))(nvmlDevice_t device, nvmlMemory_t *memory);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetComputeMode))(nvmlDevice_t device, nvmlComputeMode_t *mode);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetCudaComputeCapability))(nvmlDevice_t device, int *major, int *minor);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetEccMode))(nvmlDevice_t device, nvmlEnableState_t *current, nvmlEnableState_t *pending);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetBoardId))(nvmlDevice_t device, unsigned int *boardId);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetMultiGpuBoard))(nvmlDevice_t device, unsigned int *multiGpuBool);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetTotalEccErrors))(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long *eccCounts);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetDetailedEccErrors))(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlEccErrorCounts_t *eccCounts);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetMemoryErrorCounter))(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlMemoryLocation_t locationType, unsigned long long *count);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetUtilizationRates))(nvmlDevice_t device, nvmlUtilization_t *utilization);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetEncoderUtilization))(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetEncoderCapacity))(nvmlDevice_t device, nvmlEncoderType_t encoderQueryType, unsigned int *encoderCapacity);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetEncoderStats))(nvmlDevice_t device, unsigned int *sessionCount, unsigned int *averageFps, unsigned int *averageLatency);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetEncoderSessions))(nvmlDevice_t device, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfos);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetDecoderUtilization))(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetFBCStats))(nvmlDevice_t device, nvmlFBCStats_t *fbcStats);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetFBCSessions))(nvmlDevice_t device, unsigned int *sessionCount, nvmlFBCSessionInfo_t *sessionInfo);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetDriverModel))(nvmlDevice_t device, nvmlDriverModel_t *current, nvmlDriverModel_t *pending);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetVbiosVersion))(nvmlDevice_t device, char *version, unsigned int length);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetBridgeChipInfo))(nvmlDevice_t device, nvmlBridgeChipHierarchy_t *bridgeHierarchy);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetComputeRunningProcesses))(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetGraphicsRunningProcesses))(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceOnSameBoard))(nvmlDevice_t device1, nvmlDevice_t device2, int *onSameBoard);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetAPIRestriction))(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t *isRestricted);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetSamples))(nvmlDevice_t device, nvmlSamplingType_t type, unsigned long long lastSeenTimeStamp, nvmlValueType_t *sampleValType, unsigned int *sampleCount, nvmlSample_t *samples);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetBAR1MemoryInfo))(nvmlDevice_t device, nvmlBAR1Memory_t *bar1Memory);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetViolationStatus))(nvmlDevice_t device, nvmlPerfPolicyType_t perfPolicyType, nvmlViolationTime_t *violTime);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetAccountingMode))(nvmlDevice_t device, nvmlEnableState_t *mode);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetAccountingStats))(nvmlDevice_t device, unsigned int pid, nvmlAccountingStats_t *stats);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetAccountingPids))(nvmlDevice_t device, unsigned int *count, unsigned int *pids);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetAccountingBufferSize))(nvmlDevice_t device, unsigned int *bufferSize);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetRetiredPages))(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int *pageCount, unsigned long long *addresses);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetRetiredPages_v2))(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int *pageCount, unsigned long long *addresses, unsigned long long *timestamps);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetRetiredPagesPendingStatus))(nvmlDevice_t device, nvmlEnableState_t *isPending); 
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetRemappedRows))(nvmlDevice_t device, unsigned int *corrRows, unsigned int *uncRows, unsigned int *isPending, unsigned int *failureOccurred);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetArchitecture))(nvmlDevice_t device, nvmlDeviceArchitecture_t *arch);
//typedef nvmlReturn_t (*FUNC_T(nvmlUnitSetLedState))(nvmlUnit_t unit, nvmlLedColor_t color);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceSetPersistenceMode))(nvmlDevice_t device, nvmlEnableState_t mode);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceSetComputeMode))(nvmlDevice_t device, nvmlComputeMode_t mode);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceSetEccMode))(nvmlDevice_t device, nvmlEnableState_t ecc);  
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceClearEccErrorCounts))(nvmlDevice_t device, nvmlEccCounterType_t counterType);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceSetDriverModel))(nvmlDevice_t device, nvmlDriverModel_t driverModel, unsigned int flags);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceSetGpuLockedClocks))(nvmlDevice_t device, unsigned int minGpuClockMHz, unsigned int maxGpuClockMHz);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceResetGpuLockedClocks))(nvmlDevice_t device);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceSetApplicationsClocks))(nvmlDevice_t device, unsigned int memClockMHz, unsigned int graphicsClockMHz);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceSetPowerManagementLimit))(nvmlDevice_t device, unsigned int limit);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceSetGpuOperationMode))(nvmlDevice_t device, nvmlGpuOperationMode_t mode);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceSetAPIRestriction))(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t isRestricted);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceSetAccountingMode))(nvmlDevice_t device, nvmlEnableState_t mode);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceClearAccountingPids))(nvmlDevice_t device);
typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetNvLinkState))(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetNvLinkVersion))(nvmlDevice_t device, unsigned int link, unsigned int *version);
typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetNvLinkCapability))(nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int *capResult); 
typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetNvLinkRemotePciInfo))(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetNvLinkErrorCounter))(nvmlDevice_t device, unsigned int link, nvmlNvLinkErrorCounter_t counter, unsigned long long *counterValue);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceResetNvLinkErrorCounters))(nvmlDevice_t device, unsigned int link);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceSetNvLinkUtilizationControl))(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t *control, unsigned int reset);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetNvLinkUtilizationControl))(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t *control);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetNvLinkUtilizationCounter))(nvmlDevice_t device, unsigned int link, unsigned int counter, unsigned long long *rxcounter, unsigned long long *txcounter);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceFreezeNvLinkUtilizationCounter))(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlEnableState_t freeze);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceResetNvLinkUtilizationCounter))(nvmlDevice_t device, unsigned int link, unsigned int counter);
//typedef nvmlReturn_t (*FUNC_T(nvmlEventSetCreate))(nvmlEventSet_t *set);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceRegisterEvents))(nvmlDevice_t device, unsigned long long eventTypes, nvmlEventSet_t set);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetSupportedEventTypes))(nvmlDevice_t device, unsigned long long *eventTypes);
//typedef nvmlReturn_t (*FUNC_T(nvmlEventSetWait))(nvmlEventSet_t set, nvmlEventData_t * data, unsigned int timeoutms);
//typedef nvmlReturn_t (*FUNC_T(nvmlEventSetFree))(nvmlEventSet_t set);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceModifyDrainState))(nvmlPciInfo_t *pciInfo, nvmlEnableState_t newState);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceQueryDrainState))(nvmlPciInfo_t *pciInfo, nvmlEnableState_t *currentState);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceRemoveGpu))(nvmlPciInfo_t *pciInfo, nvmlDetachGpuState_t gpuState, nvmlPcieLinkState_t linkState);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceDiscoverGpus))(nvmlPciInfo_t *pciInfo);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetFieldValues))(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t *values);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetVirtualizationMode))(nvmlDevice_t device, nvmlGpuVirtualizationMode_t *pVirtualMode);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetHostVgpuMode))(nvmlDevice_t device, nvmlHostVgpuMode_t *pHostVgpuMode);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceSetVirtualizationMode))(nvmlDevice_t device, nvmlGpuVirtualizationMode_t virtualMode);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetGridLicensableFeatures))(nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetProcessUtilization))(nvmlDevice_t device, nvmlProcessUtilizationSample_t *utilization, unsigned int *processSamplesCount, unsigned long long lastSeenTimeStamp);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetSupportedVgpus))(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuTypeId_t *vgpuTypeIds);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetCreatableVgpus))(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuTypeId_t *vgpuTypeIds);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuTypeGetClass))(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeClass, unsigned int *size);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuTypeGetName))(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeName, unsigned int *size);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuTypeGetDeviceID))(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *deviceID, unsigned long long *subsystemID);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuTypeGetFramebufferSize))(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *fbSize);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuTypeGetNumDisplayHeads))(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *numDisplayHeads);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuTypeGetResolution))(nvmlVgpuTypeId_t vgpuTypeId, unsigned int displayIndex, unsigned int *xdim, unsigned int *ydim);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuTypeGetLicense))(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeLicenseString, unsigned int size);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuTypeGetFrameRateLimit))(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *frameRateLimit);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuTypeGetMaxInstances))(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCount);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuTypeGetMaxInstancesPerVm))(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCountPerVm);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetActiveVgpus))(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuInstance_t *vgpuInstances);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuInstanceGetVmID))(nvmlVgpuInstance_t vgpuInstance, char *vmId, unsigned int size, nvmlVgpuVmIdType_t *vmIdType);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuInstanceGetUUID))(nvmlVgpuInstance_t vgpuInstance, char *uuid, unsigned int size);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuInstanceGetVmDriverVersion))(nvmlVgpuInstance_t vgpuInstance, char* version, unsigned int length);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuInstanceGetFbUsage))(nvmlVgpuInstance_t vgpuInstance, unsigned long long *fbUsage);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuInstanceGetLicenseStatus))(nvmlVgpuInstance_t vgpuInstance, unsigned int *licensed);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuInstanceGetType))(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuTypeId_t *vgpuTypeId);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuInstanceGetFrameRateLimit))(nvmlVgpuInstance_t vgpuInstance, unsigned int *frameRateLimit);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuInstanceGetEccMode))(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t *eccMode);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuInstanceGetEncoderCapacity))(nvmlVgpuInstance_t vgpuInstance, unsigned int *encoderCapacity);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuInstanceSetEncoderCapacity))(nvmlVgpuInstance_t vgpuInstance, unsigned int  encoderCapacity);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuInstanceGetEncoderStats))(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, unsigned int *averageFps, unsigned int *averageLatency);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuInstanceGetEncoderSessions))(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfo);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuInstanceGetFBCStats))(nvmlVgpuInstance_t vgpuInstance, nvmlFBCStats_t *fbcStats);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuInstanceGetFBCSessions))(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, nvmlFBCSessionInfo_t *sessionInfo);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuInstanceGetMetadata))(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuMetadata_t *vgpuMetadata, unsigned int *bufferSize);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetVgpuMetadata))(nvmlDevice_t device, nvmlVgpuPgpuMetadata_t *pgpuMetadata, unsigned int *bufferSize);
//typedef nvmlReturn_t (*FUNC_T(nvmlGetVgpuCompatibility))(nvmlVgpuMetadata_t *vgpuMetadata, nvmlVgpuPgpuMetadata_t *pgpuMetadata, nvmlVgpuPgpuCompatibility_t *compatibilityInfo);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetPgpuMetadataString))(nvmlDevice_t device, char *pgpuMetadata, unsigned int *bufferSize);
//typedef nvmlReturn_t (*FUNC_T(nvmlGetVgpuVersion))(nvmlVgpuVersion_t *supported, nvmlVgpuVersion_t *current);
//typedef nvmlReturn_t (*FUNC_T(nvmlSetVgpuVersion))(nvmlVgpuVersion_t *vgpuVersion);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetVgpuUtilization))(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, nvmlValueType_t *sampleValType, unsigned int *vgpuInstanceSamplesCount, nvmlVgpuInstanceUtilizationSample_t *utilizationSamples);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetVgpuProcessUtilization))(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, unsigned int *vgpuProcessSamplesCount, nvmlVgpuProcessUtilizationSample_t *utilizationSamples);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuInstanceGetAccountingMode))(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t *mode);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuInstanceGetAccountingPids))(nvmlVgpuInstance_t vgpuInstance, unsigned int *count, unsigned int *pids);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuInstanceGetAccountingStats))(nvmlVgpuInstance_t vgpuInstance, unsigned int pid, nvmlAccountingStats_t *stats);
//typedef nvmlReturn_t (*FUNC_T(nvmlVgpuInstanceClearAccountingPids))(nvmlVgpuInstance_t vgpuInstance);
//typedef nvmlReturn_t (*FUNC_T(nvmlGetBlacklistDeviceCount))(unsigned int *deviceCount);
//typedef nvmlReturn_t (*FUNC_T(nvmlGetBlacklistDeviceInfoByIndex))(unsigned int index, nvmlBlacklistDeviceInfo_t *info);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceSetMigMode))(nvmlDevice_t device, unsigned int mode, nvmlReturn_t *activationStatus);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetMigMode))(nvmlDevice_t device, unsigned int *currentMode, unsigned int *pendingMode);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetGpuInstanceProfileInfo))(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_t *info);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetGpuInstancePossiblePlacements))(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t *placements, unsigned int *count);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetGpuInstanceRemainingCapacity))(nvmlDevice_t device, unsigned int profileId, unsigned int *count);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceCreateGpuInstance))(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t *gpuInstance);
//typedef nvmlReturn_t (*FUNC_T(nvmlGpuInstanceDestroy))(nvmlGpuInstance_t gpuInstance);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetGpuInstances))(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t *gpuInstances, unsigned int *count);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetGpuInstanceById))(nvmlDevice_t device, unsigned int id, nvmlGpuInstance_t *gpuInstance);
//typedef nvmlReturn_t (*FUNC_T(nvmlGpuInstanceGetInfo))(nvmlGpuInstance_t gpuInstance, nvmlGpuInstanceInfo_t *info);
//typedef nvmlReturn_t (*FUNC_T(nvmlGpuInstanceGetComputeInstanceProfileInfo))(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_t *info);
//typedef nvmlReturn_t (*FUNC_T(nvmlGpuInstanceGetComputeInstanceRemainingCapacity))(nvmlGpuInstance_t gpuInstance, unsigned int profileId, unsigned int *count);
//typedef nvmlReturn_t (*FUNC_T(nvmlGpuInstanceCreateComputeInstance))(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t *computeInstance);
//typedef nvmlReturn_t (*FUNC_T(nvmlComputeInstanceDestroy))(nvmlComputeInstance_t computeInstance);
//typedef nvmlReturn_t (*FUNC_T(nvmlGpuInstanceGetComputeInstances))(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t *computeInstances, unsigned int *count);
//typedef nvmlReturn_t (*FUNC_T(nvmlGpuInstanceGetComputeInstanceById))(nvmlGpuInstance_t gpuInstance, unsigned int id, nvmlComputeInstance_t *computeInstance);
//typedef nvmlReturn_t (*FUNC_T(nvmlComputeInstanceGetInfo))(nvmlComputeInstance_t computeInstance, nvmlComputeInstanceInfo_t *info);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceIsMigDeviceHandle))(nvmlDevice_t device, unsigned int *isMigDevice);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetGpuInstanceId))(nvmlDevice_t device, unsigned int *id);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetComputeInstanceId))(nvmlDevice_t device, unsigned int *id);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetMaxMigDeviceCount))(nvmlDevice_t device, unsigned int *count);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetMigDeviceHandleByIndex))(nvmlDevice_t device, unsigned int index, nvmlDevice_t *migDevice);
//typedef nvmlReturn_t (*FUNC_T(nvmlDeviceGetDeviceHandleFromMigDeviceHandle))(nvmlDevice_t migDevice, nvmlDevice_t *device);



struct NVMLFunctionTable
{
  FUNC_P(nvmlInit);
  //FUNC_P(nvmlInitWithFlags);
  FUNC_P(nvmlShutdown);
  //FUNC_P(nvmlErrorString);
  //FUNC_P(nvmlSystemGetDriverVersion);
  //FUNC_P(nvmlSystemGetNVMLVersion);
  //FUNC_P(nvmlSystemGetCudaDriverVersion);
  //FUNC_P(nvmlSystemGetCudaDriverVersion_v2);
  //FUNC_P(nvmlSystemGetProcessName);
  //FUNC_P(nvmlUnitGetCount);
  //FUNC_P(nvmlUnitGetHandleByIndex);
  //FUNC_P(nvmlUnitGetUnitInfo);
  //FUNC_P(nvmlUnitGetLedState);
  //FUNC_P(nvmlUnitGetPsuInfo);
  //FUNC_P(nvmlUnitGetTemperature);
  //FUNC_P(nvmlUnitGetFanSpeedInfo);
  //FUNC_P(nvmlUnitGetDevices);
  //FUNC_P(nvmlSystemGetHicVersion);
  //FUNC_P(nvmlDeviceGetCount);
  //FUNC_P(nvmlDeviceGetAttributes);
  //FUNC_P(nvmlDeviceGetHandleByIndex);
  //FUNC_P(nvmlDeviceGetHandleBySerial);
  //FUNC_P(nvmlDeviceGetHandleByUUID);
  FUNC_P(nvmlDeviceGetHandleByPciBusId);
  //FUNC_P(nvmlDeviceGetName);
  //FUNC_P(nvmlDeviceGetBrand);
  //FUNC_P(nvmlDeviceGetIndex);
  //FUNC_P(nvmlDeviceGetSerial);
  //FUNC_P(nvmlDeviceGetMemoryAffinity);
  //FUNC_P(nvmlDeviceGetCpuAffinityWithinScope);
  //FUNC_P(nvmlDeviceGetCpuAffinity);
  //FUNC_P(nvmlDeviceSetCpuAffinity);
  //FUNC_P(nvmlDeviceClearCpuAffinity);
  //FUNC_P(nvmlDeviceGetTopologyCommonAncestor);
  //FUNC_P(nvmlDeviceGetTopologyNearestGpus);
  //FUNC_P(nvmlSystemGetTopologyGpuSet);
  //FUNC_P(nvmlDeviceGetP2PStatus);
  //FUNC_P(nvmlDeviceGetUUID);
  //FUNC_P(nvmlVgpuInstanceGetMdevUUID);
  //FUNC_P(nvmlDeviceGetMinorNumber);
  //FUNC_P(nvmlDeviceGetBoardPartNumber);
  //FUNC_P(nvmlDeviceGetInforomVersion);
  //FUNC_P(nvmlDeviceGetInforomImageVersion);
  //FUNC_P(nvmlDeviceGetInforomConfigurationChecksum);
  //FUNC_P(nvmlDeviceValidateInforom);
  //FUNC_P(nvmlDeviceGetDisplayMode);
  //FUNC_P(nvmlDeviceGetDisplayActive);
  //FUNC_P(nvmlDeviceGetPersistenceMode);
  //FUNC_P(nvmlDeviceGetPciInfo);
  //FUNC_P(nvmlDeviceGetMaxPcieLinkGeneration);
  //FUNC_P(nvmlDeviceGetMaxPcieLinkWidth);
  //FUNC_P(nvmlDeviceGetCurrPcieLinkGeneration);
  //FUNC_P(nvmlDeviceGetCurrPcieLinkWidth);
  //FUNC_P(nvmlDeviceGetPcieThroughput);
  //FUNC_P(nvmlDeviceGetPcieReplayCounter);
  //FUNC_P(nvmlDeviceGetClockInfo);
  //FUNC_P(nvmlDeviceGetMaxClockInfo);
  //FUNC_P(nvmlDeviceGetApplicationsClock);
  //FUNC_P(nvmlDeviceGetDefaultApplicationsClock);
  //FUNC_P(nvmlDeviceResetApplicationsClocks);
  //FUNC_P(nvmlDeviceGetClock);
  //FUNC_P(nvmlDeviceGetMaxCustomerBoostClock);
  //FUNC_P(nvmlDeviceGetSupportedMemoryClocks);
  //FUNC_P(nvmlDeviceGetSupportedGraphicsClocks);
  //FUNC_P(nvmlDeviceGetAutoBoostedClocksEnabled);
  //FUNC_P(nvmlDeviceSetAutoBoostedClocksEnabled);
  //FUNC_P(nvmlDeviceSetDefaultAutoBoostedClocksEnabled);
  //FUNC_P(nvmlDeviceGetFanSpeed);
  //FUNC_P(nvmlDeviceGetFanSpeed_v2);
  //FUNC_P(nvmlDeviceGetTemperature);
  //FUNC_P(nvmlDeviceGetTemperatureThreshold);
  //FUNC_P(nvmlDeviceGetPerformanceState);
  //FUNC_P(nvmlDeviceGetCurrentClocksThrottleReasons);
  //FUNC_P(nvmlDeviceGetSupportedClocksThrottleReasons);
  //FUNC_P(nvmlDeviceGetPowerState);
  //FUNC_P(nvmlDeviceGetPowerManagementMode);
  //FUNC_P(nvmlDeviceGetPowerManagementLimit);
  //FUNC_P(nvmlDeviceGetPowerManagementLimitConstraints);
  //FUNC_P(nvmlDeviceGetPowerManagementDefaultLimit);
  //FUNC_P(nvmlDeviceGetPowerUsage);
  //FUNC_P(nvmlDeviceGetTotalEnergyConsumption);
  //FUNC_P(nvmlDeviceGetEnforcedPowerLimit);
  //FUNC_P(nvmlDeviceGetGpuOperationMode);
  //FUNC_P(nvmlDeviceGetMemoryInfo);
  //FUNC_P(nvmlDeviceGetComputeMode);
  //FUNC_P(nvmlDeviceGetCudaComputeCapability);
  //FUNC_P(nvmlDeviceGetEccMode);
  //FUNC_P(nvmlDeviceGetBoardId);
  //FUNC_P(nvmlDeviceGetMultiGpuBoard);
  //FUNC_P(nvmlDeviceGetTotalEccErrors);
  //FUNC_P(nvmlDeviceGetDetailedEccErrors);
  //FUNC_P(nvmlDeviceGetMemoryErrorCounter);
  //FUNC_P(nvmlDeviceGetUtilizationRates);
  //FUNC_P(nvmlDeviceGetEncoderUtilization);
  //FUNC_P(nvmlDeviceGetEncoderCapacity);
  //FUNC_P(nvmlDeviceGetEncoderStats);
  //FUNC_P(nvmlDeviceGetEncoderSessions);
  //FUNC_P(nvmlDeviceGetDecoderUtilization);
  //FUNC_P(nvmlDeviceGetFBCStats);
  //FUNC_P(nvmlDeviceGetFBCSessions);
  //FUNC_P(nvmlDeviceGetDriverModel);
  //FUNC_P(nvmlDeviceGetVbiosVersion);
  //FUNC_P(nvmlDeviceGetBridgeChipInfo);
  //FUNC_P(nvmlDeviceGetComputeRunningProcesses);
  //FUNC_P(nvmlDeviceGetGraphicsRunningProcesses);
  //FUNC_P(nvmlDeviceOnSameBoard);
  //FUNC_P(nvmlDeviceGetAPIRestriction);
  //FUNC_P(nvmlDeviceGetSamples);
  //FUNC_P(nvmlDeviceGetBAR1MemoryInfo);
  //FUNC_P(nvmlDeviceGetViolationStatus);
  //FUNC_P(nvmlDeviceGetAccountingMode);
  //FUNC_P(nvmlDeviceGetAccountingStats);
  //FUNC_P(nvmlDeviceGetAccountingPids);
  //FUNC_P(nvmlDeviceGetAccountingBufferSize);
  //FUNC_P(nvmlDeviceGetRetiredPages);
  //FUNC_P(nvmlDeviceGetRetiredPages_v2);
  //FUNC_P(nvmlDeviceGetRetiredPagesPendingStatus);
  //FUNC_P(nvmlDeviceGetRemappedRows);
  //FUNC_P(nvmlDeviceGetArchitecture);
  //FUNC_P(nvmlUnitSetLedState);
  //FUNC_P(nvmlDeviceSetPersistenceMode);
  //FUNC_P(nvmlDeviceSetComputeMode);
  //FUNC_P(nvmlDeviceSetEccMode);
  //FUNC_P(nvmlDeviceClearEccErrorCounts);
  //FUNC_P(nvmlDeviceSetDriverModel);
  //FUNC_P(nvmlDeviceSetGpuLockedClocks);
  //FUNC_P(nvmlDeviceResetGpuLockedClocks);
  //FUNC_P(nvmlDeviceSetApplicationsClocks);
  //FUNC_P(nvmlDeviceSetPowerManagementLimit);
  //FUNC_P(nvmlDeviceSetGpuOperationMode);
  //FUNC_P(nvmlDeviceSetAPIRestriction);
  //FUNC_P(nvmlDeviceSetAccountingMode);
  //FUNC_P(nvmlDeviceClearAccountingPids);
  FUNC_P(nvmlDeviceGetNvLinkState);
  //FUNC_P(nvmlDeviceGetNvLinkVersion);
  FUNC_P(nvmlDeviceGetNvLinkCapability);
  FUNC_P(nvmlDeviceGetNvLinkRemotePciInfo);
  //FUNC_P(nvmlDeviceGetNvLinkErrorCounter);
  //FUNC_P(nvmlDeviceResetNvLinkErrorCounters);
  //FUNC_P(nvmlDeviceSetNvLinkUtilizationControl);
  //FUNC_P(nvmlDeviceGetNvLinkUtilizationControl);
  //FUNC_P(nvmlDeviceGetNvLinkUtilizationCounter);
  //FUNC_P(nvmlDeviceFreezeNvLinkUtilizationCounter);
  //FUNC_P(nvmlDeviceResetNvLinkUtilizationCounter);
  //FUNC_P(nvmlEventSetCreate);
  //FUNC_P(nvmlDeviceRegisterEvents);
  //FUNC_P(nvmlDeviceGetSupportedEventTypes);
  //FUNC_P(nvmlEventSetWait);
  //FUNC_P(nvmlEventSetFree);
  //FUNC_P(nvmlDeviceModifyDrainState);
  //FUNC_P(nvmlDeviceQueryDrainState);
  //FUNC_P(nvmlDeviceRemoveGpu);
  //FUNC_P(nvmlDeviceDiscoverGpus);
  //FUNC_P(nvmlDeviceGetFieldValues);
  //FUNC_P(nvmlDeviceGetVirtualizationMode);
  //FUNC_P(nvmlDeviceGetHostVgpuMode);
  //FUNC_P(nvmlDeviceSetVirtualizationMode);
  //FUNC_P(nvmlDeviceGetGridLicensableFeatures);
  //FUNC_P(nvmlDeviceGetProcessUtilization);
  //FUNC_P(nvmlDeviceGetSupportedVgpus);
  //FUNC_P(nvmlDeviceGetCreatableVgpus);
  //FUNC_P(nvmlVgpuTypeGetClass);
  //FUNC_P(nvmlVgpuTypeGetName);
  //FUNC_P(nvmlVgpuTypeGetDeviceID);
  //FUNC_P(nvmlVgpuTypeGetFramebufferSize);
  //FUNC_P(nvmlVgpuTypeGetNumDisplayHeads);
  //FUNC_P(nvmlVgpuTypeGetResolution);
  //FUNC_P(nvmlVgpuTypeGetLicense);
  //FUNC_P(nvmlVgpuTypeGetFrameRateLimit);
  //FUNC_P(nvmlVgpuTypeGetMaxInstances);
  //FUNC_P(nvmlVgpuTypeGetMaxInstancesPerVm);
  //FUNC_P(nvmlDeviceGetActiveVgpus);
  //FUNC_P(nvmlVgpuInstanceGetVmID);
  //FUNC_P(nvmlVgpuInstanceGetUUID);
  //FUNC_P(nvmlVgpuInstanceGetVmDriverVersion);
  //FUNC_P(nvmlVgpuInstanceGetFbUsage);
  //FUNC_P(nvmlVgpuInstanceGetLicenseStatus);
  //FUNC_P(nvmlVgpuInstanceGetType);
  //FUNC_P(nvmlVgpuInstanceGetFrameRateLimit);
  //FUNC_P(nvmlVgpuInstanceGetEccMode);
  //FUNC_P(nvmlVgpuInstanceGetEncoderCapacity);
  //FUNC_P(nvmlVgpuInstanceSetEncoderCapacity);
  //FUNC_P(nvmlVgpuInstanceGetEncoderStats);
  //FUNC_P(nvmlVgpuInstanceGetEncoderSessions);
  //FUNC_P(nvmlVgpuInstanceGetFBCStats);
  //FUNC_P(nvmlVgpuInstanceGetFBCSessions);
  //FUNC_P(nvmlVgpuInstanceGetMetadata);
  //FUNC_P(nvmlDeviceGetVgpuMetadata);
  //FUNC_P(nvmlGetVgpuCompatibility);
  //FUNC_P(nvmlDeviceGetPgpuMetadataString);
  //FUNC_P(nvmlGetVgpuVersion);
  //FUNC_P(nvmlSetVgpuVersion);
  //FUNC_P(nvmlDeviceGetVgpuUtilization);
  //FUNC_P(nvmlDeviceGetVgpuProcessUtilization);
  //FUNC_P(nvmlVgpuInstanceGetAccountingMode);
  //FUNC_P(nvmlVgpuInstanceGetAccountingPids);
  //FUNC_P(nvmlVgpuInstanceGetAccountingStats);
  //FUNC_P(nvmlVgpuInstanceClearAccountingPids);
  //FUNC_P(nvmlGetBlacklistDeviceCount);
  //FUNC_P(nvmlGetBlacklistDeviceInfoByIndex);
  //FUNC_P(nvmlDeviceSetMigMode);
  //FUNC_P(nvmlDeviceGetMigMode);
  //FUNC_P(nvmlDeviceGetGpuInstanceProfileInfo);
  //FUNC_P(nvmlDeviceGetGpuInstancePossiblePlacements);
  //FUNC_P(nvmlDeviceGetGpuInstanceRemainingCapacity);
  //FUNC_P(nvmlDeviceCreateGpuInstance);
  //FUNC_P(nvmlGpuInstanceDestroy);
  //FUNC_P(nvmlDeviceGetGpuInstances);
  //FUNC_P(nvmlDeviceGetGpuInstanceById);
  //FUNC_P(nvmlGpuInstanceGetInfo);
  //FUNC_P(nvmlGpuInstanceGetComputeInstanceProfileInfo);
  //FUNC_P(nvmlGpuInstanceGetComputeInstanceRemainingCapacity);
  //FUNC_P(nvmlGpuInstanceCreateComputeInstance);
  //FUNC_P(nvmlComputeInstanceDestroy);
  //FUNC_P(nvmlGpuInstanceGetComputeInstances);
  //FUNC_P(nvmlGpuInstanceGetComputeInstanceById);
  //FUNC_P(nvmlComputeInstanceGetInfo);
  //FUNC_P(nvmlDeviceIsMigDeviceHandle);
  //FUNC_P(nvmlDeviceGetGpuInstanceId);
  //FUNC_P(nvmlDeviceGetComputeInstanceId);
  //FUNC_P(nvmlDeviceGetMaxMigDeviceCount);
  //FUNC_P(nvmlDeviceGetMigDeviceHandleByIndex);
  //FUNC_P(nvmlDeviceGetDeviceHandleFromMigDeviceHandle);
};

#undef FUNC_T_V
#undef FUNC_T

#undef FUNC_P_V
#undef FUNC_P

class NVMLImpl
{
public:
  NVMLImpl();
  //~NVMLImpl();

  bool initFunctionTable();

public:
  NVMLFunctionTable m_api;

private:
  void* m_handle; // nullptr when the library could not be found 
};


#endif // NVML_IMPL_H

