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

#include "inc/Raytracer.h"

#include "inc/CheckMacros.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

bool static saveString(const std::string& filename, const std::string& text)
{
  std::ofstream outputStream(filename);

  if (!outputStream)
  {
    std::cerr << "ERROR: saveString() Failed to open file " << filename << '\n';
    return false;
  }

  outputStream << text;

  if (outputStream.fail())
  {
    std::cerr << "ERROR: saveString() Failed to write file " << filename << '\n';
    return false;
  }

  return true;
}

static std::string getDateTime()
{
#if defined(_WIN32)
  SYSTEMTIME time;
  GetLocalTime(&time);
#elif defined(__linux__)
  time_t rawtime;
  struct tm* ts;
  time(&rawtime);
  ts = localtime(&rawtime);
#else
  #error "OS not supported."
#endif

  std::ostringstream oss;

#if defined( _WIN32 )
  oss << time.wYear;
  if (time.wMonth < 10)
  {
    oss << '0';
  }
  oss << time.wMonth;
  if (time.wDay < 10)
  {
    oss << '0';
  }
  oss << time.wDay << '_';
  if (time.wHour < 10)
  {
    oss << '0';
  }
  oss << time.wHour;
  if (time.wMinute < 10)
  {
    oss << '0';
  }
  oss << time.wMinute;
  if (time.wSecond < 10)
  {
    oss << '0';
  }
  oss << time.wSecond << '_';
  if (time.wMilliseconds < 100)
  {
    oss << '0';
  }
  if (time.wMilliseconds <  10)
  {
    oss << '0';
  }
  oss << time.wMilliseconds; 
#elif defined(__linux__)
  oss << ts->tm_year;
  if (ts->tm_mon < 10)
  {
    oss << '0';
  }
  oss << ts->tm_mon;
  if (ts->tm_mday < 10)
  {
    oss << '0';
  }
  oss << ts->tm_mday << '_';
  if (ts->tm_hour < 10)
  {
    oss << '0';
  }
  oss << ts->tm_hour;
  if (ts->tm_min < 10)
  {
    oss << '0';
  }
  oss << ts->tm_min;
  if (ts->tm_sec < 10)
  {
    oss << '0';
  }
  oss << ts->tm_sec << '_';
  oss << "000"; // No milliseconds available.
#else
  #error "OS not supported."
#endif

  return oss.str();
}


Raytracer::Raytracer(const int maskDevices,
                     const TypeLight typeEnv,
                     const int interop,
                     const unsigned int tex,
                     const unsigned int pbo,
                     const size_t sizeArena)
: m_maskDevices(maskDevices)
, m_typeEnv(typeEnv)
, m_interop(interop)
, m_tex(tex)
, m_pbo(pbo)
, m_sizeArena(sizeArena)
, m_isValid(false)
, m_numDevicesVisible(0)
, m_indexDeviceOGL(-1)
, m_maskDevicesActive(0)
, m_iterationIndex(0)
, m_samplesPerPixel(1)
{
  CU_CHECK( cuInit(0) ); // Initialize CUDA driver API.

  int versionDriver = 0;
  CU_CHECK( cuDriverGetVersion(&versionDriver) ); 
  
  // The version is returned as (1000 * major + 10 * minor).
  int major =  versionDriver / 1000;
  int minor = (versionDriver - 1000 * major) / 10;
  std::cout << "CUDA Driver Version = " << major << "." << minor << '\n';
  
  CU_CHECK( cuDeviceGetCount(&m_numDevicesVisible) );
  std::cout << "CUDA Device Count   = " << m_numDevicesVisible << '\n';

  // Match user defined m_maskDevices with the number of visible devices.
  // Builds m_maskActiveDevices and fills m_devicesActive which defines the device count.
  selectDevices();

  m_isValid = !m_devicesActive.empty();
}


Raytracer::~Raytracer()
{
  try
  {
    // The GeometryData is either created on each device or only on one device of an NVLINK island.
    // In any case GeometryData is unique and must only be destroyed by the device owning the data.
    for (auto& data : m_geometryData)
    {
      m_devicesActive[data.owner]->destroyGeometry(data);
    }

    for (size_t i = 0; i < m_devicesActive.size(); ++i)
    {
      delete m_devicesActive[i];
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << '\n';
  }

}

int Raytracer::matchUUID(const char* uuid)
{
  const int size = static_cast<int>(m_devicesActive.size());

  for (int i = 0; i < size; ++i)
  {
    if (m_devicesActive[i]->matchUUID(uuid))
    {
      // Use the first device which matches with the OpenGL UUID.
      // DEBUG This might not be the right thing to do with multicast enabled.
      m_indexDeviceOGL = i;
      break;
    }
  }

  std::cout << "OpenGL on active device index " << m_indexDeviceOGL << '\n'; // DEBUG 

  return m_indexDeviceOGL; // If this stays -1, the active devices do not contain the one running the OpenGL implementation.
}

int Raytracer::matchLUID(const char* luid, const unsigned int nodeMask)
{
  const int size = static_cast<int>(m_devicesActive.size());

  for (int i = 0; i < size; ++i)
  {
    if (m_devicesActive[i]->matchLUID(luid, nodeMask))
    {
      // Use the first device which matches with the OpenGL LUID and test of the node mask bit.
      // DEBUG This might not be the right thing to do with multicast enabled.
      m_indexDeviceOGL = i;
      break;
    }
  }
  
  std::cout << "OpenGL on active device index " << m_indexDeviceOGL << '\n'; // DEBUG 

  return m_indexDeviceOGL; // If this stays -1, the active devices do not contain the one running the OpenGL implementation.
}

void Raytracer::synchronize()
{
  for (size_t i = 0; i < m_devicesActive.size(); ++i)
  {
    m_devicesActive[i]->activateContext();
    m_devicesActive[i]->synchronizeStream();
  }
}

// FIXME This cannot handle cases where the same Picture would be used for different texture objects, but that is not happening in this example.
void Raytracer::initTextures(const std::map<std::string, Picture*>& mapPictures)
{ 
  for (std::map<std::string, Picture*>::const_iterator it = mapPictures.begin(); it != mapPictures.end(); ++it)
  {
    const Picture* picture = it->second;

    const bool isEnv = ((picture->getFlags() & IMAGE_FLAG_ENV) != 0);

    const unsigned int numDevices = static_cast<unsigned int>(m_devicesActive.size());

    for (unsigned int device = 0; device < numDevices; ++device)
    {
      printf("init texture %i of %i \n", device, numDevices);
      (void) m_devicesActive[device]->initTexture(it->first, picture, picture->getFlags());
    }
  }

  printf("finished init textures\n");
}


void Raytracer::initCameras(const std::vector<CameraDefinition>& cameras)
{
  for (size_t i = 0; i < m_devicesActive.size(); ++i)
  {
    m_devicesActive[i]->initCameras(cameras);
  }
}

// For mesh lights this needs to be aware of the GAS sharing which results in different sizes of the m_geometryData built in initScene()!
void Raytracer::initLights(const std::vector<LightGUI>& lightsGUI)
{
    const unsigned int numDevices = static_cast<unsigned int>(m_devicesActive.size());

    for (unsigned int device = 0; device < numDevices; ++device)
    {
        m_devicesActive[device]->initLights(lightsGUI, m_geometryData, numDevices, device);
    }
}

// Traverse the SceneGraph and store Groups, Instances and Triangles nodes in the raytracer representation.
void Raytracer::initScene(std::shared_ptr<sg::Group> root, const unsigned int numGeometries)
{
    // Allocate the number of GeometryData per active device.
    m_geometryData.resize(numGeometries * m_devicesActive.size()); // Not sharing, all devices hold all geometry data.

    InstanceData instanceData(~0u, -1, -1, -1);

    float matrix[12];

    // Set the affine matrix to identity by default.
    memset(matrix, 0, sizeof(float) * 12);
    matrix[ 0] = 1.0f;
    matrix[ 5] = 1.0f;
    matrix[10] = 1.0f;

    traverseNode(root, instanceData, matrix);

    const unsigned int numDevices = static_cast<unsigned int>(m_devicesActive.size());

    for (unsigned int device = 0; device < numDevices; ++device)
    {
      m_devicesActive[device]->createTLAS();
      m_devicesActive[device]->createGeometryInstanceData(m_geometryData, numDevices, device);
    }
}


void Raytracer::initState(const DeviceState& state)
{
    m_samplesPerPixel = state.spp;

    for (size_t i = 0; i < m_devicesActive.size(); ++i)
    {
        m_devicesActive[i]->setState(state);
    }
}

void Raytracer::updateCamera(const int idCamera, const CameraDefinition& camera)
{
  for (size_t i = 0; i < m_devicesActive.size(); ++i)
  {
    m_devicesActive[i]->updateCamera(idCamera, camera);
  }
  m_iterationIndex = 0; // Restart accumulation.
}

void Raytracer::updateLight(const int idLight, const LightGUI& lightGUI)
{
  for (size_t i = 0; i < m_devicesActive.size(); ++i)
  {
    m_devicesActive[i]->updateLight(idLight, lightGUI);
  }
  m_iterationIndex = 0; // Restart accumulation.
}

//void Raytracer::updateLight(const int idLight, const LightDefinition& light)
//{
//  for (size_t i = 0; i < m_devicesActive.size(); ++i)
//  {
//    m_devicesActive[i]->updateLight(idLight, light);
//  }
//  m_iterationIndex = 0; // Restart accumulation.
//}

void Raytracer::updateMaterial(const int idMaterial, const MaterialMDL* materialMDL)
{
    for (size_t i = 0; i < m_devicesActive.size(); ++i)
    {
        m_devicesActive[i]->updateMaterial(idMaterial, materialMDL);
    }
    m_iterationIndex = 0; // Restart accumulation.
}

void Raytracer::updateState(const DeviceState& state)
{
    m_samplesPerPixel = state.spp;

    for (size_t i = 0; i < m_devicesActive.size(); ++i)
    {
        m_devicesActive[i]->setState(state);
    }
    m_iterationIndex = 0; // Restart accumulation.
}

void Raytracer::updateRenderingOptions(int32_t num_panes, const PaneFlags& pane_a, const PaneFlags& pane_b, const PaneFlags& pane_c) {
    for (size_t i = 0; i < m_devicesActive.size(); ++i) {
        m_devicesActive[i]->updateRenderingOptions(num_panes, pane_a, pane_b, pane_c);
    }
    m_iterationIndex = 0;
}

// The public function which does the multi-GPU wrapping.
// Returns the count of renderered iterations (m_iterationIndex after it has been incremented).
unsigned int Raytracer::render(const int mode, bool ref)
{
    // HACK!
    int32_t one = ref ? 0 : 1;
    // Continue manual accumulation rendering if the samples per pixel have not been reached.
    if (m_iterationIndex < m_samplesPerPixel + one) // non-reference computation needs this for spatial reuse
    {
        void* buffer = nullptr;

        // Make sure the OpenGL device is allocating the full resolution backing storage.
        const int index = (m_indexDeviceOGL != -1) ? m_indexDeviceOGL : 0; // Destination device.

        // This is the device which needs to allocate the peer-to-peer buffer to reside on the same device as the PBO or Texture
        m_devicesActive[index]->render(m_iterationIndex, &buffer, mode); // Interactive rendering. All devices work on the same iteration index.

        ++m_iterationIndex;
    }
    //std::cout << "ref " << ref << " iteration_index: " << m_iterationIndex << std::endl;
    return m_iterationIndex;
}

void Raytracer::updateDisplayTexture()
{
    const int index = (m_indexDeviceOGL != -1) ? m_indexDeviceOGL : 0; // Destination device.

    // Only need to composite the resulting frame when using multiple decvices.
    // Single device renders directly into the full resolution output buffer.
    if (1 < m_devicesActive.size())
    {
        // First, copy the texelBuffer of the primary device into its tileBuffer and then place the tiles into the outputBuffer.
        m_devicesActive[index]->compositor(m_devicesActive[index]);

        // Now copy the other devices' texelBuffers over to the main tileBuffer and repeat the compositing for that other device.
        // The cuMemcpyPeerAsync done in that case is fast when the devices are in the same peer island, otherwise it's copied via PCI-E, but only N-1 copies of 1/N size are done.
        // The saving here is no peer-to-peer read-modify-write when rendering, because everything is happening in GPU local buffers, which are also tightly packed.
        // The final compositing is just a kernel implementing a tiled memcpy.
        // PERF If all tiles are copied to the main device at once, such kernel would only need to be called once.
        for (size_t i = 0; i < m_devicesActive.size(); ++i)
        {
            if (index != static_cast<int>(i))
            {
                m_devicesActive[index]->compositor(m_devicesActive[i]);
            }
        }
    }
    // Finally copy the primary device outputBuffer to the display texture.
    // FIXME DEBUG Does that work when m_indexDeviceOGL is not in the list of active devices?
    m_devicesActive[index]->updateDisplayTexture();
}

const void* Raytracer::getOutputBufferHost()
{
    // Same initial steps to fill the outputBuffer on the primary device as in updateDisplayTexture()
    const int index = (m_indexDeviceOGL != -1) ? m_indexDeviceOGL : 0; // Destination device.

    // Only need to composite the resulting frame when using multiple decvices.
    // Single device renders  directly into the full resolution output buffer.
    if (1 < m_devicesActive.size())
    {
        // First, copy the texelBuffer of the primary device into its tileBuffer and then place the tiles into the outputBuffer.
        m_devicesActive[index]->compositor(m_devicesActive[index]);

        // Now copy the other devices' texelBuffers over to the main tileBuffer and repeat the compositing for that other device.
        for (size_t i = 0; i < m_devicesActive.size(); ++i)
        {
            if (index != static_cast<int>(i))
            {
                m_devicesActive[index]->compositor(m_devicesActive[i]);
            }
        }
    }
    // The full outputBuffer resides on device "index" and the host buffer is also only resized by that device.
    return m_devicesActive[index]->getOutputBufferHost();
}

// Private functions.

void Raytracer::selectDevices()
{
  // Need to determine the number of active devices first to have it available as device constructor argument.
  int count   = 0; 
  int ordinal = 0;

  while (ordinal < m_numDevicesVisible) // Don't try to enable more devices than visible to CUDA.
  {
    const unsigned int mask = (1 << ordinal);

    if (m_maskDevices & mask)
    {
      // Track which and how many devices have actually been enabled.
      m_maskDevicesActive |= mask; 
      ++count;
    }

    ++ordinal;
  }

  // Now really construct the Device objects.
  ordinal = 0;

  while (ordinal < m_numDevicesVisible)
  {
    const unsigned int mask = (1 << ordinal);

    if (m_maskDevicesActive & mask)
    {
      const int index = static_cast<int>(m_devicesActive.size());

      Device* device = new Device(ordinal, index, count, m_typeEnv, m_interop, m_tex, m_pbo, m_sizeArena);

      m_devicesActive.push_back(device);

      std::cout << "Device ordinal " << ordinal << ": " << device->m_deviceName << " selected as active device index " << index << '\n';
    }

    ++ordinal;
  }
}

#if 1
// This implementation does not consider the actually free amount of VRAM on the individual devices in an island, but assumes they are equally loaded.
// This method works more fine grained with the arena allocator.
int Raytracer::getDeviceHome(const std::vector<int>& island) const
{
  // Find the device inside each island which has the least amount of allocated memory.
  size_t sizeMin    = ~0ull; // Biggest unsigned 64-bit number.
  int    deviceHome = 0;     // Default to zero if all devices are OOM. That will fail in CU_CHECK later.

  for (auto device : island)
  {
    const size_t size = m_devicesActive[device]->getMemoryAllocated();

    if (size < sizeMin)
    {
      sizeMin    = size;
      deviceHome = device;
    }
  }

  //std::cout << "deviceHome = " << deviceHome << ", allocated [MiB] = " << double(sizeMin) / (1024.0 * 1024.0) << '\n'; // DEBUG 

  return deviceHome;
}

#else

// This implementation uses the actual free amount of VRAM on the individual devices in an NVLINK island.
// With the arena allocator this will result in less fine grained distribution of resources because the free memory only changes when a new arena is allocated.
// Using a smaller arena size would switch allocations between devices more often in this case.
int Raytracer::getDeviceHome(const std::vector<int>& island) const
{
  // Find the device inside each island which has the most free memory.
  size_t sizeMax    = 0;
  int    deviceHome = 0; // Default to zero if all devices are OOM. That will fail in CU_CHECK later.

  for (auto device : island)
  {
    const size_t size = m_devicesActive[device]->getMemoryFree(); // Actual free VRAM overall.

    if (sizeMax < size)
    {
      sizeMax    = size;
      deviceHome = device;
    }
  }
  
  //std::cout << "deviceHome = " << deviceHome << ", free [MiB] = " << double(sizeMax) / (1024.0 * 1024.0) << '\n'; // DEBUG 

  return deviceHome;
}
#endif

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


// Depth-first traversal of the scene graph to flatten all unique paths to a geometry node to one-level instancing inside the OptiX render graph.
void Raytracer::traverseNode(std::shared_ptr<sg::Node> node, InstanceData instanceData, float matrix[12])
{
  switch (node->getType())
  {
    case sg::NodeType::NT_GROUP:
    {
      std::shared_ptr<sg::Group> group = std::dynamic_pointer_cast<sg::Group>(node);

      for (size_t i = 0; i < group->getNumChildren(); ++i)
      {
        traverseNode(group->getChild(i), instanceData, matrix);
      }
    }
    break;

    case sg::NodeType::NT_INSTANCE:
    {
      std::shared_ptr<sg::Instance> instance = std::dynamic_pointer_cast<sg::Instance>(node);

      // Track the assigned material and light indices. Only the bottom-most instance node matters.
      instanceData.idMaterial = instance->getMaterial();
      instanceData.idLight    = instance->getLight();
      instanceData.idObject   = instance->getId();

      // Concatenate the transformations along the path.
      float trafo[12];

      multiplyMatrix(trafo, matrix, instance->getTransform());

      traverseNode(instance->getChild(), instanceData, trafo);
    }
    break;

    case sg::NodeType::NT_TRIANGLES:
    {
      std::shared_ptr<sg::Triangles> geometry = std::dynamic_pointer_cast<sg::Triangles>(node);
      
      instanceData.idGeometry = geometry->getId();

      {
        const unsigned int numDevices = static_cast<unsigned int>(m_devicesActive.size());

        for (unsigned int device = 0; device < numDevices; ++device)
        {
          GeometryData& geometryData = m_geometryData[instanceData.idGeometry * numDevices + device];

          if (geometryData.traversable == 0) // If there is no traversable handle for this geometry on this device, try to create one.
          {
            geometryData = m_devicesActive[device]->createGeometry(geometry);
          }

          m_devicesActive[device]->createInstance(geometryData, instanceData, matrix);
        }
      }
    }
    break;

    case sg::NodeType::NT_CURVES:
    {
      std::shared_ptr<sg::Curves> geometry = std::dynamic_pointer_cast<sg::Curves>(node);
      
      instanceData.idGeometry = geometry->getId();

      {
        const unsigned int numDevices = static_cast<unsigned int>(m_devicesActive.size());

        for (unsigned int device = 0; device < numDevices; ++device)
        {
          GeometryData& geometryData = m_geometryData[instanceData.idGeometry * numDevices + device];

          if (geometryData.traversable == 0) // If there is no traversable handle for this geometry on this device, try to create one.
          {
            geometryData = m_devicesActive[device]->createGeometry(geometry);
          }

          m_devicesActive[device]->createInstance(geometryData, instanceData, matrix);
        }
      }
    }
    break;

  }
}
