/*
 * Copyright (c) 2013-2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MATERIAL_MDL_H
#define MATERIAL_MDL_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <mi/mdl_sdk.h>
#include <mi/base/config.h>

#include <list>
#include <map>
#include <string>
#include <vector>

#include "shaders/texture_handler.h"


 // Possible enum values if any.
struct Enum_value
{
  std::string name;
  int         value;

  Enum_value(const std::string& name, int value)
    : name(name)
    , value(value)
  {
  }
};


// Info for an enum type.
struct Enum_type_info
{
  std::vector<Enum_value> values;

  // Adds a enum value and its integer value to the enum type info.
  void add(const std::string& name, int value)
  {
    values.push_back(Enum_value(name, value));
  }
};


// Material parameter information structure.
class Param_info
{
public:
  enum Param_kind
  {
    PK_UNKNOWN,
    PK_FLOAT,
    PK_FLOAT2,
    PK_FLOAT3,
    PK_COLOR,
    PK_ARRAY,
    PK_BOOL,
    PK_INT,
    PK_ENUM,
    PK_STRING,
    PK_TEXTURE,
    PK_LIGHT_PROFILE,
    PK_BSDF_MEASUREMENT
  };

  Param_info(size_t index,
             char const* name,
             char const* display_name,
             char const* group_name,
             Param_kind kind,
             Param_kind array_elem_kind,
             mi::Size   array_size,
             mi::Size   array_pitch,
             char* data_ptr,
             const Enum_type_info* enum_info = nullptr)
    : m_index(index)
    , m_name(name)
    , m_display_name(display_name)
    , m_group_name(group_name)
    , m_kind(kind)
    , m_array_elem_kind(array_elem_kind)
    , m_array_size(array_size)
    , m_array_pitch(array_pitch)
    , m_data_ptr(data_ptr)
    , m_range_min(0.0f)
    , m_range_max(1.0f)
    , m_enum_info(enum_info)
  {
  }

  // Get data as T&.
  template<typename T>
  T& data()
  {
    return *reinterpret_cast<T*>(m_data_ptr);
  }

  // Get data as const T&.
  template<typename T>
  const T& data() const
  {
    return *reinterpret_cast<const T*>(m_data_ptr);
  }

  const char* display_name() const
  {
    return m_display_name.c_str();
  }
  void set_display_name(const char* display_name)
  {
    m_display_name = display_name;
  }

  const char* group_name() const
  {
    return m_group_name.c_str();
  }
  void set_group_name(const char* group_name)
  {
    m_group_name = group_name;
  }

  Param_kind kind() const
  {
    return m_kind;
  }

  Param_kind array_elem_kind() const
  {
    return m_array_elem_kind;
  }
  mi::Size array_size() const
  {
    return m_array_size;
  }
  mi::Size array_pitch() const
  {
    return m_array_pitch;
  }

  float& range_min()
  {
    return m_range_min;
  }
  float range_min() const
  {
    return m_range_min;
  }
  float& range_max()
  {
    return m_range_max;
  }
  float range_max() const
  {
    return m_range_max;
  }

  template<typename T, int N = 1>
  void update_range()
  {
    T* val_ptr = &data<T>();
    for (int i = 0; i < N; ++i)
    {
      float val = float(val_ptr[i]);
      if (val < m_range_min)
        m_range_min = val;
      if (m_range_max < val)
        m_range_max = val;
    }
  }

  const Enum_type_info* enum_info() const
  {
    return m_enum_info;
  }

private:
  size_t               m_index;
  std::string          m_name;
  std::string          m_display_name;
  std::string          m_group_name;
  Param_kind           m_kind;
  Param_kind           m_array_elem_kind;
  mi::Size             m_array_size;
  mi::Size             m_array_pitch;   // the distance between two array elements
  char*                m_data_ptr;
  float                m_range_min;
  float                m_range_max;
  const Enum_type_info* m_enum_info;
};


struct MaterialDeclaration
{
  std::string nameReference;
  std::string nameMaterial;
  std::string pathMaterial;
};


// This gets store per material reference because the same shader can be reused among multiple references.
struct MaterialMDL
{
  MaterialMDL(const MaterialDeclaration& materialDeclaration);

  bool getIsValid() const;
  void setIsValid(bool value);
  std::string getReference() const;
  std::string getName() const;
  std::string getPath() const;

  void storeMaterialInfo(const int indexShader,
                         mi::neuraylib::IFunction_definition const* mat_def,
                         mi::neuraylib::ICompiled_material const* comp_mat,
                         mi::neuraylib::ITarget_value_layout const* arg_block_layout,
                         mi::neuraylib::ITarget_argument_block const* arg_block);
  void add_sorted_by_group(const Param_info& info);
  void add_enum_type(const std::string name, std::shared_ptr<Enum_type_info> enum_info);
  const Enum_type_info* get_enum_type(const std::string name);
  int getShaderIndex() const;
  char const* name() const;
  std::list<Param_info>& params();
  char* getArgumentBlockData() const;
  size_t getArgumentBlockSize() const;
  std::vector<std::string> const& getReferencedSceneDataNames() const;

  bool m_isValid;

  MaterialDeclaration m_declaration;

  // Index into the m_shaders and m_shaderConfigurations vectors.
  int m_indexShader;
  // Name of the material.
  std::string m_name;
  // Modifiable argument block.
  mi::base::Handle<mi::neuraylib::ITarget_argument_block> m_arg_block;
  // Parameters of the material.
  std::list<Param_info> m_params;

  // Used enum types of the material
  std::map<std::string, std::shared_ptr<Enum_type_info> > m_mapEnumTypes;

  // Scene data names referenced by the material
  std::vector<std::string> m_referencedSceneDataNames;

  // These are indices to the Device class' resource caches.
  std::vector<int> m_indicesToTextures;
  std::vector<int> m_indicesToMBSDFs;
  std::vector<int> m_indicesToLightprofiles;
};

#endif // MATERIAL_MDL_H
