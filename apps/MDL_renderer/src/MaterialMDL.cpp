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

#include "inc/MaterialMDL.h"

#include <algorithm>
#include <iostream>
//#include <sstream>

#include "inc/MyAssert.h"


MaterialMDL::MaterialMDL(const MaterialDeclaration& declaration)
  : m_isValid(false) // Set to true once all MDL and OptiX initializations have been done.
  , m_declaration(declaration)
  , m_indexShader(-1) // Invalid
{
}

bool MaterialMDL::getIsValid() const
{
  return m_isValid;
}

void MaterialMDL::setIsValid(bool value)
{
  m_isValid = value;
}


std::string MaterialMDL::getReference() const
{
  return m_declaration.nameReference;
}

std::string MaterialMDL::getName() const
{
  return m_declaration.nameMaterial;
}

std::string MaterialMDL::getPath() const
{
  return m_declaration.pathMaterial;
}

void MaterialMDL::storeMaterialInfo(const int indexShader,
                                    mi::neuraylib::IFunction_definition const* mat_def,
                                    mi::neuraylib::ICompiled_material const* comp_mat,
                                    mi::neuraylib::ITarget_value_layout const* arg_block_layout,
                                    mi::neuraylib::ITarget_argument_block const* arg_block)
{
  m_indexShader = indexShader;
  m_name = mat_def->get_mdl_name();

  char* arg_block_data = nullptr;

  if (arg_block != nullptr)
  {
    m_arg_block = mi::base::Handle<mi::neuraylib::ITarget_argument_block>(arg_block->clone());
    arg_block_data = m_arg_block->get_data();
  }

  mi::base::Handle<mi::neuraylib::IAnnotation_list const> anno_list(mat_def->get_parameter_annotations());

  for (mi::Size j = 0, num_params = comp_mat->get_parameter_count(); j < num_params; ++j)
  {
    const char* name = comp_mat->get_parameter_name(j);
    if (name == nullptr)
    {
      continue;
    }

    // Determine the type of the argument
    mi::base::Handle<mi::neuraylib::IValue const> arg(comp_mat->get_argument(j));
    mi::neuraylib::IValue::Kind kind = arg->get_kind();

    Param_info::Param_kind param_kind = Param_info::PK_UNKNOWN;
    Param_info::Param_kind param_array_elem_kind = Param_info::PK_UNKNOWN;
    mi::Size               param_array_size = 0;
    mi::Size               param_array_pitch = 0;

    const Enum_type_info* enum_type = nullptr;

    switch (kind)
    {
      case mi::neuraylib::IValue::VK_FLOAT:
        param_kind = Param_info::PK_FLOAT;
        break;
      case mi::neuraylib::IValue::VK_COLOR:
        param_kind = Param_info::PK_COLOR;
        break;
      case mi::neuraylib::IValue::VK_BOOL:
        param_kind = Param_info::PK_BOOL;
        break;
      case mi::neuraylib::IValue::VK_INT:
        param_kind = Param_info::PK_INT;
        break;
      case mi::neuraylib::IValue::VK_VECTOR:
      {
        mi::base::Handle<mi::neuraylib::IValue_vector const> val(arg.get_interface<mi::neuraylib::IValue_vector const>());
        mi::base::Handle<mi::neuraylib::IType_vector const> val_type(val->get_type());
        mi::base::Handle<mi::neuraylib::IType_atomic const> elem_type(val_type->get_element_type());

        if (elem_type->get_kind() == mi::neuraylib::IType::TK_FLOAT)
        {
          switch (val_type->get_size())
          {
            case 2:
              param_kind = Param_info::PK_FLOAT2;
              break;
            case 3:
              param_kind = Param_info::PK_FLOAT3;
              break;
          }
        }
      }
      break;
      case mi::neuraylib::IValue::VK_ARRAY:
      {
        mi::base::Handle<mi::neuraylib::IValue_array const> val(arg.get_interface<mi::neuraylib::IValue_array const>());
        mi::base::Handle<mi::neuraylib::IType_array const> val_type(val->get_type());
        mi::base::Handle<mi::neuraylib::IType const> elem_type(val_type->get_element_type());

        // we currently only support arrays of some values
        switch (elem_type->get_kind())
        {
          case mi::neuraylib::IType::TK_FLOAT:
            param_array_elem_kind = Param_info::PK_FLOAT;
            break;
          case mi::neuraylib::IType::TK_COLOR:
            param_array_elem_kind = Param_info::PK_COLOR;
            break;
          case mi::neuraylib::IType::TK_BOOL:
            param_array_elem_kind = Param_info::PK_BOOL;
            break;
          case mi::neuraylib::IType::TK_INT:
            param_array_elem_kind = Param_info::PK_INT;
            break;
          case mi::neuraylib::IType::TK_VECTOR:
          {
            mi::base::Handle<mi::neuraylib::IType_vector const> val_type(elem_type.get_interface<mi::neuraylib::IType_vector const>());
            mi::base::Handle<mi::neuraylib::IType_atomic const> velem_type(val_type->get_element_type());

            if (velem_type->get_kind() == mi::neuraylib::IType::TK_FLOAT)
            {
              switch (val_type->get_size())
              {
                case 2:
                  param_array_elem_kind = Param_info::PK_FLOAT2;
                  break;
                case 3:
                  param_array_elem_kind = Param_info::PK_FLOAT3;
                  break;
              }
            }
          }
          break;
          default:
            break;
        }
        if (param_array_elem_kind != Param_info::PK_UNKNOWN)
        {
          param_kind = Param_info::PK_ARRAY;
          param_array_size = val_type->get_size();

          // determine pitch of array if there are at least two elements
          if (param_array_size > 1)
          {
            mi::neuraylib::Target_value_layout_state array_state(arg_block_layout->get_nested_state(j));
            mi::neuraylib::Target_value_layout_state next_elem_state(arg_block_layout->get_nested_state(1, array_state));

            mi::neuraylib::IValue::Kind kind;
            mi::Size param_size;

            mi::Size start_offset = arg_block_layout->get_layout(kind, param_size, array_state);
            mi::Size next_offset = arg_block_layout->get_layout(kind, param_size, next_elem_state);

            param_array_pitch = next_offset - start_offset;
          }
        }
      }
      break;
      case mi::neuraylib::IValue::VK_ENUM:
      {
        mi::base::Handle<mi::neuraylib::IValue_enum const> val(arg.get_interface<mi::neuraylib::IValue_enum const>());
        mi::base::Handle<mi::neuraylib::IType_enum const> val_type(val->get_type());

        // prepare info for this enum type if not seen so far
        const Enum_type_info* info = get_enum_type(val_type->get_symbol());
        if (info == nullptr)
        {
          std::shared_ptr<Enum_type_info> p(new Enum_type_info());

          for (mi::Size i = 0, n = val_type->get_size(); i < n; ++i)
          {
            p->add(val_type->get_value_name(i), val_type->get_value_code(i));
          }
          add_enum_type(val_type->get_symbol(), p);
          info = p.get();
        }
        enum_type = info;

        param_kind = Param_info::PK_ENUM;
      }
      break;
      case mi::neuraylib::IValue::VK_STRING:
        param_kind = Param_info::PK_STRING;
        break;
      case mi::neuraylib::IValue::VK_TEXTURE:
        param_kind = Param_info::PK_TEXTURE;
        break;
      case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
        param_kind = Param_info::PK_LIGHT_PROFILE;
        break;
      case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
        param_kind = Param_info::PK_BSDF_MEASUREMENT;
        break;
      default:
        // Unsupported? -> skip
        continue;
    }

    // Get the offset of the argument within the target argument block
    mi::neuraylib::Target_value_layout_state state(arg_block_layout->get_nested_state(j));
    mi::neuraylib::IValue::Kind kind2;
    mi::Size param_size;
    mi::Size offset = arg_block_layout->get_layout(kind2, param_size, state);
    if (kind != kind2)
    {
      continue;  // layout is invalid -> skip
    }

    Param_info param_info(j,
                          name,
                          name,
                          /*group_name=*/ "",
                          param_kind,
                          param_array_elem_kind,
                          param_array_size,
                          param_array_pitch,
                          arg_block_data + offset,
                          enum_type);

    // Check for annotation info
    mi::base::Handle<mi::neuraylib::IAnnotation_block const> anno_block(anno_list->get_annotation_block(name));
    if (anno_block)
    {
      mi::neuraylib::Annotation_wrapper annos(anno_block.get());
      mi::Size anno_index = annos.get_annotation_index("::anno::hard_range(float,float)");
      if (anno_index != mi::Size(-1))
      {
        annos.get_annotation_param_value(anno_index, 0, param_info.range_min());
        annos.get_annotation_param_value(anno_index, 1, param_info.range_max());
      }
      else
      {
        anno_index = annos.get_annotation_index("::anno::soft_range(float,float)");
        if (anno_index != mi::Size(-1))
        {
          annos.get_annotation_param_value(anno_index, 0, param_info.range_min());
          annos.get_annotation_param_value(anno_index, 1, param_info.range_max());
        }
      }
      anno_index = annos.get_annotation_index("::anno::display_name(string)");
      if (anno_index != mi::Size(-1))
      {
        char const* display_name = nullptr;
        annos.get_annotation_param_value(anno_index, 0, display_name);
        param_info.set_display_name(display_name);
      }
      anno_index = annos.get_annotation_index("::anno::in_group(string)");
      if (anno_index != mi::Size(-1))
      {
        char const* group_name = nullptr;
        annos.get_annotation_param_value(anno_index, 0, group_name);
        param_info.set_group_name(group_name);
      }
    }

    add_sorted_by_group(param_info);
  }
}

// Add the parameter information as last entry of the corresponding group, or to the
// end of the list, if no group name is available.
void MaterialMDL::add_sorted_by_group(const Param_info& info)
{
  bool group_found = false;
  if (info.group_name() != nullptr)
  {
    for (std::list<Param_info>::iterator it = params().begin(); it != params().end(); ++it)
    {
      const bool same_group = (it->group_name() != nullptr && strcmp(it->group_name(), info.group_name()) == 0);
      if (group_found && !same_group)
      {
        m_params.insert(it, info);
        return;
      }
      if (same_group)
      {
        group_found = true;
      }
    }
  }
  m_params.push_back(info);
}

// Add a new enum type to the map of used enum types.
void MaterialMDL::add_enum_type(const std::string name, std::shared_ptr<Enum_type_info> enum_info)
{
  m_mapEnumTypes[name] = enum_info;
}

// Lookup enum type info for a given enum type absolute MDL name.
const Enum_type_info* MaterialMDL::get_enum_type(const std::string name)
{
  std::map<std::string, std::shared_ptr<Enum_type_info> >::const_iterator it = m_mapEnumTypes.find(name);
  if (it != m_mapEnumTypes.end())
  {
    return it->second.get();
  }
  return nullptr;
}

int MaterialMDL::getShaderIndex() const
{
  return m_indexShader;
}

// Get the name of the material.
char const* MaterialMDL::name() const
{
  return m_name.c_str();
}

// Get the parameters of this material.
std::list<Param_info>& MaterialMDL::params()
{
  return m_params;
}

// Get the modifiable argument block data.
char* MaterialMDL::getArgumentBlockData() const
{
  if (!m_arg_block)
  {
    return nullptr;
  }
  return m_arg_block->get_data();
}

// Get the modifiable argument block size.
size_t MaterialMDL::getArgumentBlockSize() const
{
  if (!m_arg_block)
  {
    return 0;
  }
  return m_arg_block->get_size();
}

// Get the scene data names referenced by the compiled material.
std::vector<std::string> const& MaterialMDL::getReferencedSceneDataNames() const
{
  return m_referencedSceneDataNames;
}

