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

#include "inc/Application.h"

#include <dp/math/math.h>
#include <dp/math/Vecnt.h>
#include <dp/math/Matmnt.h>
#include <dp/math/Quatt.h>
#include <dp/math/Trafo.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "inc/MyAssert.h"


std::shared_ptr<sg::Group> Application::createASSIMP(const std::string& filename)
{
  std::map< std::string, std::shared_ptr<sg::Group> >::const_iterator itGroup = m_mapGroups.find(filename);
  if (itGroup != m_mapGroups.end())
  {
    return itGroup->second; // Full model instancing under an Instance node.
  }

  std::ifstream fin(filename);
  if (!fin.fail())
  {
    fin.close(); // Ok, file found.
  }
  else
  {
    std::cerr << "ERROR: createASSIMP() could not open " << filename << '\n';

    // Generate a Group node in any case. It will not have children when the file loading fails. 
    std::shared_ptr<sg::Group> group(new sg::Group(m_idGroup++));
    m_mapGroups[filename] = group; // Allow instancing of this whole model (to fail again quicker next time).
    return group;
  }

  Assimp::Logger::LogSeverity severity = Assimp::Logger::NORMAL; // or Assimp::Logger::VERBOSE;

  Assimp::DefaultLogger::create("", severity, aiDefaultLogStream_STDOUT);               // Create a logger instance for Console Output
  //Assimp::DefaultLogger::create("assimp_log.txt", severity, aiDefaultLogStream_FILE); // Create a logger instance for File Output (found in project folder or near .exe)

  Assimp::DefaultLogger::get()->info("Assimp::DefaultLogger initialized."); // Will add message with "info" tag.
  // Assimp::DefaultLogger::get()->debug(""); // Will add message with "debug" tag.

    unsigned int postProcessSteps = 0
        //| aiProcess_CalcTangentSpace
        //| aiProcess_JoinIdenticalVertices
        //| aiProcess_MakeLeftHanded
        | aiProcess_Triangulate
        //| aiProcess_RemoveComponent
        //| aiProcess_GenNormals
        | aiProcess_GenSmoothNormals
        //| aiProcess_SplitLargeMeshes
        //| aiProcess_PreTransformVertices
        //| aiProcess_LimitBoneWeights
        //| aiProcess_ValidateDataStructure
        //| aiProcess_ImproveCacheLocality
        | aiProcess_RemoveRedundantMaterials
        //| aiProcess_FixInfacingNormals
        | aiProcess_SortByPType
        //| aiProcess_FindDegenerates
        //| aiProcess_FindInvalidData 
        //| aiProcess_GenUVCoords
        //| aiProcess_TransformUVCoords 
        //| aiProcess_FindInstances
        //| aiProcess_OptimizeMeshes 
        //| aiProcess_OptimizeGraph
        //| aiProcess_FlipUVs
        //| aiProcess_FlipWindingOrder
        //| aiProcess_SplitByBoneCount
        //| aiProcess_Debone
        //| aiProcess_GlobalScale
        //| aiProcess_EmbedTextures
        //| aiProcess_ForceGenNormals
        //| aiProcess_DropNormals
        ;

  Assimp::Importer importer;

  if (m_optimize)
  {
    postProcessSteps |= aiProcess_FindDegenerates | aiProcess_OptimizeMeshes | aiProcess_OptimizeGraph;

    // Removing degenerate triangles.
    // If you don't support lines and points, then
    // specify the aiProcess_FindDegenerates flag,
    // specify the aiProcess_SortByPType flag,
    // set the AI_CONFIG_PP_SBP_REMOVE importer property to (aiPrimitiveType_POINT | aiPrimitiveType_LINE).
    importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_POINT | aiPrimitiveType_LINE);
    // This step also removes very small triangles with a surface area smaller than 10^-6.
    // If you rely on having these small triangles, or notice holes in your model,
    // set the property AI_CONFIG_PP_FD_CHECKAREA to false.
    importer.SetPropertyBool(AI_CONFIG_PP_FD_CHECKAREA, false);
    // The degenerate triangles are put into point or line primitives which are then ignored when building the meshes.
    // Finally the traverseScene() function filters out any instance node in the hierarchy which doesn't have a polygonal mesh assigned.
  }

  const aiScene* scene = importer.ReadFile(filename, postProcessSteps);

  // If the import failed, report it
  if (!scene)
  {
    Assimp::DefaultLogger::get()->info(importer.GetErrorString());
    Assimp::DefaultLogger::kill(); // Kill it after the work is done

    std::shared_ptr<sg::Group> group(new sg::Group(m_idGroup++));
    m_mapGroups[filename] = group; // Allow instancing of this whole model (to fail again quicker next time).
    return group;
  }

  // Each scene needs to know where its geometries begin in the m_geometries to calculate the correct mesh index in traverseScene()
  const unsigned int indexSceneBase = static_cast<unsigned int>(m_geometries.size());

  m_remappedMeshIndices.clear(); // Clear the local remapping vector from iMesh to m_geometries index.

  // Create all geometries in the assimp scene with triangle data. Ignore the others and remap their geometry indices.
  for (unsigned int iMesh = 0; iMesh < scene->mNumMeshes; ++iMesh)
  {
    const aiMesh* mesh = scene->mMeshes[iMesh];
    
    unsigned int remapMeshToGeometry = ~0u; // Remap mesh index to geometry index. ~0 means there was no geometry for a mesh.

    // The post-processor took care of meshes per primitive type and triangulation.
    // Need to do a bitwise comparison of the mPrimitiveTypes here because newer ASSIMP versions
    // indicate triangulated former polygons with the additional aiPrimitiveType_NGON EncodingFlag.
    if ((mesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) && 2 < mesh->mNumVertices)
    {
      std::vector<TriangleAttributes> attributes(mesh->mNumVertices);
      
      bool needsTangents  = false;
      bool needsNormals   = false;
      bool needsTexcoords = false;

      for (unsigned int iVertex = 0; iVertex < mesh->mNumVertices; ++iVertex)
      {
        TriangleAttributes& attrib = attributes[iVertex];

        const aiVector3D& v = mesh->mVertices[iVertex];
        attrib.vertex = make_float3(v.x, v.y, v.z);

        if (mesh->HasTangentsAndBitangents())
        {
          const aiVector3D& t = mesh->mTangents[iVertex];
          attrib.tangent = make_float3(t.x, t.y, t.z);
        }
        else
        {
          needsTangents = true;
          attrib.tangent = make_float3(1.0f, 0.0f, 0.0f);
        }

        if (mesh->HasNormals())
        {
          const aiVector3D& n = mesh->mNormals[iVertex];
          attrib.normal = make_float3(n.x, n.y, n.z);
        }
        else
        {
          needsNormals = true;
          attrib.normal = make_float3(0.0f, 0.0f, 1.0f);
        }

        if (mesh->HasTextureCoords(0))
        {
          const aiVector3D& t = mesh->mTextureCoords[0][iVertex];
          attrib.texcoord = make_float3(t.x, t.y, t.z);
        }
        else
        {
          needsTexcoords = true;
          attrib.texcoord = make_float3(0.0f, 0.0f, 0.0f);
        }
      }

      std::vector<unsigned int> indices;

      for (unsigned int iFace = 0; iFace < mesh->mNumFaces; ++iFace)
      {
        const struct aiFace* face = &mesh->mFaces[iFace];
        MY_ASSERT(face->mNumIndices == 3); // Must be true because of aiProcess_Triangulate.

        for (unsigned int iIndex = 0; iIndex < face->mNumIndices; ++iIndex)
        {
          indices.push_back(face->mIndices[iIndex]);
        }
      }

      //if (needsNormals) // Assimp handled that via the aiProcess_GenSmoothNormals flag.
      //{
      //  calculateNormals(attributes, indices);
      //}
      if (needsTangents)
      {
        calculateTangents(attributes, indices); // This calculates geometry tangents though.
      }

      remapMeshToGeometry = static_cast<unsigned int>(m_geometries.size());

      std::shared_ptr<sg::Triangles> geometry(new sg::Triangles(m_idGeometry++));
      geometry->setAttributes(attributes);
      geometry->setIndices(indices);
      
      // Note that the ASSIMP meshes are not tracked inside the m_mapGeometries!
      // They are all unique and material assignments are fixed. => No issues with GAS without and with OMM.
      // Only the full model group can be instatiated via the m_mapGroups.
      m_geometries.push_back(geometry);
    }

    m_remappedMeshIndices.push_back(remapMeshToGeometry); 
  }

  std::shared_ptr<sg::Group> group = traverseScene(scene, indexSceneBase, scene->mRootNode);
  m_mapGroups[filename] = group; // Allow instancing of this whole model.
  
  Assimp::DefaultLogger::kill(); // Kill it after the work is done

  return group;
}
  
std::shared_ptr<sg::Group> Application::traverseScene(const struct aiScene *scene, const unsigned int indexSceneBase, const struct aiNode* node)
{
  // Create a group to hold all children and all meshes of this node.
  std::shared_ptr<sg::Group> group(new sg::Group(m_idGroup++));

  const aiMatrix4x4& m = node->mTransformation;

  const float trafo[12] =
  {
    float(m.a1), float(m.a2), float(m.a3), float(m.a4),
    float(m.b1), float(m.b2), float(m.b3), float(m.b4),
    float(m.c1), float(m.c2), float(m.c3), float(m.c4)
  };

  // Need to do a depth first traversal here to attach the bottom most nodes to each node's group.
  for (unsigned int iChild = 0; iChild < node->mNumChildren; ++iChild)
  {
    std::shared_ptr<sg::Group> child = traverseScene(scene, indexSceneBase, node->mChildren[iChild]);

    // Create an instance which holds the subtree.
    std::shared_ptr<sg::Instance> instance(new sg::Instance(m_idInstance++));

    instance->setTransform(trafo);
    instance->setChild(child);

    group->addChild(instance); 
  }

  // Now also gather all meshes assigned to this node.
  for (unsigned int iMesh = 0; iMesh < node->mNumMeshes; ++iMesh)
  {
    const unsigned int indexMesh = node->mMeshes[iMesh];  // Original mesh index in the assimp scene.
    MY_ASSERT(indexMesh < m_remappedMeshIndices.size())

    if (m_remappedMeshIndices[indexMesh] != ~0u) // If there exists a Triangles geometry for this assimp mesh, then build the Instance.
    {
      const unsigned int indexGeometry = m_remappedMeshIndices[indexMesh];
      
      // Create an instance with the current nodes transformation and append it to the parent group.
      std::shared_ptr<sg::Instance> instance(new sg::Instance(m_idInstance++));
      
      instance->setTransform(trafo);
      instance->setChild(m_geometries[indexGeometry]);

      const struct aiMesh* mesh = scene->mMeshes[indexMesh];

      // Allow to specify different materials per assimp model by using the filename (no path no extension) and the material index.
      struct aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

      std::string nameMaterialReference;
      aiString materialName;
      if (material->Get(AI_MATKEY_NAME, materialName) == aiReturn_SUCCESS)
      {
        nameMaterialReference = std::string(materialName.C_Str());
      }

      int indexMaterial = -1;
      std::map<std::string, int>::const_iterator itm = m_mapMaterialReferences.find(nameMaterialReference);
      if (itm != m_mapMaterialReferences.end())
      {
        indexMaterial = itm->second;
        
        // The materials had been created with default albedo colors.
        // Change it to the diffuse color of the assimp material.
        aiColor4D diffuse;
        if (material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse) == aiReturn_SUCCESS)
        {
          m_materialsGUI[indexMaterial].colorAlbedo = make_float3(diffuse.r, diffuse.g, diffuse.b);
        }
      }
      else
      {
        std::cerr << "WARNING: traverseScene() No material found for " << nameMaterialReference << ". Trying default.\n";

        std::map<std::string, int>::const_iterator itmd = m_mapMaterialReferences.find(std::string("default"));
        if (itmd != m_mapMaterialReferences.end())
        {
          indexMaterial = itmd->second;
        }
        else 
        {
          std::cerr << "ERROR: loadSceneDescription() No default material found\n";
        }
      }
      instance->setMaterial(indexMaterial);

      group->addChild(instance);
    }
  }
  return group;
}
