# this file must be in the same folder as the executables e.g. optix_apps/build/bin
# Please read the main README.md for more details.

./rtigo3 -s system_rtigo3_cornell_box.txt -d scene_rtigo3_cornell_box.txt
./rtigo3 -s system_rtigo3_single_gpu_interop.txt -d scene_rtigo3_models.txt

# heavy scene 1 or multi GPU-s
./nvlink_shared -s system_nvlink_shared.txt -d scene_nvlink_spheres_5_5_5.txt

# ???
./bench_shared -s  system_nvlink_shared.txt -d scene_nvlink_spheres_5_5_5.txt

# The rtigo9 and rtigo10 examples use an enhanced scene description where camera and tone mapper values can be 
# overridden and materials for surfaces and lights and all light types themselves can be defined per scene now.
# For that the material definition has changed slightly to support surface and emission distribution functions and some more parameters.
# Read the provided scene_rtigo9_demo.txt file for how to define all supported light types.
./rtigo9 -s system_rtigo9_demo.txt -d scene_rtigo9_demo.txt

# That scene_rtigo9_demo.txt is not using cutout opacity or surface materials on arbitrary mesh lights,
# which means using it with rtigo10 will result in the same image, it will just run considerably faster.

./rtigo10 -s system_rtigo9_demo.txt -d scene_rtigo9_demo.txt

# The rtigo9_omm example uses Opacity Micromaps (OMM) which are built using the OptiX Toolkit CUDA OMM Baking tool (CuOmmBaking.dll).
# The following command loads a generated OBJ file with 15,000 unit quads randomly placed and oriented inside 
# a sphere with radius 20 units. (Generator code is in createQuads()).
# The material assigned to the quads is texture mapped with a leaf texture for albedo and cutout opacity.
# The same command line can be used with rtigo9 to see the performance difference esp. on Ada generation GPUs
# which accelerate OMMs in hardware. (Try higher rendering resolutions than the default 1024x1024.)

./rtigo9_omm -s system_rtigo9_leaf.txt -d scene_rtigo9_leaf.txt


# The rtigo12 example uses a slightly enhanced scene description format than rtigo9 and rtigo10 in that 
# it added material parameters for the volume scattering color, scale and bias.
# Above command lines for rtigo10 work as well, though mesh and rectangle lights will be 1/PI darker due
# to a change from radiant intensity to radiant exitance definition with diffuse EDFs.
# The following scene files demonstrate all BXDF implementations and the volume scattering parameters
# and shows that volumetric shadows just work when placing lights and objects into surrounding objects with volume scattering.

./rtigo12 -s system_rtigo12_demo.txt -d scene_rtigo12_demo.txt
./rtigo12 -s system_rtigo12_scattering_bias.txt -d scene_rtigo12_scattering_bias.txt
./rtigo12 -s system_rtigo12_volume_scattering.txt -d scene_rtigo12_volume_scattering.txt

# The MDL_renderer example uses the NVIDIA Material definition language for the shader generation. 
# The following scene only uses the *.mdl files and resources from the data/mdl folder you copied next 
# to the executable after building the examples. These show most of the fundamental MDL BSDFs, EDFs, 
# VDFs, layers, mixers, modifiers, thin-walled geometry, textures, cutout opacity, base helper functions, etc.
./MDL_renderer -s system_mdl_demo.txt -d scene_mdl_demo.txt

# vMaterials need a 2GB download from https://developer.nvidia.com/vmaterials and some setup, see the main README.md.
./MDL_renderer -s system_mdl_vMaterials.txt -d scene_mdl_vMaterials.txt
./MDL_renderer -s system_mdl_vMaterials_2.4.txt -d scene_mdl_vMaterials_2.4.txt

./mdl_sdf -s system_mdl_demo.txt -d scene_mdl_demo.txt

./intro_runtime
./intro_motion_blur
./intro_driver

# For the curves rendering with MDL hair BSDF materials, issue the command line. 
# That will display a sphere with cubic B-spline curves using a red hair material lit by an
# area light from above. Please read the scene_mdl_hair.txt for other possible material and
# model configurations.

./MDL_renderer -s system_mdl_hair.txt -d scene_mdl_hair.txt


# gltf renderer:  assuming the app is in the optix_apps/build/bin folder!

./GLTF_renderer -f "Buggy.gltf"

# aoshimapier point cloud, you need to download the GLTF file from 
# https://sketchfab.com/3d-models/aoshima-pier-point-cloud-9b8a296659274a39a054ca3408145a1b
# This scene has per-point normals, so the lighting is not intuitive.
# Uncheck the "unlit" box to see spheres instead of flat disks.
# ./GLTF_renderer -f aoshimapierpointcloud/scene.gltf -r 0.00005
