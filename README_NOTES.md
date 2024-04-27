# Building

1. Follow the instructions in README.md
2. `cd build`
3. `npm run cmake` generates the project solution
4. `npm run build` runs `MSBuild.exe`, make sure it's on path

# Demos

Commands to run from the build/bin/Debug/ folder:

.\rtigo12.exe -s system_rtigo12_demo.txt -d scene_rtigo12_demo.txt

.\intro_runtime.exe

.\intro_runtime.exe --miss 0 --light

.\intro_runtime.exe --miss 2 --env NV_Default_HDR_3000x1500.hdr

.\test_app.exe -s ../../../data/living-room/scene-v4.pbrt

.\test_app_2.exe -s system_rtigo12_demo.txt -d scene_rtigo12_demo.txt

.\test_app_3.exe -s system_mdl_demo.txt -d scene_mdl_demo.txt

.\test_app_3.exe -s system_mdl_simple.txt -d scene_mdl_simple.txt

BEST -> .\test_app_3.exe -s system_mdl_simple.txt -d scene_mdl_simple_lights.txt -w 1280 -h 720


.\test_app_3.exe -s system_mdl_demo.txt -d scene_mdl_demo.txt -w 1280 -h 720

.\test_app_3.exe -s system_mdl_hair.txt -d scene_mdl_hair.txt

# Playgrounds

* test_app -> intro_runtime
* test_app_2 -> rtigo12
* test_app_3 -> MDL_renderer