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