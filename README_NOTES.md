# Building

1. Follow the instructions in README.md
2. `cd build`
3. `npm run cmake` generates the project solution
4. `npm run build` runs `MSBuild.exe`, make sure it's on path

# Demos

.\rtigo12.exe -s system_rtigo12_demo.txt -d scene_rtigo12_demo.txt

.\intro_runtime.exe

.\intro_runtime.exe --miss 0 --light

.\intro_runtime.exe --miss 2 --env NV_Default_HDR_3000x1500.hdr