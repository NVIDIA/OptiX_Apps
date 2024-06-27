/* 
 * Copyright (c) 2013-2024, NVIDIA CORPORATION. All rights reserved.
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


#include "cuda/config.h"

#include "Options.h"
#include "Application.h"

#include <IL/il.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>


static Application* g_app = nullptr;

static bool displayGUI = true;

static void error_callback(int error, const char* description)
{
  std::cerr << "ERROR: "<< error << ": " << description << '\n';
}


// Drag-and-Drop callback.
static void callbackDropFunction(GLFWwindow* window, int countPaths, const char* paths[])
{
  // Route the dropped files to the application and let that handle the contents.
  if (g_app)
  {
    g_app->drop(countPaths, paths);
  }
}


int runApp(Options const& options)
{
  int widthClient  = std::max(1, options.getWidthClient());
  int heightClient = std::max(1, options.getHeightClient());

  //glfwWindowHint(GLFW_DECORATED, windowBorder);

  GLFWwindow* window = glfwCreateWindow(widthClient, heightClient, "GLTF_renderer - Copyright (c) 2024 NVIDIA Corporation", NULL, NULL);
  if (!window)
  {
    error_callback(APP_ERROR_CREATE_WINDOW, "glfwCreateWindow() failed.");
    return APP_ERROR_CREATE_WINDOW;
  }

  glfwSetDropCallback(window, callbackDropFunction); // Implement drag-and-drop callback.

  glfwMakeContextCurrent(window);

  if (glewInit() != GL_NO_ERROR)
  {
    error_callback(APP_ERROR_GLEW_INIT, "GLEW failed to initialize.");
    return APP_ERROR_GLEW_INIT;
  }
    
  ilInit(); // Initialize DevIL once.

  g_app = new Application(window, options);

  std::chrono::steady_clock::time_point time0;
  std::chrono::steady_clock::time_point time1;

  // Main loop
  while (!glfwWindowShouldClose(window))
  {
    glfwPollEvents(); // Render continuously.

    glfwGetFramebufferSize(window, &widthClient, &heightClient);
    g_app->reshape(widthClient, heightClient);
    
    const int benchmarkMode = g_app->getBenchmarkMode();
    if (benchmarkMode == 1) // FPS mode
    {
      time0 = std::chrono::steady_clock::now(); // Start time.
    }

    if (g_app->render()) // OptiX rendering. Returns true when there is a new image which is not the case while picking.
    {
      g_app->update();
    }
    g_app->guiNewFrame();
    //g_app->guiReferenceManual(); // The ImGui "Programming Manual" as example code.
    g_app->guiWindow();
    g_app->guiEventHandler(); // SPACE to toggle the GUI windows and all mouse tracking via GuiState.
    g_app->display();         // OpenGL display always required to lay the background for the GUI.
    g_app->guiRender();       // Render all ImGUI elements at last.

    glfwSwapBuffers(window);

    if (benchmarkMode == 1) // FPS mode
    {
      // Wait for the OpenGL SwapBuffers() to have finished. 
      // If the result is limited to the monitor refresh Hz, disable VSYNC inside the NVIDIA Control Panel!
      // Never benchmark graphics performance with VSYNC enabled!
      glFinish();

      time1 = std::chrono::steady_clock::now(); // End time.

      std::chrono::duration<double> timeRender = time1 - time0;
      const float milliseconds = std::chrono::duration<float, std::milli>(timeRender).count();

      g_app->setBenchmarkValue(1000.0f / milliseconds); // Convert ms/frame to frames/second.
    }

    // Instead of glfwPollEvents() this would only render when an event is happening.
    // This requires glfwPostEmptyEvent() when ending an action to prevent GUI lagging one frame behind.
    //glfwWaitEvents();
  }

  delete g_app;

  ilShutDown();

  return APP_EXIT_SUCCESS; // Success.
}


int main(int argc, char *argv[])
{
  glfwSetErrorCallback(error_callback);

  if (!glfwInit())
  {
    error_callback(APP_ERROR_GLFW_INIT, "GLFW failed to initialize.");
    return APP_ERROR_GLFW_INIT;
  }

  int result = APP_ERROR_UNKNOWN;

  Options options;
  
  if (options.parseCommandLine(argc, argv))
  {
    try
    {
      result = runApp(options);
    }
    catch(std::exception& e)
    {
      std::cerr << "ERROR: Caught exception: " << e.what() << "\n";
    }
  }

  glfwTerminate();

  return result;
}
