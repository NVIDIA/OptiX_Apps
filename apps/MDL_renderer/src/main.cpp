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

#include "shaders/config.h"

#include "inc/Application.h"

#include <IL/il.h>

#include <algorithm>
#include <iostream>


static Application* g_app = nullptr;

static void callbackError(int error, const char* description)
{
  std::cerr << "ERROR: "<< error << ": " << description << '\n';
}


static int runApp(const Options& options)
{
  int width  = std::max(1, options.getWidth());
  int height = std::max(1, options.getHeight());
 
  GLFWwindow* window = glfwCreateWindow(width, height, "MDL_renderer - Copyright (c) 2023 NVIDIA Corporation", NULL, NULL);
  if (!window)
  {
    callbackError(APP_ERROR_CREATE_WINDOW, "glfwCreateWindow() failed.");
    return APP_ERROR_CREATE_WINDOW;
  }

  glfwMakeContextCurrent(window);

  if (glewInit() != GL_NO_ERROR)
  {
    callbackError(APP_ERROR_GLEW_INIT, "GLEW failed to initialize.");
    return APP_ERROR_GLEW_INIT;
  }
    
  ilInit(); // Initialize DevIL once.

  g_app = new Application(window, options);

  if (!g_app->isValid())
  {
    std::cerr << "ERROR: Application() failed to initialize successfully.\n";
    ilShutDown();
    return APP_ERROR_APP_INIT;
  }

  const int mode = std::max(0, options.getMode());

  if (mode == 0) // Interactive, default.
  {
    // Main loop
    bool finish = false;
    while (!finish && !glfwWindowShouldClose(window))
    {
      glfwPollEvents(); // Render continuously. Battery drainer!

      glfwGetFramebufferSize(window, &width, &height);

      g_app->reshape(width, height);
      g_app->guiNewFrame();
      //g_app->guiReferenceManual();  // HACK The ImGUI "Programming Manual" as example code.
      g_app->guiWindow();             // This application's GUI window rendering commands. 
      g_app->guiEventHandler();       // SPACE to toggle the GUI windows and all mouse tracking via GuiState.
      finish = g_app->render();       // OptiX rendering, returns true when benchmark is enabled and the samples per pixel have been rendered.
      g_app->display();               // OpenGL display always required to lay the background for the GUI.
      g_app->guiRender();             // Render all ImGUI elements at last.

      glfwSwapBuffers(window);

      //glfwWaitEvents(); // Render only when an event is happening. Needs some glfwPostEmptyEvent() to prevent GUI lagging one frame behind when ending an action.
    }
  }
  else if (mode == 1) // Batched benchmark single shot.
  {
    g_app->benchmark();
  }

  delete g_app;

  ilShutDown();

  return APP_EXIT_SUCCESS;
}


int main(int argc, char *argv[])
{
  glfwSetErrorCallback(callbackError);

  if (!glfwInit())
  {
    callbackError(APP_ERROR_GLFW_INIT, "GLFW failed to initialize.");
    return APP_ERROR_GLFW_INIT;
  }

  int result = APP_ERROR_UNKNOWN;

  Options options;

  if (options.parseCommandLine(argc, argv))
  {
    result = runApp(options);
  }

  glfwTerminate();

  return result;
}
