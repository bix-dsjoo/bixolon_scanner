#include <flutter/dart_project.h>
#include <flutter/flutter_view_controller.h>
#include <windows.h>

#include "flutter_window.h"
#include "utils.h"
#include "worker_process.h"

namespace {

constexpr unsigned int kInitialWindowWidth = 1440;
constexpr unsigned int kInitialWindowHeight = 900;
constexpr unsigned int kMinimumWindowWidth = 1280;
constexpr unsigned int kMinimumWindowHeight = 720;

}  // namespace

int APIENTRY wWinMain(_In_ HINSTANCE instance, _In_opt_ HINSTANCE prev,
                      _In_ wchar_t *command_line, _In_ int show_command) {
  // Attach to console when present (e.g., 'flutter run') or create a
  // new console when running with a debugger.
  if (!::AttachConsole(ATTACH_PARENT_PROCESS) && ::IsDebuggerPresent()) {
    CreateAndAttachConsole();
  }

  // Initialize COM, so that it is available for use in the library and/or
  // plugins.
  ::CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

  flutter::DartProject project(L"data");

  WorkerProcess worker_process;
  worker_process.Start();

  std::vector<std::string> command_line_arguments =
      GetCommandLineArguments();

  project.set_dart_entrypoint_arguments(std::move(command_line_arguments));

  FlutterWindow window(project);
  Win32Window::Point origin(10, 10);
  Win32Window::Size size(kInitialWindowWidth, kInitialWindowHeight);
  const std::string product_version = BIXOLON_PRODUCT_VERSION;
  const std::wstring window_title =
      L"BIXOLON Scanner v" +
      std::wstring(product_version.begin(), product_version.end());
  window.SetMinimumSize(
      Win32Window::Size(kMinimumWindowWidth, kMinimumWindowHeight));
  if (!window.Create(window_title.c_str(), origin, size)) {
    worker_process.Stop();
    return EXIT_FAILURE;
  }
  window.SetQuitOnClose(true);

  ::MSG msg;
  while (::GetMessage(&msg, nullptr, 0, 0)) {
    ::TranslateMessage(&msg);
    ::DispatchMessage(&msg);
  }

  worker_process.Stop();
  ::CoUninitialize();
  return EXIT_SUCCESS;
}
