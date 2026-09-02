#ifndef FLUTTER_PLUGIN_BIXOLON_SCANNER_SDK_PLUGIN_H_
#define FLUTTER_PLUGIN_BIXOLON_SCANNER_SDK_PLUGIN_H_

#include <windows.h>

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>

#include <memory>

namespace bixolon_scanner_sdk {

class BixolonScannerSdkPlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarWindows* registrar);

  BixolonScannerSdkPlugin();
  ~BixolonScannerSdkPlugin() override;

  BixolonScannerSdkPlugin(const BixolonScannerSdkPlugin&) = delete;
  BixolonScannerSdkPlugin& operator=(const BixolonScannerSdkPlugin&) = delete;

 private:
  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue>& method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);
  bool IsWorkerRunning();
  void ReleaseHandles(bool terminate_process);

  HANDLE process_handle_ = nullptr;
  HANDLE job_handle_ = nullptr;
  bool terminate_with_app_ = true;
};

}  // namespace bixolon_scanner_sdk

#endif
