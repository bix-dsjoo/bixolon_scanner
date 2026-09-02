#include "include/bixolon_scanner_sdk/bixolon_scanner_sdk_plugin_c_api.h"

#include <flutter/plugin_registrar_windows.h>

#include "bixolon_scanner_sdk_plugin.h"

void BixolonScannerSdkPluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  bixolon_scanner_sdk::BixolonScannerSdkPlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
