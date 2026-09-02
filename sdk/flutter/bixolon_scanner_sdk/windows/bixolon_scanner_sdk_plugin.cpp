#include "bixolon_scanner_sdk_plugin.h"

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>
#include <flutter/standard_method_codec.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace bixolon_scanner_sdk {
namespace {

struct CaseInsensitiveWideLess {
  bool operator()(const std::wstring& left, const std::wstring& right) const {
    return _wcsicmp(left.c_str(), right.c_str()) < 0;
  }
};

std::wstring Utf8ToWide(const std::string& value) {
  if (value.empty()) {
    return std::wstring();
  }
  const int length = MultiByteToWideChar(
      CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
      static_cast<int>(value.size()), nullptr, 0);
  if (length <= 0) {
    return std::wstring();
  }
  std::wstring converted(length, L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                      static_cast<int>(value.size()), converted.data(), length);
  return converted;
}

const std::string* StringValue(const flutter::EncodableMap& arguments,
                               const char* name) {
  const auto iterator = arguments.find(flutter::EncodableValue(name));
  if (iterator == arguments.end()) {
    return nullptr;
  }
  return std::get_if<std::string>(&iterator->second);
}

bool BoolValue(const flutter::EncodableMap& arguments, const char* name,
               bool fallback) {
  const auto iterator = arguments.find(flutter::EncodableValue(name));
  if (iterator == arguments.end()) {
    return fallback;
  }
  const auto* value = std::get_if<bool>(&iterator->second);
  return value == nullptr ? fallback : *value;
}

std::vector<wchar_t> BuildEnvironmentBlock(
    const flutter::EncodableMap& overrides) {
  std::map<std::wstring, std::wstring, CaseInsensitiveWideLess> variables;
  LPWCH current_environment = GetEnvironmentStringsW();
  if (current_environment != nullptr) {
    for (LPWCH entry = current_environment; *entry != L'\0';) {
      std::wstring item(entry);
      entry += item.size() + 1;
      const size_t separator = item.find(L'=', item[0] == L'=' ? 1 : 0);
      if (separator != std::wstring::npos) {
        variables[item.substr(0, separator)] = item.substr(separator + 1);
      }
    }
    FreeEnvironmentStringsW(current_environment);
  }

  for (const auto& pair : overrides) {
    const auto* key = std::get_if<std::string>(&pair.first);
    const auto* value = std::get_if<std::string>(&pair.second);
    if (key != nullptr && value != nullptr) {
      variables[Utf8ToWide(*key)] = Utf8ToWide(*value);
    }
  }

  std::vector<wchar_t> block;
  for (const auto& pair : variables) {
    const std::wstring item = pair.first + L"=" + pair.second;
    block.insert(block.end(), item.begin(), item.end());
    block.push_back(L'\0');
  }
  block.push_back(L'\0');
  return block;
}

}  // namespace

void BixolonScannerSdkPlugin::RegisterWithRegistrar(
    flutter::PluginRegistrarWindows* registrar) {
  auto channel =
      std::make_unique<flutter::MethodChannel<flutter::EncodableValue>>(
          registrar->messenger(), "bixolon_scanner_sdk/worker",
          &flutter::StandardMethodCodec::GetInstance());
  auto plugin = std::make_unique<BixolonScannerSdkPlugin>();
  channel->SetMethodCallHandler(
      [plugin_pointer = plugin.get()](const auto& call, auto result) {
        plugin_pointer->HandleMethodCall(call, std::move(result));
      });
  registrar->AddPlugin(std::move(plugin));
}

BixolonScannerSdkPlugin::BixolonScannerSdkPlugin() = default;

BixolonScannerSdkPlugin::~BixolonScannerSdkPlugin() {
  ReleaseHandles(terminate_with_app_);
}

bool BixolonScannerSdkPlugin::IsWorkerRunning() {
  if (process_handle_ == nullptr) {
    return false;
  }
  DWORD exit_code = 0;
  if (!GetExitCodeProcess(process_handle_, &exit_code) ||
      exit_code != STILL_ACTIVE) {
    ReleaseHandles(false);
    return false;
  }
  return true;
}

void BixolonScannerSdkPlugin::ReleaseHandles(bool terminate_process) {
  if (terminate_process && process_handle_ != nullptr) {
    TerminateProcess(process_handle_, 0);
    WaitForSingleObject(process_handle_, 3000);
  }
  if (process_handle_ != nullptr) {
    CloseHandle(process_handle_);
    process_handle_ = nullptr;
  }
  if (job_handle_ != nullptr) {
    CloseHandle(job_handle_);
    job_handle_ = nullptr;
  }
}

void BixolonScannerSdkPlugin::HandleMethodCall(
    const flutter::MethodCall<flutter::EncodableValue>& method_call,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  if (method_call.method_name() == "isWorkerRunning") {
    result->Success(flutter::EncodableValue(IsWorkerRunning()));
    return;
  }

  if (method_call.method_name() == "stopWorker") {
    ReleaseHandles(true);
    result->Success();
    return;
  }

  if (method_call.method_name() != "ensureWorkerStarted") {
    result->NotImplemented();
    return;
  }

  if (IsWorkerRunning()) {
    result->Success(flutter::EncodableValue(true));
    return;
  }

  const auto* arguments =
      std::get_if<flutter::EncodableMap>(method_call.arguments());
  if (arguments == nullptr) {
    result->Error("INVALID_ARGUMENTS", "Worker launch arguments are required.");
    return;
  }
  const auto* executable_value = StringValue(*arguments, "executable");
  const auto* directory_value = StringValue(*arguments, "workingDirectory");
  const auto environment_iterator =
      arguments->find(flutter::EncodableValue("environment"));
  if (executable_value == nullptr || directory_value == nullptr ||
      environment_iterator == arguments->end()) {
    result->Error("INVALID_ARGUMENTS", "Worker launch arguments are incomplete.");
    return;
  }
  const auto* environment =
      std::get_if<flutter::EncodableMap>(&environment_iterator->second);
  if (environment == nullptr) {
    result->Error("INVALID_ARGUMENTS", "environment must be a string map.");
    return;
  }

  const std::wstring executable = Utf8ToWide(*executable_value);
  const std::wstring working_directory = Utf8ToWide(*directory_value);
  if (executable.empty() || GetFileAttributesW(executable.c_str()) ==
                                INVALID_FILE_ATTRIBUTES) {
    result->Error("WORKER_NOT_FOUND", "Worker executable was not found.");
    return;
  }

  terminate_with_app_ = BoolValue(*arguments, "terminateWithApp", true);
  std::vector<wchar_t> environment_block = BuildEnvironmentBlock(*environment);
  std::wstring command_line = L"\"" + executable + L"\"";
  std::vector<wchar_t> mutable_command(command_line.begin(), command_line.end());
  mutable_command.push_back(L'\0');

  STARTUPINFOW startup_info{};
  startup_info.cb = sizeof(startup_info);
  PROCESS_INFORMATION process_info{};
  const DWORD creation_flags =
      CREATE_NO_WINDOW | CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT;
  if (!CreateProcessW(executable.c_str(), mutable_command.data(), nullptr,
                      nullptr, FALSE, creation_flags, environment_block.data(),
                      working_directory.c_str(), &startup_info, &process_info)) {
    result->Error("WORKER_START_FAILED", "Windows could not start the Worker.",
                  flutter::EncodableValue(static_cast<int>(GetLastError())));
    return;
  }

  process_handle_ = process_info.hProcess;
  if (terminate_with_app_) {
    job_handle_ = CreateJobObjectW(nullptr, nullptr);
    if (job_handle_ == nullptr) {
      CloseHandle(process_info.hThread);
      ReleaseHandles(true);
      result->Error("JOB_OBJECT_FAILED", "Could not create Worker job object.");
      return;
    }
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION job_information{};
    job_information.BasicLimitInformation.LimitFlags =
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!SetInformationJobObject(job_handle_, JobObjectExtendedLimitInformation,
                                 &job_information,
                                 sizeof(job_information)) ||
        !AssignProcessToJobObject(job_handle_, process_handle_)) {
      CloseHandle(process_info.hThread);
      ReleaseHandles(true);
      result->Error("JOB_OBJECT_FAILED", "Could not attach Worker job object.");
      return;
    }
  }

  ResumeThread(process_info.hThread);
  CloseHandle(process_info.hThread);
  result->Success(flutter::EncodableValue(true));
}

}  // namespace bixolon_scanner_sdk
