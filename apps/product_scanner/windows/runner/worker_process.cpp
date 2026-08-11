#include "worker_process.h"

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr wchar_t kAutoStartEnvironmentName[] =
    L"SCANNER_AUTO_START_WORKER";
constexpr wchar_t kPackageEnvironmentName[] = L"BIXOLON_PACKAGE_DIR";
constexpr wchar_t kProviderEnvironmentName[] = L"BIXOLON_PROVIDER";
constexpr wchar_t kCudaDllEnvironmentName[] = L"BIXOLON_CUDA_DLL_DIR";

std::optional<std::wstring> ReadEnvironmentVariable(const wchar_t* name) {
  const DWORD required = ::GetEnvironmentVariableW(name, nullptr, 0);
  if (required == 0) {
    return std::nullopt;
  }

  std::vector<wchar_t> value(required);
  if (::GetEnvironmentVariableW(name, value.data(), required) == 0) {
    return std::nullopt;
  }
  return std::wstring(value.data());
}

void RestoreEnvironmentVariable(
    const wchar_t* name,
    const std::optional<std::wstring>& value) {
  ::SetEnvironmentVariableW(name, value ? value->c_str() : nullptr);
}

std::filesystem::path ExecutableDirectory() {
  std::vector<wchar_t> buffer(MAX_PATH);
  while (true) {
    const DWORD length = ::GetModuleFileNameW(
        nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0) {
      return {};
    }
    if (length < static_cast<DWORD>(buffer.size())) {
      return std::filesystem::path(buffer.data()).parent_path();
    }
    buffer.resize(buffer.size() * 2);
  }
}

std::filesystem::path FindWorkerExecutable(
    const std::filesystem::path& executable_directory) {
  const auto bundled = executable_directory / L"worker" /
                       L"bixolon-worker.exe";
  if (std::filesystem::is_regular_file(bundled)) {
    return bundled;
  }

  const DWORD required =
      ::SearchPathW(nullptr, L"bixolon-worker.exe", nullptr, 0, nullptr,
                    nullptr);
  if (required == 0) {
    return {};
  }
  std::vector<wchar_t> buffer(required + 1);
  if (::SearchPathW(nullptr, L"bixolon-worker.exe", nullptr,
                    static_cast<DWORD>(buffer.size()), buffer.data(),
                    nullptr) == 0) {
    return {};
  }
  return std::filesystem::path(buffer.data());
}

bool AutoStartEnabled() {
  const auto value = ReadEnvironmentVariable(kAutoStartEnvironmentName);
  return !value || (*value != L"0" && *value != L"false" &&
                    *value != L"FALSE");
}

std::optional<std::filesystem::path> FindBundledCudaRuntime(
    const std::filesystem::path& executable_directory) {
  const auto runtime_directory =
      executable_directory / L"worker" / L"cuda-runtime";
  constexpr const wchar_t* kRequiredRuntimeFiles[] = {
      L"cublas64_13.dll",
      L"cublasLt64_13.dll",
      L"cudart64_13.dll",
      L"cudnn64_9.dll",
      L"cudnn_adv64_9.dll",
      L"cudnn_cnn64_9.dll",
      L"cudnn_engines_precompiled64_9.dll",
      L"cudnn_engines_runtime_compiled64_9.dll",
      L"cudnn_graph64_9.dll",
      L"cudnn_heuristic64_9.dll",
      L"cudnn_ops64_9.dll",
      L"cufft64_12.dll",
      L"zlibwapi.dll",
  };
  for (const auto* filename : kRequiredRuntimeFiles) {
    if (!std::filesystem::is_regular_file(runtime_directory / filename)) {
      return std::nullopt;
    }
  }
  return runtime_directory;
}

}  // namespace

WorkerProcess::~WorkerProcess() {
  Stop();
}

bool WorkerProcess::Start() {
  if (process_ != nullptr || !AutoStartEnabled()) {
    return process_ != nullptr;
  }

  const auto executable_directory = ExecutableDirectory();
  const auto worker_executable = FindWorkerExecutable(executable_directory);
  if (worker_executable.empty()) {
    ::OutputDebugStringW(
        L"BIXOLON Scanner: bixolon-worker.exe was not found.\n");
    return false;
  }

  const auto existing_package =
      ReadEnvironmentVariable(kPackageEnvironmentName);
  const auto bundled_package =
      executable_directory / L"worker" / L"model-package";
  if (!existing_package &&
      !std::filesystem::is_regular_file(bundled_package / L"metadata.json")) {
    ::OutputDebugStringW(
        L"BIXOLON Scanner: bundled Worker model package was not found.\n");
    return false;
  }

  const auto existing_provider =
      ReadEnvironmentVariable(kProviderEnvironmentName);
  const auto existing_cuda_dll_directory =
      ReadEnvironmentVariable(kCudaDllEnvironmentName);
  const auto bundled_cuda_runtime =
      FindBundledCudaRuntime(executable_directory);
  if (!existing_package) {
    ::SetEnvironmentVariableW(kPackageEnvironmentName,
                              bundled_package.c_str());
  }
  if (!existing_cuda_dll_directory && bundled_cuda_runtime) {
    ::SetEnvironmentVariableW(kCudaDllEnvironmentName,
                              bundled_cuda_runtime->c_str());
  }
  if (!existing_provider) {
    // A release that bundles the CUDA runtime is an explicitly GPU-targeted
    // package. Force CUDA so an incomplete deployment fails readiness instead
    // of silently running the full pipeline on CPU at roughly 10x latency.
    ::SetEnvironmentVariableW(
        kProviderEnvironmentName,
        (existing_cuda_dll_directory || bundled_cuda_runtime) ? L"cuda"
                                                               : L"auto");
  }

  job_ = ::CreateJobObjectW(nullptr, nullptr);
  if (job_ == nullptr) {
    RestoreEnvironmentVariable(kPackageEnvironmentName, existing_package);
    RestoreEnvironmentVariable(kProviderEnvironmentName, existing_provider);
    RestoreEnvironmentVariable(kCudaDllEnvironmentName,
                               existing_cuda_dll_directory);
    return false;
  }

  JOBOBJECT_EXTENDED_LIMIT_INFORMATION job_information = {};
  job_information.BasicLimitInformation.LimitFlags =
      JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  if (!::SetInformationJobObject(job_, JobObjectExtendedLimitInformation,
                                 &job_information,
                                 sizeof(job_information))) {
    ::CloseHandle(job_);
    job_ = nullptr;
    RestoreEnvironmentVariable(kPackageEnvironmentName, existing_package);
    RestoreEnvironmentVariable(kProviderEnvironmentName, existing_provider);
    RestoreEnvironmentVariable(kCudaDllEnvironmentName,
                               existing_cuda_dll_directory);
    return false;
  }

  STARTUPINFOW startup_information = {};
  startup_information.cb = sizeof(startup_information);
  startup_information.dwFlags = STARTF_USESHOWWINDOW;
  startup_information.wShowWindow = SW_HIDE;
  PROCESS_INFORMATION process_information = {};
  std::wstring command_line = L"\"" + worker_executable.wstring() + L"\"";

  const BOOL created = ::CreateProcessW(
      worker_executable.c_str(), command_line.data(), nullptr, nullptr, FALSE,
      CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr,
      executable_directory.c_str(), &startup_information,
      &process_information);

  RestoreEnvironmentVariable(kPackageEnvironmentName, existing_package);
  RestoreEnvironmentVariable(kProviderEnvironmentName, existing_provider);
  RestoreEnvironmentVariable(kCudaDllEnvironmentName,
                             existing_cuda_dll_directory);

  if (!created) {
    ::CloseHandle(job_);
    job_ = nullptr;
    ::OutputDebugStringW(
        L"BIXOLON Scanner: failed to start bixolon-worker.exe.\n");
    return false;
  }

  if (!::AssignProcessToJobObject(job_, process_information.hProcess)) {
    ::TerminateProcess(process_information.hProcess, EXIT_FAILURE);
    ::CloseHandle(process_information.hThread);
    ::CloseHandle(process_information.hProcess);
    ::CloseHandle(job_);
    job_ = nullptr;
    ::OutputDebugStringW(
        L"BIXOLON Scanner: failed to supervise bixolon-worker.exe.\n");
    return false;
  }

  process_ = process_information.hProcess;
  ::ResumeThread(process_information.hThread);
  ::CloseHandle(process_information.hThread);
  return true;
}

void WorkerProcess::Stop() {
  if (job_ != nullptr) {
    ::TerminateJobObject(job_, EXIT_SUCCESS);
  }
  if (process_ != nullptr) {
    ::CloseHandle(process_);
    process_ = nullptr;
  }
  if (job_ != nullptr) {
    ::CloseHandle(job_);
    job_ = nullptr;
  }
}
