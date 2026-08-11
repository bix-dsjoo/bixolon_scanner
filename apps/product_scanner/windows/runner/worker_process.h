#ifndef RUNNER_WORKER_PROCESS_H_
#define RUNNER_WORKER_PROCESS_H_

#include <windows.h>

class WorkerProcess {
 public:
  WorkerProcess() = default;
  ~WorkerProcess();

  WorkerProcess(const WorkerProcess&) = delete;
  WorkerProcess& operator=(const WorkerProcess&) = delete;

  bool Start();
  void Stop();

 private:
  HANDLE job_ = nullptr;
  HANDLE process_ = nullptr;
};

#endif  // RUNNER_WORKER_PROCESS_H_
