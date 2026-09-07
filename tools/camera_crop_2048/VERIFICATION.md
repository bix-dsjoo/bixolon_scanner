# Camera Crop 2048 1.0.0 검증

검증일: 2026-09-07, Windows x64 / Python 3.11 / PyInstaller 6.11.1 / Inno Setup 6.

## 배포 파일

- 설치 파일: `artifacts/camera-crop-2048/1.0.0/CameraCrop2048-1.0.0-Setup.exe`
- 크기: 56,681,035 bytes.
- SHA-256: `95412cf1fafb74bcee5df4ccc6eb93ae2b006b80aaee26c270e766d67b18cee6`
- 실제 빌드: `artifacts/camera-crop-2048/1.0.0/build-20260907-134504`.
- 별도 유틸리티이며 Scanner / Worker / Runtime / Catalog를 포함하거나 수정하지 않습니다.

## 통과한 검증

- PyInstaller EXE 생성, Inno Setup 생성과 파일 버전 `1.0.0.0` 확인.
- Setup 실제 설치 종료 코드 0, 설치된 manifest 대상 1,112개 파일 SHA-256 전부 일치.
- 설치된 EXE 자체로 Tk 초기화, 3264×2448 → 중앙 2048×2048, PNG 재디코딩 픽셀 일치 검증.
- 앞서 현재 카메라에서 촬영한 3264×2448 사진을 사용하는 UI 검증: `(608, 200)` 중앙 크롭,
  미리보기 전체 표시, 실제 촬영 버튼 경로의 PNG 저장과 원본 픽셀 일치.
- 설치본 UI 실행 및 카메라 연결 실패 안내·촬영 비활성화 확인.
- 촬영 도구 Python 테스트 12개 통과. 크롭 위치, 작은 입력 거부, 세로·홀수 크기,
  PNG 색상·픽셀 보존, 파일명 중복 방지, 저장 오류, 카메라 오류 후 해제를 포함합니다.
- `ruff check`, `ruff format --check`, `git diff --check` 통과.
- Product Scanner / Scanner Lite `flutter analyze` 통과.
- Product Scanner 전체 Flutter 테스트 193개, Lite 12개 통과.

## 한계와 별도 실패

- 패키징 검증 시 Windows의 현재 Camera/Image 장치 목록에 카메라가 없었으며
  DirectShow 카메라 0 열기도 실패했습니다. **최종 설치본의 실시간 USB 촬영은 미검증**입니다.
  저장된 실사진과 합성 프레임을 사용한 크롭·화면·저장 검증을 실시간 검증으로 취급하지 않습니다.
- 다른 깨끗한 PC에서의 설치는 미검증입니다. Python, Tcl/Tk, OpenCV, NumPy 및 필요한
  VCRUNTIME DLL은 자체 포함하며 설치된 EXE 검증은 이 PC에서 수행했습니다.
- 전체 Python 테스트는 **1,080 통과 / 1 실패**입니다. 실패한 기존 Worker 테스트는
  `tests/test_inference_resilience.py::test_slow_decode_does_not_block_health_or_decode_waiting_requests`이며
  `/health/ready` 예상 503에 실제 200을 반환했습니다. 단독 재실행에서도 재현됐습니다.
  이 유틸리티는 Worker를 import하거나 패키징하지 않으며 해당 Worker 코드는 이번 작업에서
  수정하지 않았습니다. 전체 저장소 검사가 모두 통과했다고 표시하지 않습니다.

로그: `tmp/camera-crop2048-python-tests.log`, `tmp/camera-crop2048-build-final.log`,
`tmp/camera-crop2048-ui-smoke/report.json`, 실제 빌드 폴더의 `package-smoke/verification.json`,
`installed-smoke/verification.json`, `install.log`.
