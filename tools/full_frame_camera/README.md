# 전체 화면 카메라

Windows에서 USB 카메라의 전체 프레임을 보고 PNG로 저장하는 별도 Python 촬영 도구입니다.
Tkinter / OpenCV를 사용하며 Worker나 모델을 실행하지 않습니다.

이 PC에서는 `start.cmd`를 더블클릭해 실행할 수 있습니다. 다른 PC에서는 먼저 아래 의존성을 설치합니다.

```powershell
python -m pip install -r tools/full_frame_camera/requirements.txt
powershell -ExecutionPolicy Bypass -File tools/full_frame_camera/start.ps1
```

- 기본 카메라 번호는 `0`, 요청 해상도는 `3264×2448`입니다.
- 미리보기는 비율을 유지해 창 안에 전체 영역을 표시합니다. 남는 부분은 여백입니다.
- 원본 PNG에는 수신 프레임 전체를 저장합니다. 자르기, 리사이즈, 좌우 반전, 표시 문구를 넣지 않습니다.
- `원본 PNG 촬영` 또는 `Space`로 현재 표시 중인 최신 프레임을 저장합니다.
- `F11`로 전체 화면을 전환하고 `Esc`로 돌아옵니다.
- 기본 저장 위치는 사용자 홈의 `Pictures/FullFrameCamera`이며 앱에서 변경할 수 있습니다.
- 해상도 변경은 `연결 해제` → 해상도 선택 → `연결` 순서입니다.
- 실제 수신 해상도를 표시합니다. 장치가 요청을 지원하지 않으면 다른 해상도가 표시될 수 있습니다.
- 이 도구는 DirectShow 영상 스트림을 사용합니다. Windows 카메라 앱의 별도 정지사진 모드와
  지원 해상도 / 처리 결과가 다를 수 있으며, 센서가 제공하지 않는 화각을 복원하지는 않습니다.
- 장치 연결 오류가 발생하면 다른 카메라 앱을 닫고 재연결하세요. 2초 이상 지난 영상은 저장하지 않습니다.
- 사진은 사용자가 촬영했을 때만 저장하며 자동 삭제하지 않습니다. 삭제는 저장 폴더에서 직접 합니다.

다른 카메라나 저장 위치를 지정할 수도 있습니다.

```powershell
./tools/full_frame_camera/start.ps1 -Camera 0 -OutputDirectory C:\captures
```

이 도구는 제품 배포 번들에 포함하지 않는 로컬 촬영 유틸리티입니다.

2026-09-07 연결된 ZHWY Camera에서 `3264×2448` 수신, 창 크기 변경 시 전체 프레임 표시,
PNG 저장 후 수신 프레임과 모든 픽셀 일치를 확인했습니다.

검증: `ruff check`, `ruff format --check`, `git diff --check` 통과. 전체 Python 테스트
1,075개와 추가한 연결 해제 테스트를 포함한 촬영 도구 테스트 7개가 통과했습니다.
Product Scanner / Scanner Lite의 `flutter analyze`와 전체 테스트(각각 188개 / 12개)도 통과했습니다.
