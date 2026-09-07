# Camera Overlay Lab

기존 Bixolon Scanner 서비스와 분리된 Windows 카메라 구도 확인 도구입니다. 카메라 원본 해상도에
맞춰 `640×640`, `768×768`, `960×960`, `1024×1024` 정사각형을 라이브 미리보기 위에
표시합니다.

```powershell
cd tools/camera_overlay_lab
flutter run -d windows
```

오른쪽 패널에서 각 가이드를 켜고 끌 수 있으며, 크롭 기준을 중앙 또는 하단으로 바꿀 수 있습니다.
이 앱은 미리보기만 제공하며 상품 스캐너의 캡처·Worker·모델 설정을 변경하지 않습니다.
