# BIXOLON Bakery AI Scanner

여러 상품이 있는 JPEG/PNG 한 장을 판정하는 Windows 시스템입니다. ONNX Runtime Worker,
PyTorch 학습·평가 도구와 Flutter 작업자 앱을 한 저장소에서 관리합니다.

## Scanner Lite

별도 제품 **BIXOLON Bakery AI Scanner Lite**는 `0.1.17` CPU Worker를 그대로 사용하며
카메라/파일 입력, 읽기 전용 결과, 자동 메타데이터 로그·내보내기만 제공한다.
기존 앱의 화면이나 복잡한 기능을 숨겨 재사용하지 않는다. 디자인 token/theme·로고·폰트만 공유한다.
Lite는 읽기 전용 검출 박스를 표시하고 카메라 상단 20%를 제거한 중앙 정사각형을 2048×2048로 변환하고 좌우 반전한다.
사용자 요청에 따라 Top-3 이름을 읽기 전용으로 표시하고, 실행 로그에서도 이미지와 박스를 복원한다.
입력 이미지는 별도 파일로 30일 보관하며 메타데이터·박스·후보 이름과 함께 ZIP으로 내보낸다.
confidence는 표시하거나 기록하지 않으며 재촬영 상태는 빨간색으로 표시한다.
`apps/bakery_scanner_lite`가 독립 구현의 canonical 경로이며 기존 앱과 설치 ID·폴더가 다르다.
빌드는 `scripts/build_lite.ps1`, 사용·로그 계약은
[`Lite README`](apps/bakery_scanner_lite/README.md)를 참조한다.

## 전체 프레임 촬영 도구

중앙 2048×2048만 표시·저장하는 독립 Windows 설치 앱은
[`Camera Crop 2048`](tools/camera_crop_2048/README.md)을 참조합니다. Python 없이 설치할 수 있으며
Scanner·Worker 번들과 별도의 로컬 촬영 유틸리티입니다.

일반 Scanner 앱의 카메라 입력은 원본 중앙 `1080×1080`을 크롭해 분석·빵 촬영에 사용합니다.
1920×1080 원본은 좌우 각 420px을 제거합니다. 짧은 변이 1080px 미만인 경우에는 확대 없이
가능한 최대 중앙 정사각형을 사용합니다. 파일 입력에는 이 크롭을 적용하지 않습니다.

카메라 전체 영역을 비율대로 미리 보고 수신 해상도 그대로 PNG를 저장하려면
[`전체 화면 카메라`](tools/full_frame_camera/README.md)를 사용합니다.
`powershell -ExecutionPolicy Bypass -File tools/full_frame_camera/start.ps1`로 실행하며,
기본 요청 해상도는 `3264×2448`입니다. 자르기 없이 저장한 뒤 Python에서 별도로 가공할 수 있습니다.

## 현재 버전

현재 배포 가능한 실행 조합은 `0.1.17`입니다. Python·Worker·Detector·Embedder·판정 정책·Catalog·Windows ProductVersion은 `0.1.17`, 두 Flutter 앱은 `0.1.17+20`입니다.
기준은 [configs/versions/0.1.17.json](configs/versions/0.1.17.json)입니다.

사용자가 지정한 현재 1등 `ssdlite-margin-dense-20260908`을 배포합니다. 정정본 단일 200장·다중 100장·배경 2장과 649개 객체를 사용하며, 겹친 객체를 하나로 검출한 ROI의 승인을 막는 head를 추가했습니다.
같은 실물 개발 진단은 올바른 승인 645/649(99.38%), 오승인 0건, 실제 다중 이미지 CPU HTTP full-path p95 최악 159.32ms입니다. 최종 300장 성적으로 표현하지 않습니다.

최종 배포 EXE의 300장 평가는 CPU·CUDA 각각 3회 모두 **1,352/1,410(95.89%)·오승인 4건**입니다. CPU HTTP p95는 **318.42 / 304.29 / 316.19ms**로 세 목표에 미달했습니다. 평가 후 모델·임계값은 변경하지 않았습니다.

[0.1.16 대비 변경과 실행 장치 지원](docs/architecture/scanner-0.1.17.md)에 학습 원본·모델·CPU 설정·성능 범위를 정리했습니다.
배포본은 CPU 설치본·Lite·SDK와 CUDA portable 앱으로 구성하며 같은 ONNX·Catalog·정책을 사용합니다.

## 판정 계약

정정본 학습·비교 과정은 [three_bakery 실행 기록](docs/experiments/three-bakery-training.md)을 참조합니다.
정확도 목표는 전체 GT 객체 기준 99% 이상, 오승인 0건, CPU HTTP p95 300ms 이하입니다.
최종 300장·1,410개 GT에서는 1,396개 이상 올바른 APPROVED를 요구합니다. 누락·UNKNOWN·재촬영·ERROR도 분모에 남습니다.
현재 1등 후보는 사용자 요청에 따라 0.1.17로 복사했으며, 추가 seed 실험은 별도로 보존합니다.

Worker는 이미지마다 다음 중 정확히 하나를 반환합니다.

- `SEGMENTATION`: 하나 이상의 객체와 객체별 `APPROVED`, `UNKNOWN`+Top-3 또는
  `SEGMENT_RECAPTURE`
- `IMAGE_RECAPTURE`: detector hard gate가 이미지 전체 재촬영을 요구함
- `ERROR`: 입력, 구성, 모델 또는 시스템 오류

`ERROR`를 재촬영으로 변환하지 않습니다. Detector 조기 종료로 classifier를 실행하지 않은 경우
classifier·Catalog 계열 버전은 `null`입니다. 나머지 공개 non-null 버전은 모두 `0.1.17`입니다.
공개 필드와 판정 순서는 [API 계약](docs/contracts/api.md)을 따릅니다.

소스 Worker는 Detector/Classifier의 NaN·Inf 출력을 `MODEL_EXECUTION_FAILED` 5xx `ERROR`로
거부합니다. 디코딩과 추론은 단일 executor에서 수행하고, 대기 요청은 슬롯 획득 전에 이미지를
디코딩하지 않습니다. 실행 deadline을 넘긴 작업이 남아 있으면 readiness는 503이며, 작업 종료 후
복구합니다. 응답 시간이 초과돼도 실행 중인 ONNX 작업의 슬롯을 임의로 해제하지 않습니다.

이전 0.1.16의 220장 학습·운영 진단은 [이전 배포 기록](docs/experiments/next-worker-0.1.16.md)에 보존합니다.
이번 데이터와 이전 데이터의 성능은 동일 조건 비교가 아닙니다.

기존 detector, YOLO 계열과 RF-DETR을 사용하지 않는 DINOv3 recall-first challenger도 별도 실험으로
구현돼 있습니다. group-aware 300장 OOF proposal recall은 98.01%였고, class-relative ranker와
count-constrained Catalog 경로의 gate 이전 spatial+Top-3 exact 상한은 99/300장까지 개선됐습니다.
그러나 held-fold label을 사용하지 않은 nested gate는 6장 중 1건, segment verifier를 추가한 gate는
11장 중 4건의 오류가 발생해 둘 다 기각했습니다. 목표였던 오류 0건과 accepted coverage 10% 이상을
동시에 충족한 후보는 없습니다. 중단된 center-heatmap 실험도 최종 OOF artifact가 없으므로 결과로
간주하지 않습니다. 이 adaptive cascade는 활성 `0.1.17`에 반영하지 않았습니다. 설계, 완료·폐기·중단
결과와 재현 경로는 [DINOv3 adaptive cascade 실험](docs/experiments/adaptive-cascade-dinov3.md)에
기록돼 있습니다.

Runtime은 전수 verifier 호환 필드를 읽을 수 있지만 활성 0.1.17은
`verify_all_approved_candidates=false`, `unknown_recapture_on_any_verifier_rejection=false`를
고정합니다. 단일 verifier 품질 실패를 재촬영으로 확대하지 않으며 최종 상태는 항상
`DecisionPipeline`의 단일 정책에서 결정합니다.

## Windows 배포물 만들기

Python 3.11, Flutter stable, Visual Studio Windows C++ build tools와 Inno Setup 6을 사용합니다.

```powershell
.\scripts\build_app.ps1 -Version 0.1.17 -PythonExecutable <CUDA-lock Python>
.\scripts\build_worker.ps1 -PythonExecutable <CPU-lock Python> -OutputDirectory artifacts/versions/0.1.17/cpu-worker-build
.\scripts\build_windows_installer.ps1 -Version 0.1.17 -VcRedistPath <vc_redist.x64.exe>
.\scripts\build_external_sdk.ps1 -ModelVersion 0.1.17 -SdkVersion 1.1.1 -Package All
# CPU packaged smoke 후 실행
.\scripts\build_lite.ps1 -Version 0.1.17
bixolon bundle verify --config configs/versions/0.1.17.json
```

최종 전달 폴더는 `artifacts/distributions/0.1.17`입니다. 전체 앱·Lite의 설치본과 portable ZIP,
CPU Worker ZIP, CUDA portable ZIP, SDK Core 1.1.1·Store Model 0.1.17 ZIP 및 SHA-256을 모읍니다.
개별 원본 산출물은 `artifacts/installers`, `artifacts/lite`, `artifacts/versions`, `artifacts/external-sdk`, `artifacts/store-models`에 보존합니다.

CPU 설치본·Lite에는 Python, Flutter, CUDA, OpenVINO 별도 설치가 필요하지 않습니다.
CUDA portable은 NVIDIA CUDA 실행용입니다. 기존 CPU+GPU 혼합 분류와 CPU fallback 코드는
남아 있지만 이번 기본 배포에서는 활성화하지 않았습니다.

Setup EXE는 Authenticode 서명이 없습니다. SHA-256은 손상·변경을 탐지하지만 발행자 진위를
인증하지 않습니다. [Windows 설치 안내](installer/windows/INSTALL-KO.txt)를 함께 제공합니다.

## 빵 원본 촬영 관리

Flutter 앱 상단의 `빵 촬영` 작업공간에서 빵을 최대 20종까지 등록하고 종류별 촬영 진행률을 관리할
수 있습니다. 각 빵은 3×3 기준 `왼쪽 상단`, `오른쪽 상단`, `중앙`, `왼쪽 하단`, `오른쪽 하단`을
`정상 면`과 `뒤집은 면`에서 각각 촬영해 총 10장을
완성합니다. 촬영 직후 사진을 확인해 사용하거나 다시 찍을 수 있고, 저장된 자세만 선택해 재촬영하거나
삭제할 수 있습니다.

기본 저장 위치는 사용자 문서의 `BIXOLON Bakery AI Scanner/single_objects`이며 화면의 `저장 위치`에서
다른 폴더를 선택할 수 있습니다. 개발 실행에서는 다음처럼 초기 폴더를 고정할 수도 있습니다.

```powershell
flutter run -d windows `
  --dart-define=BREAD_CAPTURE_ROOT=C:\workspace\bixolon_scanner\datasets\bread_dataset\single_objects
```

폴더와 파일명은 기존 `single_objects` 합성기가 직접 읽는
`bread_01_<이름>/bread_01_normal_ground_30_dir_01.jpg` 계약을 유지합니다. 화면의 5개 위치는
`ground_30_dir_01`, `ground_30_dir_02`, `vertical`, `ground_30_dir_03`, `ground_30_dir_04` 순으로
저장되므로 기존 데이터와 함께 사용할 수 있습니다. 기존 계약 폴더를 선택하면 등록 정보가 없어도 자동으로 빵과
촬영 완료 상태를 불러옵니다.
빵 삭제는 해당 종류의 촬영 사진도 함께 지우므로 앱에서 확인한 뒤에만 실행됩니다.

## 실제 트레이 기반 다중 빵 합성

COCO 어노테이션이 있는 단일 빵 사진과 같은 촬영 조건의 빈 트레이 사진으로 detector 개발용
다중 객체 이미지를 만들 수 있습니다. 합성기는 어노테이션 bbox 안에서 원본과 빈 배경의 차이를
계산해 빵 마스크를 추출하고, 밝기·대비·채도·좌우 반전·회전·크기와 투영 그림자를 변형합니다.
실제 전경 마스크 기준으로 배치 충돌을 제한하므로 donut처럼 내부 구멍이 있는 객체도 빈 영역을
유지합니다.

```powershell
bixolon-compose-operational-scenes `
  --annotations datasets/bread_dataset/operational_collections/2026-08-27/annotations/instances.json `
  --background-image 041_a65820e8740744f8a9984ab467fd7f95.jpg `
  --output-root artifacts/synthetic/operational-2026-08-27 `
  --image-count 100 --minimum-objects 2 --maximum-objects 4 `
  --minimum-source-scale 0.82 --maximum-source-scale 1.05 --seed 20260831
```

`--source-image <파일명>`을 반복하면 지정한 단일 객체 사진만 사용하고, 생략하면 COCO에서 annotation이
정확히 하나인 사진을 모두 선택합니다. `--background-image`도 반복할 수 있으며, 생략하면 annotation이
없는 사진을 모두 실제 배경 후보로 사용합니다. 출력은 JPEG와 함께 detector용 `instances.json`, 내부
manifest, 원본 SHA-256과 모든 변형값을 담은 `provenance.jsonl`, 재현 설정인 `metadata.json`입니다.
크기는 원본 촬영 프레임에서 빵이 차지하던 비율을 기준으로 유지하고, 객체 수가 늘어나면 밀도에
따라 완만하게 축소합니다. `--minimum-source-scale`과 `--maximum-source-scale`은 이 원본 상대
크기의 배율입니다. 기본 최대 개수는 큰 빵 크기를 유지하기 위해 4개이며, 5~6개 장면이 필요할 때만
`--maximum-objects`를 높입니다.

합성 결과는 자동으로 학습 데이터에 편입되지 않습니다. 운영 회귀나 test로 사용한 원본을 합성에
썼다면 같은 원본·촬영 session에서 파생된 결과도 해당 평가와 분리해야 하며, 독립 성능 증빙으로
사용할 수 없습니다.

20종 각각을 앞·뒤 두 면과 5개 위치에서 촬영한 200장, 빈 트레이 1장으로 조건별 1,500장을 만드는
위치 인지 합성기는 다음 명령을 사용합니다.

```powershell
bixolon-compose-grid-dataset `
  --capture-root D:\bread-grid-captures `
  --output-root artifacts\synthetic\bread-grid-1500-seed-20260831 `
  --seed 20260831
```

정확한 201장 폴더 이름, `600/300/225/150/125/100` scenario 구성, 가림 후 visible bbox와 검수
방법은 [20종 빵 위치 인지 합성 데이터 가이드](docs/training/synthetic-grid-composition.md)를
따릅니다.

## 개발과 검증

```powershell
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest

cd apps\product_scanner
flutter analyze
flutter test
```

정확도, provider parity, latency와 장기 신뢰성 평가는 선택 진단입니다. 결과가 미실행됐거나 목표를
넘었다는 이유로 별도 promotion·waiver·release-lock 수명주기를 만들지 않습니다.

## 저장소 구조

```text
apps/product_scanner/           Flutter Windows 앱
configs/versions/               활성 제품 0.1.16 설정 하나
configs/archive/                과거 제품·실험 설정 원문
docs/evaluation/                활성 제품 평가와 한계
docs/archive/                   과거 계약·평가·판단·가이드
installer/windows/              일반 Windows Setup·Worker 배포 소스
src/bixolon_scanner/            Python canonical 코드
tests/                          계약과 회귀 테스트
artifacts/, datasets/, runs/    Git 밖의 모델·데이터·빌드 산출물
```

구현 경계와 변경 규칙은 [AGENTS.md](AGENTS.md), 문서 목록은
[docs/README.md](docs/README.md), 로컬 산출물 보존 기준은
[repository-layout.md](docs/maintenance/repository-layout.md)에서 확인할 수 있습니다.

## ROI 다중 객체 재촬영 정책 추가 작업

사용자의 2026-09-08 추가 요청에 따라, 기존 192 primary 특징을 공유하는 ROI 객체 수 head를
추가한다. metadata에 함께 설정한 `embedder.multi_object_output_name`과
`quality.multi_object_recapture_threshold`로 활성화한다. 해당 ROI의 다중 객체 확률이 임계값
이상이면 품목 승인 신뢰도와 무관하게 `SEGMENT_RECAPTURE`이고, 다른 정상 ROI는 계속 판정한다.
detail 또는 회전·ViT 합의로 이 재촬영을 다시 승인하지 않는다. 공개 응답은 기존
`SEGMENT_RECAPTURE_REQUIRED`, `prediction=null`, 빈 `top3` 계약을 유지한다.
설정으로 활성화한 head가 없거나 출력이 손상된 경우 모델 오류로 처리한다.
별도 backbone이나 전수 ViT 검증은 추가하지 않는다. 제품 버전은 `0.1.16`을 유지한다.

현재 작업과 이전 성적은 [실험 문서](docs/experiments/three-bakery-training.md)에 구분해 기록한다.
현재 재학습의 사용자 목표는 전체 GT 객체 기준 정답 승인율 99% 이상·오승인 0건·CPU HTTP
p95 300ms 이하다. 이미지 전체 성공률은 참고 진단이며, 누락·UNKNOWN·재촬영 객체도 GT 분모에
남긴다. 목표 설정은 `configs/experiments/bread/three_bakery_objective.json`에 기록한다.
