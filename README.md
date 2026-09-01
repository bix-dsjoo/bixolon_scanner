# BIXOLON Bakery AI Scanner

여러 상품이 있는 JPEG/PNG 한 장을 판정하는 Windows 시스템입니다. ONNX Runtime Worker,
PyTorch 학습·평가 도구와 Flutter 작업자 앱을 한 저장소에서 관리합니다.

## 현재 버전

현재 배포 가능한 실행 조합은 `0.1.12` 하나입니다.

| 구성 | 버전 |
|---|---|
| Python 패키지·Worker | `0.1.12` |
| Detector·Embedder·판정 정책 | `0.1.12` |
| Store Catalog | `0.1.12` |
| Flutter 앱 | `0.1.12+15` |
| 사용자 표시·Windows ProductVersion | `0.1.12` |

기준 설정은 [`configs/versions/0.1.12.json`](configs/versions/0.1.12.json)입니다. 모델 graph·weight와
Catalog payload는 고정하고 Runtime·Catalog metadata의 공개 실행 버전만 하나의 제품 버전으로
맞춥니다. 과거 운영 `0.1.11` 계약은 `configs/archive`와 `docs/archive`에 보존합니다.

`0.1.12`는 외부 pretrained weight 없이 프로젝트 데이터로 학습한 class-aware SSDLite320
MobileNetV3-Large를 1차 분류기로 사용합니다. 검증된 class는 Detector 결과로 직접 판정하고,
혼동·신규·저신뢰·동일 class 중복 ROI만 DINOv3로 분류합니다. `0.1.11`의 모델 graph·weight,
Catalog payload, Flutter 촬영 작업공간과 공개 API는 유지하며 Windows OpenVINO CPU detector + Intel
GPU classifier, GPU 초기화 실패 시의 명시적 CPU fallback을 사용합니다.

## 판정 계약

Worker는 이미지마다 다음 중 정확히 하나를 반환합니다.

- `SEGMENTATION`: 하나 이상의 객체와 객체별 `APPROVED`, `UNKNOWN`+Top-3 또는
  `SEGMENT_RECAPTURE`
- `IMAGE_RECAPTURE`: detector hard gate가 이미지 전체 재촬영을 요구함
- `ERROR`: 입력, 구성, 모델 또는 시스템 오류

`ERROR`를 재촬영으로 변환하지 않습니다. Detector 조기 종료로 classifier를 실행하지 않은 경우
classifier·Catalog 계열 버전은 `null`입니다. 나머지 공개 non-null 버전은 모두 `0.1.12`입니다.
공개 필드와 판정 순서는 [API 계약](docs/contracts/api.md)을 따릅니다.

현재 Runtime은 class-aware Detector가 이미 SKU를 예측한다는 운영 조건을 이용합니다. 검증된
class와 detector score `0.98` 이상인 단독 ROI는 Detector 결과를 바로
`APPROVED`로 사용하고, 혼동 이력이 있는 class·신규 class·저신뢰·동일 class 중복 ROI만 DINOv3
Classifier로 보냅니다. DINOv3 weight를 freeze하는 것은 학습 update만 막을 뿐 ONNX 추론량을 줄이지
않으므로, embedder graph를 바꾸지 않고 호출 ROI를 약 35%로 줄입니다. 415장과 운영
69장, multi-object 300장에서 활성 결과의 status·prediction·Top-3를 유지했고 개발 PC OpenVINO CPU
full-path p95는 각각 `143.532→99.139ms`, `79.618→37.102ms`, `132.087→96.934ms`였습니다. 고정
415장 1,914개와 multi300 1,410개 모두 FP/FN·오승인·Top-3 실패·`ERROR`가 0입니다. 설계와 증빙은
[Detector-first 선택 DINO 실험](docs/experiments/detector-primary-selective-dino-0.1.11.md)에
기록합니다.

상품 class를 Detector에서 분리하는 `class_agnostic` 실험 경로도 지원합니다. 이 모드의 Detector는
`bread/object` 한 class만 출력하고 SKU 수와 Catalog label 수는 독립적이므로, 일반적인 새 빵은
Catalog와 Classifier support만 추가하고 Detector를 다시 학습하지 않습니다. Detector class 합의는
사용하지 않으며, 극단적인 ROI 형태의 선택적 224 재검사, 해상도 간 승인 안정성 검사와
`UNKNOWN` Top-3 융합으로 안전성을 보완합니다. 이 후보는 활성 `0.1.12` 배포물이 아니며
실험 결과와 N100 실측 한계는
[class-agnostic Detector 실험](docs/experiments/class-agnostic-detector-0.1.11.md)에 기록합니다.
매장별로 Detector와 Classifier를 함께 갱신할 수 있다는 운영 조건에서는 정확도와 실제 N100 실측이
더 좋은 현재 class-aware `0.1.12`를 유지합니다. 이 경우 새 SKU마다 SSDLite를 재학습·ONNX export하고
Classifier/Catalog와 같은 제품 버전으로 묶어 전체 검증합니다. `.pt` checkpoint는 학습·재현용이며
Worker 배포물에는 ONNX Runtime graph만 포함합니다.

기존 detector, YOLO 계열과 RF-DETR을 사용하지 않는 DINOv3 recall-first challenger도 별도 실험으로
구현돼 있습니다. group-aware 300장 OOF proposal recall은 98.01%였고, class-relative ranker와
count-constrained Catalog 경로의 gate 이전 spatial+Top-3 exact 상한은 99/300장까지 개선됐습니다.
그러나 held-fold label을 사용하지 않은 nested gate는 6장 중 1건, segment verifier를 추가한 gate는
11장 중 4건의 오류가 발생해 둘 다 기각했습니다. 목표였던 오류 0건과 accepted coverage 10% 이상을
동시에 충족한 후보는 없습니다. 중단된 center-heatmap 실험도 최종 OOF artifact가 없으므로 결과로
간주하지 않습니다. 이 adaptive cascade는 활성 `0.1.12`에 반영하지 않았습니다. 설계, 완료·폐기·중단
결과와 재현 경로는 [DINOv3 adaptive cascade 실험](docs/experiments/adaptive-cascade-dinov3.md)에
기록돼 있습니다.

Runtime의 선택적 classifier verifier 계약은
`unknown_recapture_on_dual_verifier_rejection`을 선택적으로 지원합니다. 기본값은 `false`이며 현재
활성 `0.1.12` package는 이를 활성화하지 않습니다. 별도 실험 package에서만 `true`로 설정하면,
승인 차단 ROI의 회전·독립 verifier가 모두 품질 실패를 반환할 때 기존 공통
`SEGMENT_RECAPTURE` 경로를 사용합니다.

## Windows 배포물 만들기

Python 3.11, Flutter stable, Visual Studio Windows C++ build tools와 Inno Setup 6이 필요합니다.

```powershell
.\scripts\build_app.ps1 -Version 0.1.12
.\scripts\build_n100_gpu_test.ps1 -Version 0.1.12 -ReuseBuildEnvironment -Force
.\scripts\build_windows_installer.ps1 -Version 0.1.12 -Force
```

첫 명령은 source manifest와 평가 증빙의 고정 해시를 검증하고 CUDA 포함 Flutter 앱 번들을
`artifacts/versions/0.1.12/bixolon-bakery-ai-scanner-0.1.12`에 만듭니다. 두 번째 명령은 Intel GPU
plugin과 CPU fallback을 포함한 OpenVINO Worker를 구성하며, N100 이름은 장비 진단 패키지에만
사용됩니다. 마지막 명령은 일반 Windows 제품 배포물을 생성합니다.

- `artifacts/installers/0.1.12/BixolonBakeryAIScanner-0.1.12-Setup.exe`
- `artifacts/installers/0.1.12/BixolonBakeryAIScanner-0.1.12-Worker.zip`
- 각 배포물의 `.sha256`과 `installer-manifest.json`

설치 PC에는 Python, Flutter 또는 CUDA가 필요하지 않습니다. Intel GPU가 없거나 초기화에 실패하면
Worker가 경고를 기록하고 CPU classifier로 시작합니다. `/health/ready`의 `provider`가
`openvino+openvino_gpu`이면 Intel GPU 구성, `openvino`이면 CPU fallback입니다.

제품명, 설치 표시명과 배포 파일명은 `BIXOLON Bakery AI Scanner`를 사용합니다. `N100`은 성능
측정의 장비명과 provenance에서만 사용하며 프로그램명이나 배포 target 이름으로 사용하지 않습니다.

같은 모델 binary로 모든 ROI에 DINOv3를 실행한 실제 N100 100장 진단 p95는 CPU-only
`676.568ms`, CPU detector+iGPU embedder `469.471ms`였습니다. 동일 SHA-256 100장에서 선택 정책은
410개 ROI 중 143개만 DINOv3를 실행했고 개발 PC p95도 `97.644→72.857ms`로 감소했습니다. 후보
자체 N100 직접 측정은 아니므로 [보수적 판정](docs/diagnostics/n100-detector-primary-0.1.12-assessment.json)에
그 한계를 명시하며 SLA나 독립 일반화 성능 인증으로 표현하지 않습니다.

Setup EXE는 Authenticode 서명이 없습니다. `.sha256`은 전송 중 손상을 확인하지만 발행자 진위를
인증하지 않습니다. 설치·제거와 데이터 보존 범위는
[Windows 설치 안내](installer/windows/INSTALL-KO.txt)에 정리돼 있습니다.

### CPU Worker 전달 패키지

CPU 전용 Flutter 개발자 전달 패키지는 다음 명령으로 별도 생성할 수 있습니다.

```powershell
.\scripts\build_worker_handoff.ps1 -Version 0.1.12
```

결과는 `artifacts/handoff/0.1.12/bixolon-worker-0.1.12-windows-x64-openvino.zip`입니다. 고정 N100
측정 스크립트는 제품명이 아니라 하드웨어 진단 도구로만 포함됩니다.

준비된 metadata와 번들을 다시 검증하려면 다음 명령을 사용합니다.

```powershell
bixolon bundle verify --config configs/versions/0.1.12.json
```

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
configs/versions/               활성 제품 0.1.12 설정 하나
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
