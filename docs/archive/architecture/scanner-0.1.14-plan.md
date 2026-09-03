# BIXOLON Bakery AI Scanner 0.1.14 최소 변경 설계

## 결론

0.1.14는 새로운 추론 기능을 추가하지 않는다. 0.1.13의 판정 구조를 그대로 유지하고 Windows
실행 프로필만 다음과 같이 고정한다.

```text
Detector                   OpenVINO CPU
Classifier·detail·verifier OpenVINO Intel GPU
GPU 시작 실패              OpenVINO CPU fallback
CPU fallback 실패          Worker 시작 실패
```

제품 버전은 `0.1.14`, Flutter 내부 빌드는 `0.1.14+17`로 맞춘다.

## 유지하는 추론 구조

```text
1-class SSDLite320
    ↓ 정상 ROI 전체
ConvNeXt-Tiny 192 primary
    ↓ 전역 위험 ROI만
ConvNeXt-Tiny 224 detail
    ↓ 전역 ambiguity ROI만
Frozen ViT-B/16 160
    ↓
단일 Safety Arbiter
```

매장명, `store_id`, SKU와 Detector class에 따른 routing을 만들지 않는다. 모델 graph·weight,
192/224/160 입력, 전처리, threshold, Catalog 판정과 공개 scan 응답도 바꾸지 않는다.

## 0.1.14 변경 사항

### 1. 실행 프로필 고정

Windows launcher는 다음 기존 설정을 사용한다.

```text
BIXOLON_PROVIDER=openvino
BIXOLON_EMBEDDER_PROVIDER=openvino_gpu
BIXOLON_EMBEDDER_FALLBACK_PROVIDER=same
BIXOLON_CPU_DETECTOR_WORKERS=1
BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS=4
BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS=0
```

GPU 초기화와 warm-up에 실패하면 Worker가 classifier를 OpenVINO CPU로 다시 만들고 시작한다.
요청을 처리하는 도중에는 provider를 바꾸거나 GPU·CPU 결과를 섞지 않는다.

현재 `GET /health/ready`의 `provider`를 그대로 사용한다.

- `openvino+openvino_gpu`: CPU detector + Intel GPU classifier
- `openvino`: CPU fallback

새 readiness 필드와 새 scan 응답 필드는 추가하지 않는다.

### 2. N100 실측 의존 제거

N100 device matrix를 0.1.14 설정, 번들 생성과 installer 생성의 필수 입력에서 제거한다.
`N100-GPU-BENCHMARK.ps1`은 필요할 때만 수동으로 실행하는 진단 도구로 남긴다.

고정 프로필의 과거 N100 100장 결과는 CPU p95 `676.568ms`, CPU detector + Intel GPU embedder
p95 `469.471ms`, 의미 mismatch 0이었다. 이 값은 프로필 선택 근거로만 보존하며 0.1.14 SLA나
현재 성능 보증으로 재표현하지 않는다.

### 3. 단일 제품 버전 반영

- Python·Worker·Detector·Embedder·Detector policy·Classifier policy·Catalog: `0.1.14`
- Flutter: `0.1.14+17`
- 사용자 표시와 Windows ProductVersion: `0.1.14`
- 기존 Runtime/Catalog binary payload는 바꾸지 않고 source SHA-256을 유지
- 과거 `0.1.13` 설정과 문서는 archive에 보존

## 여러 매장 적용

모든 매장은 같은 Worker, Runtime, 실행 순서와 threshold를 사용한다. 설치 한 건에는 Catalog 하나만
넣으며 매장에 맞는 Catalog 디렉터리를 패키징한다. Catalog 선택이 추론 routing을 바꾸지는 않는다.

새 매장 이미지는 별도 shadow 진단으로 평가하되, 특정 매장만 threshold를 조정하지 않는다. 공통
정책을 바꿔야 한다면 모든 매장 회귀 후 다음 patch 버전에서 함께 변경한다.

## 검증 범위

0.1.14에서는 다음만 확인한다.

1. 415장, 운영 69장, multi-object 300장의 기존 판정 회귀
2. GPU 초기화 성공 시 `/health/ready.provider=openvino+openvino_gpu`
3. GPU 초기화 실패 시 `/health/ready.provider=openvino`
4. CPU fallback도 실패하면 Worker 시작 실패
5. Runtime/Catalog checksum과 모든 공개 version 일치
6. Windows build와 packaged Worker smoke
7. `ruff`, Python 테스트, `flutter analyze`, Flutter 테스트, `git diff --check`

N100 직접 성능 측정은 필수 검증에 포함하지 않는다.

## 이번 버전에서 하지 않는 것

- 새 모델, INT8, primary 160 또는 전수 ViT
- 0.1.12의 Detector SKU 직접 승인
- 새 API 필드와 execution fingerprint
- 별도 Worker 로그 시스템
- 실행 중 GPU→CPU 자동 전환이나 Worker 자동 재시작
- Catalog hot-swap
- 매장별·SKU별 threshold와 routing

## 구현 순서

1. installer의 N100 device matrix 필수 의존을 제거한다.
2. CPU+GPU 기본값과 CPU fallback 설정을 그대로 고정한다.
3. 제품과 앱 버전을 `0.1.14`·`0.1.14+17`로 맞춘다.
4. 기존 API·판정 회귀와 fallback 테스트를 실행한다.
5. 0.1.14 번들을 실제로 만들고 checksum과 packaged smoke를 확인한다.
