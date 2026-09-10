# BIXOLON Bakery AI Scanner 0.1.17

현재 1등인 `ssdlite-margin-dense-20260908`을 사용자가 지정해 배포한다.
제품·Worker·Runtime·Catalog는 `0.1.17`, 두 Flutter 앱은 `0.1.17+20`이다.
기준은 [버전 설정](../../configs/versions/0.1.17.json)이며, 실험 모델의 graph·weight·support는
수정하지 않고 실행 구성요소의 버전 metadata만 맞춘다.

## 0.1.16과의 차이

| 항목 | 0.1.16 | 0.1.17 |
|---|---|---|
| 배포 후보 | limited220-surfaces | three_bakery / SSDLite + margin + dense, seed 20260908 |
| 학습 원본 | 단일 200장 + 다중 20장 | 정정된 단일 200장 + 다중 100장 + 배경 2장 |
| 분류 학습 | 이전 220장 기반 | 이번 원본의 649개 객체 crop 전부 |
| 검출 학습 | 이전 220장 기반 | 302개 원본 전부 + 밀집 합성 6,000장 |
| Catalog | limited220 기반 | 정정본 649개에서 새로 고른 200 support, 품목별 10개 |
| 겹친 객체 | 기존 품질·경계 판정 | primary 특징에 다중 객체 head 추가; 0.8 이상 ROI는 SEGMENT_RECAPTURE |
| CPU 설정 | Detector 4 / Embedder 4 threads | Detector 8 / Embedder 12 threads, Worker 1개 |
| CPU ONNX Runtime | 1.28.0 | 1.29.0 |
| CUDA ONNX Runtime | 1.28.0 | 1.28.0 |
| Flutter 내부 build | 19 | 20 |
| 외부 SDK Core | 1.1.0 | 1.1.1; 실행 Worker 변경을 포함, 제품 모델 버전은 0.1.17 |

223번 객체 4의 꽃빵(C07), 284번 객체 3의 와플(C03), 271번 누락 객체 추가를 반영했다.
다른 빵 데이터에서 학습한 checkpoint·Catalog·teacher는 재사용하지 않았다.
일반 사전학습 가중치에서 시작하고, 같은 실물의 원본·파생물 관계와 소비 이력을 보존했다.

## 추론 순서

1. Class-agnostic SSDLite320이 객체 위치와 프레임 품질을 판단한다.
2. 프레임 hard 품질 실패는 분류기를 실행하지 않고 IMAGE_RECAPTURE로 종료한다.
3. 정상 ROI 전체를 ConvNeXt-Tiny 192 한 batch로 처리한다. 같은 특징을 쓰는 다중 객체 head가
   하나로 합쳐진 ROI를 판정하며, 기준 이상이면 그 ROI의 승인을 차단한다.
4. 나머지 ROI 중 전역 위험 조건에 해당하는 것만 224 detail로 보낸다.
5. 경계 승인 후보만 Frozen DINOv3 ViT-B/16 160으로 검증한다. 회전 합의도 유지한다.
6. 최종 상태는 pipeline에서 결정한다. 다중 객체 판정은 SEGMENT_RECAPTURE로 유지하며
   reason code는 기존 SEGMENT_RECAPTURE_REQUIRED를 사용한다.

승인 margin 0.8, detail·verifier 경계 상한 0.85는 그대로다. 품목별 예외와 전수 verifier는 없다.
겹침 판정이 모든 겹침을 탐지한다는 뜻은 아니며, 누락된 객체를 정답 승인으로 세지 않는다.
공개 `/v1/scan` 필드·Top-3·버전 null·ERROR 계약은 유지하고 Worker/runtime은 PyTorch를 쓰지 않는다.

## 실행 장치와 fallback

Windows 설치본·CPU Worker·Lite·외부 SDK는 CPU 전용이다. CUDA portable 앱은
NVIDIA CUDA를 명시적으로 사용하며 관련 DLL을 포함한다. 두 배포본은 같은 ONNX·Catalog·정책을 쓴다.

과거 CPU 검출 + GPU 분류의 혼합 provider와 분류 초기화 실패 시 CPU fallback 코드는 남아 있다.
그러나 이번 기본 배포 설정에서는 활성화하지 않으며, 새 모델의 혼합 실행 지원을 보증하지 않는다.
CUDA 초기화·실행 장애를 조용히 CPU 승인 결과로 바꾸지 않는다. 준비 실패는 readiness 오류,
요청 실행 장애는 ERROR로 반환한다.

## 성능 수치의 범위

현재 후보의 같은 실물 개발 진단은 원본 649개 중 올바른 APPROVED 645개(99.38%), 오승인 0건이다.
실제 다중 100장 완전 성공은 97장으로 참고 집계한다. Intel Core Ultra 9 285K의 CPU HTTP
full-path p95는 실제 다중 100장의 세 반복에서 159.32 / 118.43 / 119.11ms,
합성 양성 320장에서는 79.36 / 78.33 / 78.67ms였다.

이 수치는 독립 validation이나 최종 300장 성적이 아니다. 최종 목표는 전체 GT 객체 1,410개 중
1,396개 이상 올바른 APPROVED, 오승인 0건, CPU HTTP 전체·full-path p95 각각 300ms 이하이다.
최종 평가 접근 전 모델·Catalog·정책·CPU 설정·평가 코드 해시를 고정하며 이후 결과에 맞춰
재학습·임계값 변경을 하지 않는다. 실제 배포 검증 결과는 배포 폴더의 검증 보고서를 따른다.

이전 버전과 이번 후보는 개발 데이터가 달라 승인율이나 지연 개선 배수를 직접 비교하지 않는다.
추가 seed 비교는 별도 실험 기록이며 이 배포물의 선택 모델을 자동 교체하지 않는다.

## 실제 배포 EXE의 최종 평가

고정한 300장·1,410개 GT를 CPU·CUDA 각각 3회 실행했다. 두 provider 모두 매 반복
정답 승인 1,352/1,410(95.89%), 오승인 4건, 완전 성공 이미지 253/300이다.
CPU HTTP 전체·full-path p95는 318.42 / 304.29 / 316.19ms로, 세 목표 모두 미달했다.
CUDA p95는 120.71 / 116.28 / 123.42ms이며 900회 CPU/CUDA 상태·품목 순위가 일치했다.
이 결과로 모델·Catalog·임계값·CPU 설정을 바꾸지 않았다.

각 반복의 공개 segmentation은 APPROVED 1,356개(정답 1,352 + 오승인 4), UNKNOWN 32개,
SEGMENT_RECAPTURE 23개다. GT 누락 3개와 UNKNOWN Top-3 정답 누락 2개를 별도로 기록했다.
IMAGE_RECAPTURE·ERROR는 0장이다. 재촬영 23개에는 GT와 대응하지 않는 추가 검출도 포함되므로
객체 상태 수를 GT 분모로 바꾸지 않는다.

PyTorch/ORT는 원본 302장 상태·품목 순위가 일치했다. 분류·다중 객체 head·detail·ViT 출력은
설정된 수치 허용오차를 통과했다. Detector의 정렬된 원시 query box는 순서 차이로 직접 수치 비교를
통과하지 못했다. 별도 원본 fixture 16개에서 metadata의 uncertainty 0.45 이상 후보 470개를
클래스 무관 bbox IoU로 대응한 검사는 PyTorch/ORT와 CPU/CUDA 모두 통과했다.
이 보충 검사가 모든 저신뢰 raw query의 수치 동일성을 증명하는 것은 아니다.
