# 현재 버전

기준일: 2026-08-26

현재 실행 조합은 `0.1.4` 하나이며 별도의 development, demo, production 상태를 두지 않습니다.

| 구성 | 값 |
|---|---|
| 제품·Python·Worker | `0.1.4` |
| Detector·Embedder·Detector policy·Classifier policy | `0.1.4` |
| Store Catalog | `0.1.4`, `CHECKSUM-SHA256` |
| Flutter 내부 빌드 | `0.1.4+7` |
| Detector | class-agnostic YOLO26 objectness, score `0.65`, NMS IoU `0.4` |
| 겹침 hard gate | raw query 기하 + 선택적 landscape 90°·180° 객체 수·bbox 합의 |
| Classifier | DINOv3 ConvNeXt-Tiny soup + 180° 검증 + DINOv3 ViT-B/16 독립 검증 |
| source candidate | `yolo26-objectness-single3-consensus-targeted-rotation` |
| Runtime 원본 manifest SHA-256 | `15f62feb45e9e7bd4d52039badfc8921976d9322367f2a1626279edc9d290d30` |
| Catalog 원본 manifest SHA-256 | `4c35abb9d03798c7df34983b6e401c1af6595cf1a71834efa944c85021e245ab` |
| 겹침 정책 report SHA-256 | `5ca7aa92a91b14c6b810d165f2a0d6b64fde2ba6639d14354deb27d089e0532c` |

## 0.1.4 변경

기존 `0.1.3` ONNX graph, weight, Classifier와 Catalog payload는 바꾸지 않았습니다. Raw query hard
조건에 더해, landscape 장면에서 기본 검출이 정확히 5개이고 bbox 합산 면적 비율이 `0.4` 이상이며
최소 정규화 중심 거리가 `0.63` 이상일 때만 90°·180° 회전 입력을 검사합니다. 회전 검출이 기본보다
하나 이상 많고 기본 bbox 전부가 IoU `0.5` 이상으로 서로 다른 회전 bbox에 대응할 때 누락 가능성으로
판정합니다. 객체 수 증가가 없으면 겹침만으로 재촬영하지 않습니다.

사용자가 제공한 겹침 이미지 5장은 5장 모두 재촬영으로 전환됐습니다. 운영 검수 표본 100장에서는
확정 `DETECTOR_UNCERTAIN_OBJECT` 16장을 모두 잡았고, 정상 통과로 확정한 `054`·`087`의 오재촬영은
0/2였습니다. 미확정 `094`·`095`도 재촬영됐으며 오재촬영 분모에는 포함하지 않았습니다.
기존 accepted detector 개발 회귀 415장의 추가 재촬영은 0장이었습니다. 이 결과는 같은 도메인의
정책 선택·개발 회귀이며 별도 독립 test가 아니므로 일반화 성능, 인증 또는 SLA로 표현하지 않습니다.

최종 운영 100장 CUDA 진단은 평균 54.03ms, p50 46.86ms, p95 86.43ms, p99 96.53ms였습니다.
Worker 실행 최적화 전 0.1.4의 61.59/51.03/106.63/111.19ms 대비 각각
12.3%/8.2%/18.9%/13.2% 감소했습니다. N100 또는 SLA 수치로 해석하지 않습니다.

## 유지되는 계약

`IMAGE_RECAPTURE`에서는 `segmentations`가 비어 있고 실행하지 않은 Classifier, Embedder,
Classifier policy와 Catalog 버전은 `null`입니다. 입력·구성·모델·시스템 장애인 `ERROR`는 재촬영으로
변환하지 않습니다. Runtime/Catalog의 checksum 불일치는 Worker 시작 오류입니다.

`0.1.4+7` Windows 번들은 158개 파일 manifest와 구성요소 version/checksum 검증을 통과했습니다.
Packaged CUDA Worker는 readiness, 제공 겹침 이미지의 `IMAGE_RECAPTURE`, 손상·누락·미지원 입력의
4xx `ERROR` 계약을 [smoke 진단](../diagnostics/packaged-worker-0.1.4-build7-smoke.json)에서
통과했습니다. 실제 N100 100장에서는 CPU-only 평균/p95 575.547/936.381ms, CPU Detector + Intel GPU
Embedder 평균/p95 438.422/727.238ms로 각각 1.313×/1.288× 빨랐고 semantic mismatch는 0건이었습니다.
반면 최대 confidence 차이 0.0085055232가 허용치 0.00001을 초과했고 peak memory 기준도 넘었습니다.
현재 N100 운영 진단 기준은 평균과 p95 모두 `500ms 이하`입니다. 하이브리드 평균은 충족하지만 p95
727.238ms는 아직 초과하므로, 새 기준으로 다시 판정해도 진단은 `passes=false`,
`recommended_provider=openvino`입니다. GPU 우선·CPU fallback
배포 산출물은 이 결과와 한계를 provenance에 포함하며, 해당 수치를 인증이나 SLA로 해석하지 않습니다.

상세 수치와 한계는 [0.1.4 개발 검증 보고서](../evaluation/scanner-0.1.4.md), 과거 판단은
[버전 이력](../archive/version-history.md)에 기록합니다.
