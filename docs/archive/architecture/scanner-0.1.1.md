# Scanner 0.1.1 번들

`configs/versions/0.1.1.json`은 배포 가능한 조합의 유일한 기준입니다. 구성에는 제품 버전, Flutter
build 번호, 선택 Runtime/Catalog의 경로와 고정 manifest SHA-256, CUDA 파일 목록, 평가 증빙의
경로와 SHA-256만 있습니다. lifecycle, gate, waiver 또는 배포 환경 이름은 없습니다.

```mermaid
flowchart LR
    CONFIG["configs/versions/0.1.1.json"] --> VERIFY["원본 해시 검증"]
    VERIFY --> META["Runtime/Catalog 메타데이터 0.1.1"]
    META --> WORKER["자체 포함 Worker"]
    WORKER --> CUDA["Flutter + CUDA 전체 번들"]
    WORKER --> CPU["OpenVINO CPU Flutter 개발자 전달 ZIP"]
```

`scripts/build_app.ps1 -Version 0.1.1`는 준비, Worker 빌드, Flutter Windows 빌드와 최종 manifest
생성을 순서대로 수행합니다. 결과의 `version.json`은 실행 버전을, `provenance.json`은
`source_candidate=n100-convnext-tiny-selective-detector-openvino`와 원본·평가 해시를 기록합니다.
`bundle-manifest.json`은 번들 안의
모든 파일을 SHA-256으로 고정합니다.

OpenVINO 전달 ZIP은 같은 ONNX, Runtime metadata와 Catalog를 사용하고 CUDA runtime을 싣지
않습니다. `BIXOLON_PROVIDER=openvino`, detector `1 worker × 4 threads`, embedder `4 threads`로
실행합니다. `production` Detector를 먼저 실행하고 선택 개수가 6이면서 불확실하거나 최소 점수가
안전 band 밖일 때만 `fold1` Detector를 추가합니다. 같은 `0.1.1` 안에서 모델 graph와 weight,
전처리·후처리 계약은 CUDA 전체 번들과 동일합니다.

Catalog 인증 방식은 `CHECKSUM-SHA256`입니다. 발행자 인증을 주장하지 않으며, Runtime과 Catalog
파일이 바뀌면 시작 시 checksum 검증에 실패합니다. 기존 lifecycle 필드는 archive 검사 전용
reader가 읽을 수 있지만 새 번들에는 생성하지 않습니다.

앱과 전달용 Dart 예제는 `/health/ready`에서 보고된 모든 non-null 구성요소 버전이 `0.1.1`인지
확인합니다. Detector 조기 종료에서는 실행하지 않은 classifier와 Catalog 계열 버전이 `null`인
공개 계약을 유지합니다.
