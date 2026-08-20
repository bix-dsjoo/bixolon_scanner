# 문서 인덱스

문서는 현재 계약, 작업 가이드, 운영 상태와 실험 기록을 분리합니다. 구현과 함께 유지해야 하는 공개 계약은 `contracts`, 변경 절차는 `guides`, 시점에 따라 달라지는 값은 `status`, 모델 후보의 근거는 `experiments`에서 관리합니다.

## 기준 문서

- [아키텍처 개요](architecture/overview.md): Python과 Flutter의 소유권 및 의존 방향
- [Scanner 2.0.0 전체 설계](architecture/scanner-2.0.0.md): 공용 Runtime 승격, frozen Embedder와 매장별 10-shot Catalog 자동 활성화
- [API 계약](contracts/api.md): `POST /v1/scan`, 네 상태, 오류와 응답 규칙
- [개발 가이드](guides/development.md): 설치, CLI, 설정, 검증 절차
- [Scanner 2.0 owner-private test](guides/scanner-2.0-private-test.md): model-free preflight, 단일 실행과 production finalization
- [모델 승격 가이드](guides/model-promotion.md): 실험 수명주기와 수동 gate
- [학습 파이프라인 1.0.0](guides/training-pipeline-1.0.0.md): Detector·Classifier 독립 계약, 검증과 버전 정책
- [Bread Classifier 200장 전용 1.1.1+ 계획](experiments/bread/bread-classifier-200-only-1.1.1-plan.md): 허용 데이터, patch 반복과 종료 조건
- [Scanner 2.0.0 300장 개발 평가](experiments/bread/scanner-2.0.0-development-300.md): DINOv3 RC.10 요청 지표, 난이도별 분석, parity와 1 IPS 성능 blocker
- [Scanner 2.0.1 RC.1 대 RC.3 및 운영 승격](experiments/bread/scanner-2.0.1-rc.3-single-objects.md): 동일 classifier 출력의 안전 정책 A/B와 rc.3 owner-waiver production 기록
- [현재 상태](status/current.md): 운영·실험 버전과 승격 여부

## 실험 기록

- [Bread](experiments/bread/README.md)
- [Detector](experiments/detector/README.md)
- [RPC200](experiments/rpc200/README.md)

실험 문서는 재현성과 의사결정 근거를 위한 기록입니다. 보고서에 `promoted`가 명시되지 않은 모델을 운영 기본값으로 해석하지 않습니다.

## 앱 문서

- [Flutter 앱 운영·개발](../apps/product_scanner/README.md)
- [현재 디자인 시스템 계약](../apps/product_scanner/DESIGN_SYSTEM.md)
- [디자인 시스템 개선 이력](../apps/product_scanner/docs/DESIGN_SYSTEM_HISTORY.md)
