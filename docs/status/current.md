# 현재 버전

기준일: 2026-08-20

현재 배포 조합은 `0.0.2` 하나이며 별도의 development, demo, production 상태를 두지 않습니다.

| 구성 | 값 |
|---|---|
| 제품·Python·Worker | `0.0.2` |
| Detector·Embedder·Detector policy·Classifier policy | `0.0.2` |
| Store Catalog | `0.0.2`, `CHECKSUM-SHA256` |
| Flutter 내부 빌드 | `0.0.2+2` |
| 사용자 표시·Windows ProductVersion | `0.0.2` |
| CPU 전달 기본 profile | detector `1 worker × 4 threads`, embedder `4 threads` |
| N100 profile 측정 | 수신 측 대표 이미지 30장 이상 회신 대기 |
| source candidate | `2.0.1-rc.3` |
| Runtime 원본 manifest SHA-256 | `7b7364c381782493d597520da4b4bc55993068ae97d7ba405e30be4b2d92389a` |
| Catalog 원본 manifest SHA-256 | `36afc8946f3661135f575f6c03fd968bd4cfd59cd36f3713e90359a6f8442c41` |
| 평가 report SHA-256 | `be7d009d5de1d41ae3ddf10333b510755016301efc54abf027b1b710e4bfe736` |
| 평가 trace SHA-256 | `b06e42514ace6d93e3d67a5fe429462f8f0a5bf49edb28339e342ee900f732cb` |
| 평가 breakdown SHA-256 | `1aee711f75f6f33051504391d16b20802f19e456e20755cc4cb20d8e74c6eddc` |

`0.0.2`는 rc.3의 model graph, weight와 판정 정책을 그대로 사용합니다. 변환 과정은 원본 payload의
파일 SHA-256 집합이 동일한지 확인하고 Runtime/Catalog 메타데이터의 구성요소 버전만 바꿉니다.
Catalog에는 `signature.json`, signing key, HMAC 또는 lifecycle 메타데이터가 없습니다. 파일별
checksum 불일치는 Worker 시작 오류입니다.

CPU 전달 기본 profile은 N100 측정 전 안전 fallback입니다. 전달 ZIP의 `benchmark-n100.ps1`이
`1×4`, `2×2`, `4×1`을 동일 이미지로 비교하고 parity-safe p95와 peak memory로 추천값을 기록합니다.
현재 문서에는 N100 수치를 기재하지 않으며 회신 결과도 진단값으로만 기록합니다.

rc.3 개발 회귀의 정답 승인 coverage는 1,311/1,410(92.9787%), 승인 오인은 0/1,410,
Candidate out은 0건이었습니다. 평균/P95/P99는 84.89/95.57/100.63ms입니다. 이 수치는 기존
개발 데이터 재평가이며 독립 일반화 성능이나 SLA 인증으로 표현하지 않습니다. 전체 과거 판단과
제한은 [버전 이력](../archive/version-history.md)에 남아 있습니다.
