# BIXOLON Bakery AI Scanner 현재 상태

최종 제품 버전은 `0.2.1`, Flutter 앱은 `0.2.1+24`다. 선택 구성은
`repvit-m0.9-supervised-r4-n100-20260910`이며 버전 설정에 원본 Runtime·Catalog와
N100 실측 응답·로그를 포함한 평가 증빙의 SHA-256을 고정했다.

D-FINE-S640 CPU detector → RepViT-M0.9 192 GPU primary → 선택적 DINO224 detail /
CPU ViT160 verifier를 사용한다. primary batch2/1, CPU4스레드, GPU FP16,
동일 검증 입력 재사용과 선택적 병렬 검증을 사용하며 CPU fallback을 유지한다.
정식 N100 Lite 앱과 독립 Worker 실행문은 이 실측 구성을 사용한다.

N100 실측 132장×3회 GPU396/396건이 HTTP1초 이내였다.
p50/p95/p99/최대는493.3/732.4/799.1/949.8ms, 매 반복 정답승인1085/1096,
오승인0·미검출0, UNKNOWN9·SEGMENT_RECAPTURE18·추가검출16이다.
CPU전용은4/396건이1초를 넘었다. 워밍업·순차 실행 결과이며 장시간·다른 장면의 SLA가 아니다.
개별 승인2개 손실과 별도 final300 기존 오승인4개, 독립 validation 부재는 남아 있다.

[버전 설정](../../configs/versions/0.2.1.json), [최종 변경점](../architecture/scanner-0.2.1.md),
[N100 실측·한계](../experiments/n100-0.2.1.md)를 참조한다.
최종 배포 폴더는 `artifacts/distributions/0.2.1-final-n100`이다.
