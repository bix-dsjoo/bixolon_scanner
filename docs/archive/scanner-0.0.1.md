# Scanner 0.0.1 번들

`0.0.1`은 `2.0.1-rc.3` Runtime/Catalog payload를 단일 제품 버전으로 처음 다시 패키징한
기준선입니다. Flutter 내부 build는 `0.0.1+1`이었고 CUDA runtime을 포함한 전체 Windows 번들을
생성했습니다.

Runtime과 Catalog의 model graph, weight, adapter, support와 prototype payload는 바꾸지 않았으며
Catalog는 `CHECKSUM-SHA256`을 사용했습니다. 당시 기준 설정은
[`configs/archive/versions/0.0.1.json`](../../configs/archive/versions/0.0.1.json)에 보존합니다.

이 버전은 독립 일반화 성능, 인증 또는 SLA 달성 상태가 아닙니다. CPU 실행 설정과 Flutter 개발자
전달 패키지가 추가되면서 활성 제품 버전은 `0.0.2`로 변경됐습니다.
