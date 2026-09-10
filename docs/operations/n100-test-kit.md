# N100 0.2.1 단독 측정 키트

현재 USB 전달 요청은 0.2.1만 측정한다. PowerShell이나 설치된 Python에 의존하지 않도록
`operations/n100_test_kit.py`를 독립 `N100-MEASURE.exe`로 패키징했다.
루트의 `2_MEASURE.cmd`는 같은 폴더의 EXE를 직접 호출하므로 PATH에 PowerShell이 없어도 된다.

`D:\N100`에서 이전 파일은 `previous-0.2.0`에 보존한다. 새 키트는 `Benchmark`의 검증된
0.2.1 Worker·Runtime·Catalog·OpenVINO·라이선스와 고정 `log132` 이미지를 사용한다.
`scripts/prepare_n100_test_kit.py --kit-root D:\N100`은 복사본의 hash를 검사하고 실행 파일,
안내문과 `KIT-MANIFEST.json`을 작성한다. 이전 폴더와 결과 폴더는 현재 파일 검증에서 제외한다.

N100 내부 SSD로 폴더를 복사한 후 Scanner와 다른 Worker를 종료하고 `2_MEASURE.cmd`를 실행한다.
파일 checksum → Worker 준비(최대 10분) → warmup 10회 → 132장 3회 순서다.
`results`의 ZIP에 396개 응답, 회차별 상태 개수와 p50/p95/p99, 하드웨어·드라이버 정보,
시작·종료 readiness 및 Worker 로그를 보존한다. 실패와 중단의 부분 결과도 보존한다.
실행기는 자신이 시작한 Worker만 종료하며 이미 실행 중인 Scanner가 있으면 측정을 시작하지 않는다.

`3_VERIFY_FILES.cmd`는 파일 검증만 한다. 개발 확인용 `N100-MEASURE.exe --smoke`는 이미지
3장만 사용하므로 정확도·성능 성적으로 해석하지 않는다. 빌드 PC에서 PATH를 비우고 독립 EXE의
검증·실제 스캔을 확인하며, 이 결과는 실제 N100 장비 측정과 구분한다.

CPU 검출 → Intel UHD GPU primary/detail → CPU verifier 및 명시적인 CPU fallback은 유지한다.
provider 전환은 readiness와 Worker 로그를 함께 확인한다. CPU fallback 시간을 GPU 성능으로
표시하지 않는다. 정답 승인·오승인·미검출은 저장한 응답을 고정 GT에 대응해 평가하며
APPROVED 개수만으로 정확도를 계산하지 않는다. Checksum은 발행자 인증이 아니다.
