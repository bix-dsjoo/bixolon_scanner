BIXOLON Bakery AI Scanner 0.1.6 N100 OpenVINO CPU 성능 테스트
=====================================================

1. 이 폴더 전체를 N100 키오스크의 로컬 디스크에 복사합니다.
2. C:\easy 폴더에 실제 촬영 JPEG/PNG 이미지 30장 이상을 넣습니다.
3. RUN-N100-TEST.cmd를 더블클릭합니다.
4. 테스트가 끝나면 n100-0.1.6-result.json을 USB에 복사합니다.
5. 결과 JSON을 Codex에 전달합니다.

CMD에서 직접 실행:
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\N100-STAGE-TEST.ps1" -ImageDirectory "C:\easy" -OutputPath ".\n100-0.1.6-result.json"

고정 OpenVINO CPU 1xauto 설정으로 IMAGE_RECAPTURE 조기 종료를 제외한 평균과 p95 500ms 운영 진단 기준을 확인합니다.
최종 0.1.6의 전체 프레임 object_presence verifier도 OpenVINO CPU로 실행하며 threshold 0.54와 ONNX checksum을 실행 전에 검증합니다.
결과의 passes, target.mean_within_target, target.p95_within_target이 모두 true이면 통과입니다.
이미지 경로와 bytes는 결과 JSON에 기록하지 않고 동일 입력 확인용 SHA-256만 기록합니다.
이 후보는 N100 실측용이며 최종 Setup 설치 프로그램이 아닙니다.
