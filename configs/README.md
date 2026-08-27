# 설정 디렉터리

- `versions/0.1.6.json`: BIXOLON Bakery AI Scanner의 유일한 활성 제품 조합
- `runtime/`: Worker provider와 고정 dependency lock
- `training/`: 재사용 가능한 학습 pipeline 설정과 recovery evidence
- `experiments/`: 제품 버전과 분리된 실험 설정
- `operations/`: 로그 검수·운영 export 설정
- `archive/`: 과거 제품·실험·release 설정 원문

루트의 작은 JSON 파일은 기존 명령 경로를 위한 `$redirect` 호환 진입점입니다. 새 코드는 canonical
하위 경로를 사용하고 설정은 `bixolon_scanner.configuration.load_json_config`로 읽습니다.
