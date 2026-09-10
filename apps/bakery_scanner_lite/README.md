# BIXOLON Bakery AI Scanner Lite

Worker 0.1.16 기반의 별도 Windows 앱이다. 기존 Scanner의 화면·기능·SDK 결과 모델을 재사용하지
않는다. `core/design_system`의 token/theme과 BIXOLON 로고·Pretendard만 디자인 자산으로 복사했다.
앱과 설치본은 0.1.16, 내부 build는 19다. 기존 Scanner와 설치 ID·폴더·로그 저장소가 다르다.

## 사용

상단 도구막대에 카메라 선택·촬영·파일 입력을 모았다. 본문은 카메라, 결과 이미지,
판정 메타데이터를 나란히 배치하며 카드 컨테이너를 사용하지 않는다. 좁은 창에서는 판정을
두 이미지 아래에 배치한다. 두 이미지 영역은 같은 정사각형 크기를 사용한다.

카메라 목록에서 장치를 선택한 뒤 **촬영하고 분석**을 누르거나 **이미지 파일 선택**으로 JPEG/JPG/PNG
한 장을 연다. 원본 미리보기와 최종 상태·객체별 판정·reason code·요청 ID·처리 시간·실행 버전을
확인한다. 분석 중 추가 실행·카메라 전환은 비활성이다. 파일 선택 취소는 실행 기록을 만들지 않는다.

결과에는 편집·승인·후보 선택 동작이 없다. 검출 박스와 객체 번호를 읽기 전용으로 표시한다.
모델이 제공한 Top-3 후보 이름을 순위대로 표시하고 박스 좌표와 함께 로그에도 저장한다.
`APPROVED`는 `prediction.class_name`을 **승인 상품**으로 표시하며 class ID와 이름을
로그·내보내기에도 보존한다. 이름이 누락된 응답은 class ID를 표시한다. 과거 로그에서 승인
상품을 기록하지 않은 경우에는 **승인 상품명 기록 없음**을 표시하며 후보로 추정하지 않는다.
이 수정의 배포본은 기존 0.1.17 모델을 유지한 0.1.19이며, 별도 소스 snapshot
`C:/workspace/bixolon_scanner_release_0_1_19`에서 빌드했다. 설치 파일과 검증 보고서는
`artifacts/distributions/0.1.19`에 보존한다.
후보 선택과 결과 수정은 없으며 confidence는 결과 모델·로그·UI에 포함하지 않는다. 기존 Worker의 응답은 수신 즉시 명시적인 허용 필드로
투영하고 원문 응답을 저장하지 않는다. Lite가 제공하는 별도 API 서버나 모델 관리 API는 없다.
IMAGE_RECAPTURE는 이미지 전체 재촬영이며 ERROR는 입력·연결·시스템 오류로 별도로 표시한다.
IMAGE_RECAPTURE와 SEGMENT_RECAPTURE의 상태와 박스는 빨간색이다. UNKNOWN은 주황색이다.

## 촬영과 박스 표시

지정된 ZHWY 카메라의 Windows 장치 ID를 기본으로 선택하며 최대 지원 해상도를 요청한다.
지정 장치가 없으면 다른 카메라를 자동으로 열지 않고 직접 선택 안내를 표시한다. 사용자가
직접 선택한 장치는 새로고침 후에도 유지한다.

촬영 처리는 상단 20% 제거 → 남은 영역의 중앙 정사각형 자르기 → 2048×2048 리사이즈 →
좌우 반전 순서다. 원본의 비율을 찌그러뜨리지 않으며 센서 해상도가 낮을 때는 확대된다.
같은 crop 계산과 좌우 반전을 미리보기에도 적용한다. 화면과 Worker는 같은 최종 PNG bytes를
사용하므로 검출 박스에 별도 반전을 적용하지 않는다. 화면은 1:1 비율로 창 크기에 맞게 표시한다.

고정한 `camera_windows 0.2.6+4`의 미리보기 TextureHandler는 기본적으로 이미 좌우 반전한다.
미리보기에는 crop만 추가하고, 촬영 PNG에 좌우 반전을 적용해 중복 반전을 막는다.
다른 프로그램의 `(장치 ID)`와 플러그인의 `<장치 ID>` 표기 차이는 내부 장치 ID 비교로 처리한다.

파일 입력은 원본을 자르지 않으며 정사각형 영역 안에 원본 비율로 표시한다. 검출 박스는 해당
원본의 여백과 배율을 함께 계산하므로 이미지와 같은 위치에 그려진다. 박스는 수정할 수 없다.

## 자동 기록과 내보내기

- 입력 접수와 최종 결과를 같은 `attempt_id`의 JSONL 이벤트로 자동 기록한다.
- Worker 요청 ID는 결과 수신 후 기록한다. Worker 실행 전 실패한 입력은 `origin=client`와
  `CLIENT_...` 오류 사유 및 앱 요청 ID를 사용하며 모델 판정으로 표현하지 않는다.
- 발생 시각(UTC), 입력 방식, 카메라 이름/파일명, 최종 상태, 객체별 상태와 사유, 처리 시간,
  실행 버전, `log_storage_status`와 실패 사유를 기록한다.
- **실행 로그**에는 최근 200개 실행의 마지막 상태를 표시한다. 결과 미완료 접수 기록도 보존한다.
- 로그를 펼치면 당시 이미지와 검출 박스, 객체별 판정과 Top-3 이름을 읽기 전용으로 확인한다.
- **전체 로그 내보내기**는 ZIP에 전체 접수·결과·비상 이벤트 `logs.jsonl`, 보관 중인
  `images/`와 포함 여부를 기록한 `image-index.json`을 담는다. 같은 실행의 접수/완료 이벤트는
  `attempt_id`로 연결하며 JSONL에는 이미지 bytes 대신 상대 참조를 저장한다.
- 기본 위치는 Windows의 사용자별 application support 디렉터리 안 `logs`다. 제품 리소스의
  CompanyName `BIXOLON`, ProductName `BIXOLON Bakery AI Scanner Lite`로 기존 앱과 분리된다.
- 기본 기록 실패는 화면에 표시하며 `%TEMP%/BIXOLON-BakeryAIScannerLite-emergency/log-failures.jsonl`에
  `failed` 상태와 결과를 별도로 기록한다. 두 저장소 모두 실패하면 메모리와 stderr에 실패를
  남기고 실행 중 내보내기를 제공한다. 모든 저장장치가 실패한 상태의 강제 종료까지 영구 기록을
  보장할 수는 없다. 기본 로그와 비상 로그는 자동 삭제하지 않으며 보관·삭제는 매장 파일 관리로 한다.

사용자가 요청한 실행 로그 이미지 조회를 위해 실제 추론 입력을 로그 저장소의 `images/`에
30일 보관한다. 카메라는 변환 후 PNG, 파일은 입력 bytes를 그대로 복사한다. 앱 시작 또는 로그
접근 시 만료 파일을 정리하며 앱이 닫혀 있는 동안은 삭제하지 않는다. 메타데이터는 유지한다.
이전 버전에서 보관하지 않은 이미지와 만료 이미지는 복원할 수 없으며 화면에 이를 표시한다.
이미지 저장 실패는 결과 표시를 막지 않고 실패 상태로 기록한다. 기본 저장 실패 시 비상 저장소를
사용한다. 내보낸 ZIP은 자동 삭제 대상이 아니므로 사용자가 보관·삭제한다.
카메라 플러그인의 임시 파일은 읽기 직후 `finally`에서 삭제한다. 모델 원문·confidence는 기록하지 않는다.

## 실행과 구성

설치본에는 0.1.16 CPU Worker·Runtime·Catalog와 VC++ 재배포 패키지가 포함된다. Python, Flutter,
CUDA 설치는 필요 없다. 실행 시 숨김 Worker를 전용 loopback 포트에서 시작하고 0.1.16 readiness를
확인한다. Detector/Embedder는 각각 4 threads다. 앱 종료 시 소유 Worker를 종료하며 Windows job으로
비정상 앱 종료 시 자식 프로세스 잔류도 방지한다. 설치 파일에는 Authenticode 서명이 없다.

기존 Scanner의 ROI 편집·라벨링·피드백·학습·모델 교체·고급 설정·계정·통계·호환 모드는 없다.

## 빌드와 검증

저장소 루트에서 `scripts/build_lite.ps1`을 실행한다. 검증된 0.1.16 CPU 배포물이 필요하다.
Lite Flutter release와 Worker copy의 전체 파일 해시를 확인한 뒤 별도 Inno Setup을 생성한다.
`version.json`, `provenance.json`, `bundle-manifest.json`, 원본 Worker 검증 기록과
DINOv3·Apache·Pretendard 라이선스를 포함한다. SHA-256은 손상 검사용이며 발행자 인증은 아니다.
다시 만들 때는 `-Force`를 사용하며 기존 Scanner 0.1.16 배포물은 변경하지 않는다.

Lite 디렉터리에서 `flutter analyze`, `flutter test`를 실행한다. 실제 Windows 통합 검사는
`integration_test`와 `artifacts/lite/0.1.16`의 검증 기록을 참조한다.

통합 검사는 `LITE_E2E_ROOT`를 저장소 절대 경로로 설정하고 검증된 payload의 `worker`를
`build/windows/x64/runner/Debug/worker`에 복사한 뒤
`flutter test integration_test/lite_windows_test.dart -d windows`로 실행한다.
파일 선택 경계만 테스트 입력으로 대체하며 앱·Worker·추론·로그는 실제 실행한다.
연결된 실카메라가 있으면 미리보기·촬영도 검사하며 촬영 원본을 검증 파일로 저장하지 않는다.
