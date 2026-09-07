# BIXOLON Bakery AI Scanner Lite 0.1.16

검증일: 2026-09-07. 별도 Windows 제품이며 모델·Worker·Runtime·Catalog는 검증된 0.1.16 CPU
배포물과 byte 단위로 같다. 새 앱은 스캔과 실행 로그 두 화면만 구현했다.

## 최신 촬영 설정 반영

공유받은 설정의 의미를 사용자에게 확인한 뒤 상단 20% 제거 → 남은 영역의 중앙 정사각형 →
2048×2048 변환 → 좌우 반전으로 적용했다. 기존 고정 2048 픽셀 crop을 대체한다.
미리보기와 촬영은 같은 source crop 사각형을 사용한다. `camera_windows 0.2.6+4`의
`TextureHandler::mirror_preview_` 기본값이 true임을 확인했으므로 미리보기를 다시 반전하지 않는다.
색상 픽셀 테스트로 상단·양측 제거 및 출력 좌우 교환, 2048×2048 크기를 확인했다.

ZHWY의 지정 USB 장치 ID를 기본 선택한다. 프로그램별 괄호/꺾쇠 표기는 제거해 비교하고
같은 모델의 다른 장치로는 대체하지 않는다. 장치 누락 시 직접 선택 안내와 파일 입력을 제공한다.
실제 열거에서 지정 장치가 일치함을 확인했으나, 이 실행의 카메라 시작은 Windows MFT 하드웨어
리소스 부족으로 실패했다. 최신 설정의 실촬영 성공으로 표현하지 않는다. 파일 입력의 실제
Worker 실행과 로그 검사는 통과했다. 픽셀·선택·회귀 Flutter 테스트 18개가 통과했으며
증빙은 `artifacts/retraining/lite-camera-profile-*.log`와 Windows e2e 보고서에 있다.

## 후속 요청 반영: 검출 박스와 2048×2048 촬영

사용자의 후속 요청에 따라 읽기 전용 박스·객체 번호를 추가했다. 후보나 confidence,
박스 편집 기능은 제공하지 않는다. Worker 좌표는 이미지 크기와 화면 여백에 맞춰 표시하며
메타데이터 로그의 직렬화에는 포함하지 않는다.

카메라는 최대 지원 해상도로 요청하고 중앙 2048×2048 영역을 자른다. 짧은 변이 2048 미만이면
중앙 정사각형을 2048×2048로 확대한다. PNG bytes를 화면과 Worker에 동일하게 전달한다.
미리보기와 결과는 1:1 정사각형으로 창 크기에 맞게 표시한다. 파일 입력은 원본을 유지한다.

후속 Lite 테스트 15개와 Windows 실제 추론 검사를 통과했다. 실카메라 결과는
2048×2048, PNG 3,246,231 bytes였고 Worker `SEGMENTATION` 결과를 수신했다.
실제 3개 빵 이미지의 박스가 객체 위치에 맞게 그려지는 화면도 확인했다.
후속 검사 로그는 `artifacts/retraining/lite-square-*.log`다.
아래 최초 구현 범위의 박스 제외 설명은 후속 요청으로 변경되었다.

후속 UI 요청에 따라 바깥 카드와 중첩 카드, 큰 패딩을 제거했다. 상단 도구막대 아래에
카메라·결과 이미지·판정 메타데이터를 평면 열로 배치하고 두 이미지는 같은 크기로 표시한다.
좁은 창에서는 판정 메타데이터가 이미지 아래로 이동한다. 로그도 평면 목록을 사용한다.
후속 전체 Python 테스트는 1,081개, 기존 Flutter 테스트는 193개가 통과했다.
평면 UI 검사는 `artifacts/retraining/lite-flat-*.log`에 기록했다.

카메라 의존성은 기존 Scanner의 `camera 0.12.0+2`, `camera_windows 0.2.6+4`로 고정했다.
반복 검사 중 Windows MFT가 하드웨어 리소스 부족으로 카메라 시작을 거부한 경우도 있었다.
이 경우 초기화 예외 안내와 파일 입력은 유지되며, 카메라가 성공한 실행의 2048×2048 확인과
구분한다. 플러그인 버전 고정만으로 초기 네이티브 종료 원인이 해결되었다고 판단하지 않는다.

## 구현 범위

- 카메라 열거·선택·실시간 미리보기·촬영, JPEG/JPG/PNG 한 장 선택 후 자동 추론.
- 실행 중 중복 입력과 카메라 변경 차단. 결과는 입력 이미지와 허용된 판정 메타데이터만 표시.
- `IMAGE_RECAPTURE`와 `ERROR`를 별도로 표시. `UNKNOWN`은 상태와 reason code만 표시.
- 입력 접수·완료 결과를 자동 JSONL 기록. 최근 200건 조회 및 전체 이벤트 내보내기.
- 기본 로그 저장 실패 시 결과를 계속 표시하고 별도 비상 로그에 실패와 결과를 기록.
  두 저장소 모두 실패하면 메모리에 보존하여 내보내기를 허용한다.
- 이미지 원문 보관, 후보·좌표·confidence, 수정·라벨링·피드백·학습·모델 관리 기능 없음.

## 검증

| 검사 | 결과 |
|---|---|
| Lite Flutter analyze | 문제 없음 |
| Lite 단위·위젯 테스트 | 12개 통과 |
| 기존 Scanner Flutter analyze / 테스트 | 문제 없음 / 188개 통과 |
| 전체 Python 테스트 | 1,069개 통과 |
| Lite 번들 후속 무결성 테스트 | 3개 통과 |
| Ruff check / format check | 통과 |
| git diff --check | 통과 |
| 실제 Windows 통합 검사 | 최종 실행 통과 |
| 패키지 release Worker 실행 | 0.1.16 readiness 및 실제 이미지 추론 통과 |
| 정상 종료 / 강제 종료 | 소유 Worker 잔류 0개 |

Windows 통합 검사는 실제 Flutter Windows 엔진과 동봉된 CPU Worker를 사용했다.
운영 이미지 `001_...jpg`는 객체 3개 모두 `APPROVED`, 빈 트레이 `092_...jpg`는
`IMAGE_RECAPTURE`, 손상 bytes는 `ERROR`였다. 3개 실행의 접수·결과 6개 이벤트를 파일로
저장한 뒤 저장소를 새로 열어 최근 목록과 내보내기를 확인했다. 내보낸 JSONL에는
`bbox`, `confidence`, `top3`, `prediction`, 이미지 bytes가 없다.

실카메라 1대의 초기화·미리보기·촬영 및 촬영 bytes의 Worker 추론을 확인했다.
촬영 원본은 검증 결과 파일에 저장하지 않았다. 파일 선택 창과 내보내기 저장 창의 경계는
통합 테스트에서 입력 fixture로 대체했으며 실제 OS 대화상자 클릭 자동화는 하지 않았다.

초기 Windows 통합 실행에서 Flutter native 종료 1회, 중간 반복 실행에서 카메라 초기화 실패
1회를 관측했다. 이후 실제 `CameraPreview` 위젯을 사용하고 테스트 종료 시 미리보기를 해제한 뒤
카메라를 dispose하도록 테스트를 정리한 최종 실행은 성공했다. 원인 확정이나 모든 카메라 장치,
장시간 실행의 안정성 보장으로 해석하지 않는다. 실제 앱에서는 카메라 초기화 예외를 안내하고
파일 입력을 계속 제공한다.

## 설치본과 증빙

- `artifacts/lite/0.1.16/BixolonBakeryAIScannerLite-0.1.16-Setup.exe`
- 같은 이름의 `.sha256` 파일: Setup 손상 검사
- `artifacts/lite/0.1.16/payload/bundle-manifest.json`: 전체 payload SHA-256
- `artifacts/lite/0.1.16/payload/provenance.json`: Worker와 Lite 소스 provenance
- `artifacts/lite/0.1.16/windows-e2e/report.json`: 실제 추론·카메라 결과
- `artifacts/lite/0.1.16/windows-e2e/real-result.png`: 실제 이미지 결과 UI
- `artifacts/lite/0.1.16/packaged-lite-smoke.json`: release 시작·추론·종료 검사
- `artifacts/retraining/lite-*.log`: 빌드·분석·테스트 실행 기록

설치 ID는 `{95361E60-673F-4999-B5FD-89B6130447D5}`, 설치 폴더와 바로가기 이름은
`BIXOLON Bakery AI Scanner Lite`다. 기존 Scanner와 함께 설치할 수 있다. CPU Worker와
VC++ runtime을 포함하며 Python·Flutter·CUDA 설치를 요구하지 않는다.
Setup을 시스템에 설치·제거하는 검사는 수행하지 않았으며, Setup에 들어가는 최종 release
payload를 직접 실행했다. Authenticode 서명은 없으며 checksum은 발행자 인증을 제공하지 않는다.

## 최신 사용자 요청 반영: 실행 로그 이미지·박스·Top-3

아래 결과가 위 초기 구현의 이미지 미보관·후보 제외 설명을 대체한다. 사용자 요청으로
입력 이미지를 별도 파일로 30일 보관하고, 로그를 펼치면 이미지 위 검출 박스와 객체 판정을
복원한다. Top-3 이름은 Worker가 제공한 순서 그대로 읽기 전용 표시·기록하며 confidence와
후보 선택은 제공하지 않는다. IMAGE_RECAPTURE와 SEGMENT_RECAPTURE는 빨간색이다.
메타데이터는 자동 삭제하지 않으며 만료 이미지 정리는 앱 시작·로그 접근 시 실행한다.
이전 버전에서 저장하지 않은 이미지는 복원할 수 없다. ZIP 내보내기는 JSONL, 보관 이미지,
포함 여부 index를 담는다. 이미지 저장 실패도 기록하며 추론 결과 표시는 계속한다.

Lite 분석 및 단위·위젯 테스트 24개를 통과했다. 재시작 후 이미지 bytes·박스·Top-3 복원,
기본 이미지 저장 실패 시 비상 저장, 30일 만료 정리, 이전 로그 조회, ZIP 포함과 읽기 전용
로그 UI를 검증했다. 실제 Windows 통합 테스트도 통과했으며 동일한 운영 이미지 3개로
APPROVED·IMAGE_RECAPTURE·ERROR, 6개 이벤트와 ZIP 이미지 포함을 확인했다.
`windows-e2e/log-result.png`를 육안 확인하여 보관 이미지의 객체 3개 박스와 판정 표시를 검증했다.

최신 카메라 검사에서는 지정 ZHWY 장치 ID를 찾았으나 `camera_error`와 Windows 하드웨어 MFT
리소스 부족 오류로 초기화하지 못했다. 현재 촬영 설정에서 실카메라 촬영 성공을 주장하지 않는다.
상단 20% 제거·중앙 정사각형·2048×2048·좌우 반전은 픽셀 테스트로 검증했다.
플러그인 미리보기의 기본 좌우 반전을 고려하여 이중 반전하지 않는다.

최종 검사에서 `ruff check`, `ruff format --check`, 전체 Python 테스트, 두 앱의
`flutter analyze`, 기존 Scanner Flutter 테스트 193개와 Lite 테스트 24개, `git diff --check`가
통과했다. 새 release payload의 전체 checksum을 확인했고 실제 실행에서 정상 종료·강제 종료
모두 소유 Worker 프로세스 잔류가 없었다. Worker HTTP 추론도 통과했다.
기록은 `artifacts/retraining/lite-review-*.log`와 `packaged-lite-smoke.json`에 남긴다.
