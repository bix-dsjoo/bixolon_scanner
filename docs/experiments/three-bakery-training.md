# three_bakery 전량 학습과 고정 벤치마크 평가

## 정정본 300장 재학습: 2026-09-08 실행

### 현재 목표: GT 객체 정답 승인율 99%

사용자가 최종 세트 접근 전에 “원본 객체 기준 99%라고 생각하면댐”으로 목표를 변경했다.
현재 정확도 기준은 **전체 GT 객체 중 올바른 APPROVED의 비율 99% 이상**이다.
개발 원본 649개에서는 643개 이상, 최종 GT 1,410개에서는 1,396개 이상을 요구한다.
누락·UNKNOWN·재촬영·ERROR의 GT는 분모에서 제거하지 않는다. 오승인 0건과
CPU 전체 요청/full-path p95 300ms 조건은 유지하며 이미지 전체 성공률은 참고 진단으로 남긴다.

설정은 `configs/experiments/bread/three_bakery_objective.json`이고, 진입점에
`--objective-config configs/experiments/bread/three_bakery_objective.json`을 추가한다.
완료된 HTTP 응답·GT 대응·지연은 그대로 재사용하며 `comparison-objects.json`에 새 순위를 저장한다.
적격 여부 이후 순위는 오승인 최소 → 원본 정답 승인 객체 최대 → 합성 정답 승인 객체 최대 →
누락 최소 → UNKNOWN Top-3 누락 최소 → CPU p95 순이다. 상위 두 구성의 추가 seed와
최악·중앙 결과 선택 규칙은 유지한다. `objective-revision.json`에 변경 시점과 해시를 보존한다.
최종 provider별 `report.json`은 원래 HTTP 측정 기록이고, `objective-report.json`은 같은 기록에
고정한 객체 목표를 적용한 판정이다. `final/report.json`과 `final/report.md`가 이를 종합한다.
이하의 297/300 이미지 목표는 이전 계획 이력이며 현재 합격 판정에는 위 객체 기준을 적용한다.

### 추가 요청: 합쳐진 ROI는 SEGMENT_RECAPTURE

기본 12개 및 추가 seed 비교 후, 최종 300장에 접근하기 전에 사용자가 합쳐진 객체 ROI를
`SEGMENT_RECAPTURE`로 차단하도록 요청했다. 이전 결과는 `three-bakery-revised300`에 보존하고,
새 정책 진단은 `artifacts/retraining/three-bakery-revised300-roi-integrity`에서 진행한다.
동일한 정정 원본으로 이미 학습한 검출기·분류기 가중치를 복사해 사용한다. 원본·합성 snapshot은
동일 해시의 디렉터리 junction으로 참조하며 새 head·Runtime·Catalog·비교·평가는 새 디렉터리에 둔다.
최종 벤치마크는 새 후보·정책을 고정한 이후에만 읽는다.

설정은 `configs/experiments/bread/three_bakery_roi_integrity.json`이다. 각 분류기의 고정 backbone
특징 768차원에 LayerNorm·128차원 MLP의 단일/다중 객체 head를 학습한다. 원본 649개 객체 모두와
해당 구성의 합성 학습 장면을 사용해 단일 3,000개·두 객체를 합친 ROI 3,000개를 만든다.
원본 다중 장면에서 가능한 겹침 pair도 포함한다. 다른 객체는 기존 이웃 소유권 마스크로 처리한다.
부모 원본 SHA-256, 대상 객체 index, 합친 bbox, 라벨과 실제 사용 표본을 보존한다.
합성 진단 400장 및 최종 300장을 이 head 학습에 쓰지 않는다.

head는 seed 20260911, 50 epoch, AdamW LR 0.003, weight decay 0.0001, dropout 0.1,
마지막 epoch로 고정한다. 실제 판정 임계값은 첫 실행 전에 0.8로 기록했다. 이 추가 head는
기존 품목 분류 head와 backbone을 수정하지 않고, 같은 ONNX의 192 primary batch에서 특징을 공유한다.
출력은 metadata로 지정하며 공개 API에 원시 출력이나 확률 필드를 추가하지 않는다.
영역의 다중 객체 확률이 임계값 이상이면 `SEGMENT_RECAPTURE`를 우선하고 detail·ViT 승격을 막는다.
단순한 bbox 겹침만으로 재촬영하지 않는다. 모델이 다중 객체를 놓칠 가능성은 남으므로
실제 개발 진단의 차단 성공과 정상 승인 손실을 함께 보고한다.

초기 공유 head의 오프라인 진단에서는 SSDLite margin dense의 합성 394번 오승인이 차단됐다.
실제 다중 완전 성공은 97/100에서 96/100으로 감소했다. 이것은 반올림된 공개 bbox를 사용한
오프라인 진단이며 Worker HTTP 결과나 CPU 지연 성적이 아니다. 실행 연결 후 12개 구성과
추가 seed, thread 조합을 새 정책으로 비교한다. 진입점의 `compare`, `export`, `evaluate`에
`--roi-integrity-config configs/experiments/bread/three_bakery_roi_integrity.json`을 추가하고
`--work artifacts/retraining/three-bakery-revised300-roi-integrity`를 사용한다.

실제 Worker HTTP를 통한 기본 seed 재비교에서 SSDLite margin dense는 다중 97/100장을 유지했고,
원본·합성 개발 진단의 오승인은 모두 0건이었다. 실제 다중 3회 중 가장 느린 full-path p95는
159.32ms, 합성 양성은 79.36ms였다. 합성 394번의 합쳐진 ROI 하나는 `SEGMENT_RECAPTURE`,
다른 다섯 ROI는 `APPROVED`로 확인했다. 재촬영으로 막은 누락 객체는 완전 성공으로 세지 않는다.
이는 기본 seed 한 구성의 개발 결과이며 최종 선택·최종 300장 평가 결과는 아니다.

### 기본 비교 실행 기록

새 실행은 `configs/experiments/bread/three_bakery_revised300.json`과
`artifacts/retraining/three-bakery-revised300`을 사용한다. 아래 기존 252장 실행 이력은 보존한다.
현재 정정본은 단일 200장·다중 100장·배경 2장, 객체 649개다. 원본 SHA-256으로 기존 ID를 유지하고
223번 객체 4 C07 꽃빵, 284번 객체 3 C03 와플, 271번 객체 6 C05 누락 보완을 검사한다.
사용자의 이번 요청은 정정본 학습 사용 승인으로 기록하며 신규 SAM 마스크의 사용자 전수 검수로
표현하지 않는다. 신규 177개 마스크는 에이전트가 원본 픽셀을 유지한 27개 검수판에서 확인했다.
일부 마스크의 트레이 잔여물·구멍·가림은 `sources/visual-review.json`에 기록했다.
신규 다중 마스크는 합성에 쓰지 않으며, 기존 검수된 단일 cutout 176개만 사용한다.
검출기는 원본 302장, 분류기는 정정된 bbox에 8% 주변 여백을 둔 RGB crop 649개 전부를 사용한다.
각 epoch에 실제 소비한 원본 SHA와 crop ID별 횟수를 검사한다.

학습량과 12개 구성은 아래와 같으며 기본 seed는 20260908, 반복 seed는 20260909·20260910,
공통 합성 진단 400장의 seed는 20261908이다. Catalog는 649개 원본 crop에서 perceptual 중복을
제거하고 품목별 hash 순으로 10개씩 선택해 새로 만든다. 이전 빵 학습 가중치·Catalog는 사용하지 않는다.

CPU는 Intel Core Ultra 9 285K의 ONNX Runtime CPU로 측정한다. 기본 seed 후보마다 detector/embedder
thread 조합 (4,4), (8,4), (4,8), (8,8), (12,8), (8,12)를 실제 다중 100장으로 비교한다.
(4,4)와 공개 상태·품목 순위가 일치하는 조합 중 full-path p95, p99, 총 thread 수, 조합 순으로 선택한다.
추가 seed에는 그 조합을 고정한다. 선택 조합으로 실제 다중 100장과 합성 양성 320장을 따로
warmup 10회 후 3회 측정하며 모든 반복의 full-path p95가 300ms 이하여야 개발 속도 조건을 충족한다.
개발 오승인 0건·실제 다중 완전 성공 99/100 이상·속도 조건 충족 후보를 우선하고,
이후 오승인·실제 다중 성공·합성 성공·누락·UNKNOWN Top-3 누락·CPU p95 순위를 적용한다.

최종 목표는 CPU·CUDA 각 반복에서 완전 정답 승인 297/300 이상, 오승인 0건이고,
CPU는 각 반복 전체 요청과 full-path HTTP p95 모두 300ms 이하여야 한다. 각 경로의 p50·p95·p99·최대·표본 수를
기록한다. HTTP 측정은 파일 읽기와 모델 로딩을 제외하며 업로드·디코딩·전체 추론·응답 파싱을 포함한다.
동시 요청은 1개이며 학습과 벤치마크를 겹치지 않는다. 이 목표는 해당 고정 벤치마크와 측정 장비의 관측값이다.

개발 CLI와 Python 자식 프로세스는 `input_isolation`의 audit guard로 최종 데이터셋과 과거 개별
평가 출력에 대한 파일 접근을 차단한다. Python audit guard는 OS sandbox가 아니며 원본·파생물의
허용된 부모 해시 검증을 함께 적용한다. 최종 모델·Catalog·정책·CPU 설정·평가 코드 해시를 고정한 후
평가셋 접근을 허용하고 그 뒤에 기존 평가셋을 읽는 전체 테스트를 실행한다.
최종 결과에 따른 재학습이나 정책 변경은 허용하지 않는다.

현재 전체 이미지 재촬영은 객체 없음, 너무 작은 검출 객체, detector 처리 한계 초과,
정상 검출에 대응하지 않는 불확실 후보 등 기존 hard 조건으로 결정한다.
겹침·가림 정도 자체를 판정하는 별도 재촬영 모델은 없으며, 활성 metadata에는 흐림·노출의
재촬영 임계값도 설정하지 않았다. 밀집·저신뢰 ROI는 기존 detail/verifier 및 객체별
UNKNOWN·SEGMENT_RECAPTURE 정책을 따른다. 여러 객체를 하나로 검출한 고신뢰 승인을
이 경로가 항상 차단하지는 않는다. IMAGE_RECAPTURE도 완전 성공 이미지에는 포함하지 않는다.

CUDA parity 평가 도구의 bbox 대응에는 SciPy가 필요하다. 이번 CUDA Python 환경에는
`python -m pip install --no-deps scipy==1.17.1`로 이를 보완했다. 이 환경에는 PyTorch가 없으며
Worker 요청 경로에서 SciPy나 학습 모듈을 import하지 않는다. 설치 명령과 검증 결과는
`reproduction/cuda-parity-dependency-repair.json`에 보존한다.

```powershell
python -m bixolon_scanner.experiments.bread.three_bakery prepare --config configs/experiments/bread/three_bakery_revised300.json --work artifacts/retraining/three-bakery-revised300
python -m bixolon_scanner.experiments.bread.three_bakery train --config configs/experiments/bread/three_bakery_revised300.json --work artifacts/retraining/three-bakery-revised300
python -m bixolon_scanner.experiments.bread.three_bakery compare --config configs/experiments/bread/three_bakery_revised300.json --work artifacts/retraining/three-bakery-revised300
python -m bixolon_scanner.experiments.bread.three_bakery export --config configs/experiments/bread/three_bakery_revised300.json --work artifacts/retraining/three-bakery-revised300
python -m bixolon_scanner.experiments.bread.three_bakery evaluate --config configs/experiments/bread/three_bakery_revised300.json --work artifacts/retraining/three-bakery-revised300
```

선택 후 `final-candidate.json`의 `cpu_profile[0]`, `cpu_profile[1]`을 각각
`BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS`, `BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS`에 적용한다.
제품 버전 0.1.16과 공개 API는 유지하며 EXE는 만들지 않는다.

최종 선택과 평가가 끝난 뒤 저장소 루트에서 다음 명령으로 CPU Worker를 실행할 수 있다.
`worker-cpu.json`에 기록된 Python 환경을 사용하며 Runtime·Catalog·thread 설정을 함께 적용한다.
서버 종료는 `Ctrl+C`다. 기본 `/v1/scan`의 multipart 필드는 `image`이며 실행 전에 모델 checksum을 검증한다.

```powershell
@'
import os
import sys
from pathlib import Path
from bixolon_scanner.configuration import load_json_config

work = Path("artifacts/retraining/three-bakery-revised300")
frozen = load_json_config(work / "final-candidate.json")
settings = load_json_config(Path(frozen["candidate_path"]) / "worker-cpu.json")
if Path(sys.executable).resolve() != Path(settings["python_executable"]).resolve():
    raise SystemExit("Use the Python executable recorded in worker-cpu.json")
for name in tuple(os.environ):
    if name.startswith("BIXOLON_"):
        del os.environ[name]
os.environ.update(settings["environment"])
from bixolon_scanner.worker.cli import serve
serve()
'@ | python -
```

## 기존 252장 실행 이력

사용자 승인 계획에 따라 `datasets/three_bakery`의 단일 200장, 멀티 50장, 배경 2장만으로
검출기·분류기·Catalog를 새로 만드는 실험이다. 제품 버전은 `0.1.16`이며 EXE 제작은 포함하지 않는다.
이 문서는 실행 중인 기록이며 최종 성적은 아직 확정되지 않았다.

## 고정 조건

설정은 `configs/experiments/bread/three_bakery.json`, 추론 정책은
`configs/experiments/bread/three_bakery_runtime.json`에 저장한다. 두 검출기(SSDLite320,
D-FINE HGNetV2-S640), 세 분류법(frozen cosine head, 마지막 stage 미세조정+192/224 consistency,
앞 구성+normalized margin), 두 합성 방식(1,200/6,000장)의 12개 구성을 비교한다.

SSDLite 20 epoch·LR 3e-4, D-FINE 12 epoch·head 1e-4·backbone 1e-5,
분류기 8 epoch·head 1e-3·backbone 3e-6·consistency 0.2·margin 가중치 4.0·목표 0.9를 사용한다.
평가로 epoch를 선택하지 않고 마지막 epoch를 쓴다. 승인 margin 0.8, detail/verifier 상한 0.85를
고정하며 이번 데이터로 임계값을 탐색하지 않는다. 빈 장면은 합성의 정확히 20%, 양성 객체 수는 1~8개다.

기본 seed 20260907로 12개 구성을 비교하고, 상위 2개는 20260908·20260909로 반복한다.
오승인 최소 → 실제 멀티 완전 성공 최대 → 공통 합성 진단 완전 성공 최대 → 검출 누락 최소 →
UNKNOWN Top-3 누락 최소 → CUDA p95 순서다. 세 seed 중 최악, 중앙 결과, 구성 ID 순서로
선택하고 선택 구성의 기본 seed payload를 최종 평가한다. 공통 합성 진단 400장은 seed 20261907이다.

이는 같은 실물에 대한 개발 진단이다. 독립 검증으로 입증된 최적 학습법이라는 주장을 하지 않는다.

## 원본과 annotation

### 2026-09-08 사용자 멀티 검수 확정

사용자가 검수 화면의 멀티 50장(source 201–250), 객체 272개에 대해
“검수 완료 확정해줘”라고 확인했다. source 223의 객체 4는 에이전트가 C10 도넛으로 잘못
기록했으며, 사용자가 확인한 C07 꽃빵(`bread_07`)으로 정정했다. 확정 범위는 화면에 표시된
멀티 객체의 품목과 보이는 bbox다. 단일·배경 원본 및 SAM 마스크 전체까지 사용자 검수했다고
확대 해석하지 않는다.

사용자 확인과 원본·annotation·수정 이력의 해시는
`configs/experiments/bread/three_bakery_user_review.json`에 기록한다. 확정 데이터 revision은
`artifacts/retraining/three-bakery-reviewed/reviews/20260908-multi50-user-confirmed/annotations.jsonl`이며,
SHA-256은 `32259b6c335416dbdbb8c0585a307bed5e347221e4e862a6368820c40efab78c`다.
이 252장 snapshot 중 사용자 검수 확정 대상은 멀티 50장이다. 확정된 crop 라벨과 prompt도
같은 디렉터리에 보존했다. 마스크의 상대 경로 기준은 기존 `sources` 디렉터리다.

기존 학습·평가 산출물은 수정 전 라벨로 실행된 이력으로 유지한다. 사용자 검수 확정만으로
학습 모델이나 Catalog가 수정된 것으로 간주하지 않는다. 위 사용자 검수 기록은 데이터 revision의
근거이며, SAM draft 검수용 `--review` 파일 형식이나 완료된 학습 캐시가 아니다.

### 기존 생성·에이전트 검수 이력

`artifacts/retraining/three-bakery-reviewed/sources/originals.jsonl`은 252장의 SHA-256, 원본 크기,
품목, 관련 그룹을 기록한다. `capture_session_id`의 날짜는 이번 수집 묶음의 식별자이며 EXIF로
확인한 촬영 시각이 아니다. 모든 사진을 하나의 관련 그룹으로 유지하고 사진 단위 fold를 만들지 않는다.
단일 폴더 정답 200개와 멀티 육안 정답 272개로 총 472개 객체를 준비했다.

일반 SAM ViT-B의 bbox prompt로 원본 해상도의 마스크를 만들었다. 단일 200장의 overlay,
멀티 50장의 원본 및 수정 overlay를 전수 확인했고 자동 proposal이 잘못 잡은 단일 29번을 수정했다.
단일 class board 20개와 멀티 객체 board 23개를 원본 픽셀 크기로 추가 검수하여 bbox 27개에서
종이·받침대·이웃 객체를 제외했다. 수정 좌표와 원본/SAM proposal 해시는
`configs/experiments/bread/three_bakery_annotation_review.json`에 기록한다.
초기 `three-bakery` 폴더의 모델은 수정 전 annotation으로 학습했으므로 후보 비교에서 제외하며,
`three-bakery-reviewed`의 모든 학습 모델은 일반 사전학습 가중치부터 다시 학습한다.
검수 주체는 에이전트이며 별도 사람의 독립 검수로 표현하지 않는다. 정확한 검수 해상도와 방법은
`sources/visual-review.json`에 기록한다. 좌표 범위·source/draft/mask checksum 검사를 병행한다.

받침대·종이가 섞인 단일 마스크 24개 및 멀티 마스크는 합성 cutout으로 사용하지 않는다.
해당 원본과 crop은 학습에서 제외하지 않는다. 분류기는 472개 원본 crop을 사용하고,
검출기는 빈 배경 2장을 포함한 252장 모두를 매 epoch 실제로 읽었는지 검사한다.
SSDLite에는 빈 배경에서도 양수 classification loss가 발생하는 hard-negative loss를 적용한다.

모든 crop과 합성 장면은 부모 원본 해시를 가진다. 실제 배경 2장 또는 그 색상으로 만든 절차적 배경을
양성·음성 장면 모두에 사용한다. 원본·mask·합성·학습 출력은 Git에 넣지 않는다.

## 실행

일반 PyTorch 학습 환경과 ONNX Runtime CPU/CUDA Worker 환경을 분리한다.
학습 전에 SAM, COCO SSDLite/D-FINE, DINOv3 ConvNeXt-Tiny·ViT-B/16 일반 사전학습 가중치를 준비한다.
이전 빵 데이터로 학습한 checkpoint, teacher 출력, adapter, prototype, Catalog는 입력하지 않는다.

```powershell
python -m bixolon_scanner.experiments.bread.three_bakery prepare --work artifacts/retraining/three-bakery-reviewed
python -m bixolon_scanner.experiments.bread.three_bakery train --work artifacts/retraining/three-bakery-reviewed
python -m bixolon_scanner.experiments.bread.three_bakery compare --work artifacts/retraining/three-bakery-reviewed
python -m bixolon_scanner.experiments.bread.three_bakery export --work artifacts/retraining/three-bakery-reviewed
python -m bixolon_scanner.experiments.bread.three_bakery evaluate --work artifacts/retraining/three-bakery-reviewed
```

새 source 작업 디렉터리에서 `prepare`는 검수 자료를 만든 후 hash-bound review JSON이 필요하다.
`--review`로 전수 검수 결과를 지정한다. 나머지 경로와 Python interpreter는 `--help`의 인자로 변경한다.
`compare`는 기본 12개 진단 후 상위 두 구성의 추가 seed 학습과 진단도 실행한다.
단계별 명령·시간·로그와 모델별 config/원본/annotation/가중치/code/output 해시를 보존한다.
완료된 학습은 입력과 출력 해시를 확인하고 재사용하며, 중단된 학습은 epoch 경계에서 재개한다.

## Runtime과 평가 계약

정상 ROI 전체 → ConvNeXt-Tiny 192 primary → 선택된 ROI만 224 detail → 전역 경계 조건의
Frozen DINOv3 ViT-B/16 160 verifier 순서를 유지한다. frozen 검증 경로는 제거하지 않았다.
일반 frozen backbone은 그대로 사용하고 이번 crop으로 verifier Ridge adapter·support를 새로 만든다.
Catalog는 품목별 고유한 객체 crop 10개로 구성한다. 단일 및 멀티 원본 crop에서 perceptual 중복을
제거한 후 hash 순으로 선택하므로 Catalog의 일부 crop만 쓰더라도 전량 학습 요구에는 영향을 주지 않는다.
checksum Runtime·Catalog는 `candidates/<구성>-<seed>`에 만들며 기존 제품 bundle을 덮어쓰지 않는다.
학습된 primary ONNX의 해시를 `embedder_id`에 포함하여 다른 후보의 Catalog를 잘못 조합하면
기존 runtime compatibility 검사에서 거부한다.

최종 선택은 `final-candidate.json`의 `candidate_path`에서 확인한다. 해당 payload를 CPU Worker로
실행하는 예시는 다음과 같다. CUDA에서는 `BIXOLON_PROVIDER=cuda`, `BIXOLON_CUDA_DLL_DIR`을 설정하고
ONNX Runtime GPU 환경의 Python을 사용한다. 학습 환경의 PyTorch는 Worker에 필요하지 않다.

```powershell
$selected = Get-Content artifacts/retraining/three-bakery-reviewed/final-candidate.json -Raw | ConvertFrom-Json
$env:BIXOLON_PACKAGE_DIR = Join-Path $selected.candidate_path 'runtime'
$env:BIXOLON_CATALOG_DIR = Join-Path $selected.candidate_path 'catalog'
$env:BIXOLON_CATALOG_STORE_ID = 'three_bakery'
$env:BIXOLON_PROVIDER = 'cpu'
python -c 'from bixolon_scanner.worker.cli import serve; serve()'
```

`final-candidate.json`에 후보와 정책 해시를 기록한 뒤에만 최종
`datasets/bread_dataset/multi_object_scenes`의 300장·1,410개 GT를 이번 실험 평가 함수에서 읽는다.
이 평가 함수의 접근 이후 재학습·후보 선택은 orchestration에서 차단한다. 과거 평가 이력이 있으므로
최종 결과는 고정 벤치마크 성적이다.

**사전 접근 이력:** 후보 확정 전에 실행한 저장소 전체 테스트의 기존 `test_bread_dataset.py`와
`test_bread_cv.py`가 이 300장의 이미지와 GT를 읽고 데이터 계약·해시·기존 fold 구성을 검사했다.
따라서 파일 접근 자체를 후보 확정 뒤로 완전히 제한한 실행은 아니다. 해당 테스트의 이미지·GT·
생성 manifest는 이번 학습 입력, 증강 설계, 후보 비교, 임계값 결정에 사용하지 않았다.
이번 실험에서는 기존 fold를 사용하지 않으며, 252장 source manifest와 실제 학습 입력 이력을
별도로 검증한다. 이 이력은 `reproduction/prefreeze-benchmark-access.json`에도 남긴다.
추가 저장소 전체 테스트는 후보 확정 뒤에 수행한다. 이 제한을 숨기거나 독립 검증으로 표현하지 않는다.

공개 `/v1/scan`의 segmentation과 GT를 클래스 정보 없이 bbox IoU≥0.5로 일대일 연결한다.
최대 대응 수를 우선하고 동률은 총 IoU로 해소한다. 모든 GT에 정답 APPROVED가 정확히 하나씩 있고
추가 segmentation이 없는 이미지만 완전 성공이다. 오분류 승인과 미대응 배경·중복 승인을 모두 센다.
UNKNOWN·재촬영·ERROR·검출 누락을 분모에서 빼지 않는다.

CPU와 CUDA 각각 warmup 후 3회 HTTP 측정을 수행하며 매 회 300장을 분모로 유지한다.
반복별 정확도와 통합 지연 p50·p95·p99·표본 수, full-path·detector 조기 종료를 구분한다.
양쪽 provider 모두 각 반복에서 완전 성공 ≥297/300 및 오승인 0건이어야 목표 달성이다.
속도는 합격 조건이 아니다. 달성하더라도 “해당 300장에서 관측된 오승인 0건”으로만 보고한다.

## 실행 결과

- 원본 252장, 객체 472개 원본 크기 검수 및 bbox 27개 수정 완료.
- 수정된 정답으로 합성 1,200/6,000장과 진단 400장을 생성했다. 기본 12개 구성 및 상위 2개의 추가 seed 2회씩 비교를 완료했다.
- 공유 모델을 포함한 학습 18건의 모든 epoch 입력 이력과 출력 해시 검사를 통과했다. 검출기는 원본 252장, 분류기는 foreground 250장의 전체 객체 crop 472개를 사용했다.
- 최종 선택은 `dfine-margin-dense`, payload seed는 `20260907`이다. Frozen ViT-B/16 160 선택 검증을 포함한 새 Runtime/Catalog를 조립했다.
- 수정된 annotation과 일치하지 않는 crop·학습 모델은 재사용 검사에서 거부한다.
- 전량 사용·파생물·bbox·오승인·분모·평가 접근 제한 관련 테스트 25개 통과.
- 후보 확정 및 최종 평가 후 전체 Python 테스트 1,106개 통과. 사전 평가셋 접근 이력은 위 제한 절 참조. Ruff check/format 및 `git diff --check` 통과.
- Flutter analyze: 두 앱 모두 통과. product_scanner 193개, bakery_scanner_lite 24개 테스트 통과.
- PyTorch/ORT 원본 252장 및 ORT CPU/CUDA 최종 300장의 공개 상태·품목 순위 불일치는 0장이다. 분류기·ViT tensor는 허용 오차를 충족했다.
- D-FINE 원시 tensor의 직접 배열 비교는 허용 오차를 충족하지 않았다. 저장 입력 13개 추가 분석에서 낮은 점수 후보 순서 차이를 확인했으며, 고정 검출 임계값 이상 후보는 수가 같고 bbox 대응 후 정규화 좌표 최대 차이는 PyTorch `8.94e-8`, CUDA `1.20e-7`이었다. 원시 tensor 실패 기록은 유지한다.

CPU·CUDA 각각 3회 실제 `/v1/scan` 평가 결과는 모두 **완전 성공 264/300장(88.0%), 오승인 3건**으로 목표에 미달했다.
오승인은 품목 오류 2건과 중복 승인 1건이다. 정답 승인 객체는 1,361/1,410개(96.52%), 검출 누락 5개,
추가 segmentation 3개, UNKNOWN 43개, segment 재촬영 1개, IMAGE_RECAPTURE 1장, ERROR 0장이다.
완전 성공 목표 297장보다 33장이 부족하며, 최종 결과에 맞춘 재학습이나 임계값 변경은 하지 않았다.

Full-path HTTP p95는 CPU 368.25 ms, CUDA 117.91 ms이며 각 897개 표본이다.
최종 결과·난이도·실패 이미지·지연 전체 통계는 작업 디렉터리의 `final/report.md`,
전체 응답은 `final/{cpu,cuda}/responses.jsonl`, 선택 근거는 `selection.json`에 보존했다.
오승인 사례는 이미지 233(중복), 260(`bread_12`→`bread_19`), 267(`bread_01`→`bread_04`)이다.

## 설계 참고

Frozen 기준선과 미세조정 비교는 [DINOv3 모델 카드](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md),
검출기는 [D-FINE 공식 구현](https://github.com/Peterande/D-FINE), 마스크 합성은
[Copy-Paste 연구](https://openaccess.thecvf.com/content/CVPR2021/html/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.html)를 참고했다.
이 자료들이 이번 252장에 대한 최적 구성이나 목표 성적을 보장하지는 않는다.
