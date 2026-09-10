# N100 구조 개선 조사와 실험 설계

조사일: 2026-09-09. 이 문서는 문헌 조사와 설계 제안이다. 아래 새 모델을 학습하거나 N100에서
측정한 결과가 아니다. R3의 실행 최적화와 별도로 **무거운 계산량 자체를 줄이는 모델 개발**을
우선한다. 목표는 로그132 정답 승인 1078/1096 이상, 오승인·미검출 0, 재촬영 최소화와 HTTP
전 요청 1000ms 이내다. CPU detector → GPU classifier → CPU verifier 및 CPU fallback을 보존한다.

## 현재 문제를 다시 정의한다

반환된 N100 결과에서 detector는 약 300ms, primary는 static1 기준 ROI당 약 42ms,
detail은 약 52ms, CPU verifier는 FP32 약 158ms / INT8 약 89ms였다. 장면에 따라 호출 수가 다르다.
병렬 실행이 있으므로 모델 시간의 합을 HTTP 시간으로 취급하지 않는다.
R2 batch2의 HTTP p95 920.7ms, 최대1203.3ms, 130/132장 1초 이내는 목표에 여유가 부족하다.
근거: [실측 보고서](../../artifacts/n100/optimization-r3/REPORT-KO.md).

현재 비용은 대략 `검출 + ROI 수 × 무거운 분류 + 선택적 detail/verifier + 기타`다.
Worker 호출 비용을 줄여도 ROI 수에 비례하는 backbone 계산은 남는다.
따라서 우선 바꿀 것은 스레드 수가 아니라 **ROI당 모델 비용과 전체 장면에서 반복하는 특징 추출**이다.

연산량만으로 모델을 고르지도 않는다. Roofline 관점에서는 메모리 이동·연산 처리량 중 제한되는
요소가 성능을 결정한다. 작은 FLOPs, 적은 파라미터, 모바일 벤치마크가 N100 저지연을 보장하지
않는다. [Berkeley Lab Roofline 설명](https://amcr.lbl.gov/departments/computer-science-department/ppan/roofline-performance-model/).

## 1순위: 현재 DINO를 teacher로 쓰고, 실행 분류기는 작은 student로 바꾼다

**제안:** ConvNeXt-Tiny 192 primary를 RepViT 또는 FastViT 계열의 작은 encoder로 교체하는
cross-architecture distillation이다. 기존 모델의 정답 class뿐 아니라 상품 간 유사도, 혼동 쌍,
촬영 품질 및 여러 객체가 섞인 ROI 증거도 학습한다. 우선 primary만 교체하여 원인을 분리하고,
224 detail과 독립 ViT verifier는 대조 경로로 보존한다.

2026년 MIDL의 DINOv3 domain-constrained distillation은 큰 ViT의 표현을 CNN에 옮기는 실험을
보여준다. 초음파 분야 결과이므로 빵 분류에서의 효과나 N100 속도 근거는 아니다.
[논문](https://proceedings.mlr.press/v315/nahian26a.html).
FastViT는 구조적 재매개변수화로 실행 시 메모리 접근을 줄이는 설계를 제시한다.
[Apple 연구 설명](https://machinelearning.apple.com/research/fastvit).
MobileCLIP2의 2025년 연구는 teacher 조합과 증류 데이터 설계의 중요성을 보여주지만,
그 대형 image-text 모델을 이 제품에 그대로 넣자는 제안은 아니다.
[Apple MobileCLIP2](https://machinelearning.apple.com/research/mobileclip2).

실험은 다음과 같이 분리한다.

- 작은 student의 일반 supervised fine-tuning, logit 증류, logit+관계 증류를 같은 데이터·seed로 비교한다.
- GT 지도 신호를 유지하고 teacher가 틀린 답을 그대로 정답으로 복제하지 않는다.
- 원본 support/crop과 실제 detector crop, 품질 불량, 경계 접촉, 다중 객체 ROI를 학습 분포에 포함한다.
- student마다 Catalog support/prototype을 다시 계산한다. 기존 embedding이나 승인 threshold를
  그대로 재사용하지 않고 별도 validation으로 보정한다.
- 작은 student와 같은 teacher에서 만든 두 head를 독립 검증기라고 부르지 않는다.
  초기에는 기존 독립 verifier를 유지하고 조건부 오류의 상관관계를 측정한다.

**탐색 목표:** 11개 정상 ROI의 primary 계산을 절반 가까이 줄일 수 있는 모델을 찾는다.
static1의 `11×42≈462ms`가 가령 절반이 되면 이 부분의 절감 여지는 약231ms다.
이는 산술적 가정이며 HTTP 개선 예측이나 N100 측정 결과가 아니다.
학생 모델 때문에 detail/verifier 호출이 늘어 이익이 사라지면 탈락이다.

증류 자체를 성공으로 가정하지 않는다. 2026년 7월 raptor 분류 preprint는 통제 비교에서
증류나 DINOv3 teacher 교체가 명확한 이점을 주지 않은 결과도 보고한다. 따라서
**증류 없는 동일 student**가 필수 대조군이다.
[반대 증거 논문](https://arxiv.org/abs/2607.26238).

## 2순위: detector를 작은 모델로 재학습하고, 검출 품질을 먼저 평가한다

활성 detector ONNX를 직접 검사하면 약1021만 parameter와 backbone/decoder 구조가 있다.
metadata의 source architecture 문자열에는 과거 `ssdlite_class_agnostic` 표기가 남아 있으므로
새 실험의 provenance는 모델 hash·graph·실제 학습 설정으로 확인해야 한다.

제품 실험의 첫 대조 후보는 **D-FINE-N 640**이다. 현재 입력 해상도를 유지하여 작은 객체 정보
손실과 backbone 축소의 영향을 분리한다. 공식 모델 표에서 N은 약4M parameter/7 GFLOPs,
S는 약10M/25 GFLOPs지만 이는 N100 속도 비율이 아니다.
[D-FINE 공식 구현](https://github.com/Peterande/D-FINE).

현재 detector의 box/objectness와 GT를 이용해 작은 detector를 학습하고, 경계·겹침·부분 객체를
포함한 training hard negative로 검출 자체를 개선한다. teacher가 만든 추가 검출까지 복제하지
않도록 GT와 대조한다. detector class를 SKU 승인에 사용하지 않는다. 프레임 품질 조기 종료와
그때 classifier를 호출하지 않는 계약을 유지한다.

재촬영의 일부가 추가 ROI에서 생기는 만큼 정확한 위치 검출은 **속도와 재촬영 수를 함께 개선할
가능성**이 있다. 하지만 실제로 존재하는 객체를 누락하여 얻은 속도나 ROI 감소는 개선이 아니다.
동일 일대일 GT matching으로 misses, extras와 최종 판정까지 확인한다.

최신 연구로는 DEIMv2(2025), EdgeCrafter(2026)가 관련 있다. 특히 EdgeCrafter의 2026-08-16
수정본은 작업에 맞춘 증류와 작은 backbone/encoder/decoder 설계를 다룬다.
[EdgeCrafter 논문](https://arxiv.org/abs/2603.18739v4),
[DEIMv2 논문](https://arxiv.org/abs/2509.20787).

두 프로젝트를 바로 배포 후보로 확정하지는 않는다. EdgeCrafter의 공개 지연은 NVIDIA T4의
TensorRT FP16 조건이며 N100 결과가 아니다.
[공식 모델 표](https://github.com/Intellindust-AI-Lab/EdgeCrafter).
현재 배포 라이선스에는 상업적 사용에 별도 라이선스가 필요하다고 명시되어 있다.
[EdgeCrafter LICENSE](https://github.com/Intellindust-AI-Lab/EdgeCrafter/blob/main/LICENSE.md),
[DEIMv2 LICENSE](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/LICENSE.md).
따라서 연구 근거로 참고하되, 현재 제품 실험에는 기존 허용 구성과 D-FINE의 작은 구조를 우선한다.
새 pretrained checkpoint의 데이터·weight 조건은 코드 라이선스와 별도로 확인한다.

**탐색 목표:** detector 300ms를 150~180ms 수준으로 줄일 수 있는지 확인한다.
이 수치는 모델 선택용 설계 목표이며 달성이 확인된 값이 아니다.

## 3순위: detector의 공간 특징을 재사용하는 분류 경로

가장 큰 구조 변경은 이미 계산한 detector 공간 특징에서 ROI 특징을 뽑아 작은 SKU embedding
head에 전달하는 것이다. 비용을 `K번의 큰 backbone`에서 `공유 backbone + K번의 작은 head`로
바꾸려는 접근이다. SKU 결정은 별도 classifier/Catalog가 담당하고 detector class로 직접 승인하지
않는다. 작은/겹친/불확실 ROI만 원본 crop classifier로 보내는 후보를 별도로 검증한다.

전체 이미지 계산을 공유한다는 근거는 R-FCN에도 있다.
[Microsoft Research R-FCN](https://www.microsoft.com/en-us/research/publication/r-fcn-object-detection-via-region-based-fully-convolutional-networks/).
이것은 오래된 이론적 출발점이며 그 논문의 가속 수치를 이 제품에 적용하지 않는다.

**이미 실패한 실험과의 차이를 명확히 해야 한다.** 저장소의 DINOv3+FCOS 공유 표현 실험은
group-aware OOF에서 GT1410개 중28개를 놓쳤고, dense 특징과 crop identity 간 괴리도 있었다.
[기존 실험](adaptive-cascade-dinov3.md). 단순 ROI pooling이나 새 detector를 처음부터 재학습하는
방식의 반복은 후순위다. 현재 검출 위치를 우선 고정하고, 그 위치의 공간 특징을 crop teacher
embedding에 맞추는 분류 전용 실험부터 한다. stride·ROIAlign·이웃 객체 mask 차이를 따로 검증한다.

전체 이미지용 별도 DINO를 추가하는 방식은 오히려 느릴 수 있다. 같은 연산 밀도를 단순 가정하면
640² 픽셀은 약11.1개의 192² crop에 해당하고, 1024²는 약28.4개다. **공유한다는 이름만으로
추론량이 줄지는 않는다.** 기존 detector 특징의 재사용 또는 훨씬 작은 encoder가 중요하다.
이 경로는 pipeline 입력 계약·metadata·학습 변경이 필요한 연구 후보이며 즉시 교체 대상은 아니다.

## 4순위: detector와 student에 정확도 제약이 있는 static INT8/QAT 적용

R2는 CPU verifier의 MatMul 양자화였다. 새로운 실험은 비중이 큰 detector와 student의
weight·activation을 다루고, 민감한 연산은 FP32로 남기는 것이다.
OpenVINO 2026 문서의 accuracy-control API는 OpenVINO 및 ONNX ModelProto 입력을 지원하며
영향이 큰 연산을 원래 정밀도로 보존한다. 따라서 최종 ONNX와 ONNX Runtime 계약을 유지하는
실험이 가능하다. 현재 설치된 구버전 도구에 해당 기능이 있다고 가정하지 않고 별도 환경에서 검증한다.
[NNCF accuracy control](https://docs.openvino.ai/2026/openvino-workflow/model-optimization-guide/quantizing-models-post-training/quantizing-with-accuracy-control.html).

평균 Top-1이나 mAP만으로 양자화 레이어를 고르지 않는다. 별도 validation에서
오승인·미검출 증가, 기존 정답 승인 손실, 품질 판정 손실을 분리한 feasibility 조건을 적용한다.
개선2개와 악화2개가 상쇄되어도 개별 변화 목록을 유지한다. `max_drop=0`만으로 미래 입력의
오류 0을 보장하지 않는다. PTQ가 이를 만족하지 못할 때에만 QAT+증류를 다음 대조군으로 둔다.

최신 기술 블로그의 INT4/FP4, sparse attention, KV cache 사례 대부분은 다른 모델·가속기 조건이다.
[OpenVINO 기술 동향](https://blog.openvino.ai/blog-posts/q425-technology-update---low-precision-and-model-optimization).
GPU INT8 지원은 장비 capability와 실제 실행 graph로 확인해야 한다.
[OpenVINO 2026 GPU 문서](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html).
N100에서 이득을 확인하기 전에는 FP16 GPU 경로를 INT8보다 열등하다고 판단하지 않는다.

## Worker 개선은 추론 엔진 경계를 측정한 뒤 범위를 정한다

같은 모델·동일 입력 tensor로 ORT CPU, ORT-OpenVINO GPU, 진단용 native OpenVINO의
compile/warmup/steady-state, graph partition과 연산별 시간을 비교한다. native 경로는 병목 진단용이며
즉시 제품 Worker backend를 교체한다는 뜻이 아니다.
ORT가 이미 C++ 연산을 호출하는데 Python Worker 전체를 C++로 옮기는 것만으로 backbone의
수백 ms가 사라지지 않는다. kernel·메모리 이동이 지배적이면 모델 변경이 먼저다.

ORT-OpenVINO에는 ORT graph optimization을 끄고 EP가 최적화하게 하는 권고가 있다.
이는 현재 `runtime/onnx_session.py`에 이미 반영되어 있어 새로운 개선안으로 재포장하지 않는다.
[ORT 공식 문서](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html).
throughput 최적화와 한 장의 지연도 구분한다. 다중 Worker·stream은 개별 요청의 1초 목표와
메모리 사용을 악화시킬 수 있다.
[Intel GPU 성능 기술 블로그](https://blog.openvino.ai/blog-posts/techniques-for-faster-ai-inference-throughput-with-openvino-on-intel-gpus).

## 실험 순서와 판단 기준

| 단계 | 실험 | 다음 단계로 넘기는 근거 |
|---|---|---|
| A0 | RepViT·FastViT 작은 encoder의 ONNX export와 CPU/GPU tensor benchmark | ROI 1/4/8/12개 실제 지연·peak memory에서 절감 여지 확인; 합성 입력은 정확도 증거로 사용하지 않음 |
| A1 | 빠른 student 2개 × supervised / 증류 | group-aware validation에서 오류·정답 승인·품질/다중 객체 판정을 유지하고 full-path 비용 감소 |
| B1 | D-FINE-N 640, 기존 모델과 비교; GT+teacher 학습 | 검출 miss·extra와 최종 scan 결과를 함께 통과 |
| Q1 | 유망한 모델에 static INT8와 accuracy restoration; 필요 시 QAT | 실제 ONNX CPU/GPU 결과와 개별 판정 변화 확인 |
| C1 | 현재 detector 특징의 ROI embedding 증류 | crop 분류 대비 SKU/품질 증거를 유지하면서 객체 수에 따른 지연 증가율 감소 |
| N1 | 살아남은 구조 후보 2~3개만 N100에서 반복 | HTTP p50/p95/p99/max와 1초 초과 수, 실제 provider, 4GB 메모리, CPU fallback 확인 |

모델 이름만 많이 늘린 N100 매트릭스보다, 서로 다른 구조적 가설을 검증하는 후보를 만든다.
validation에서 후보를 고른 뒤 로그132는 고정 회귀 평가에 사용한다. 반복 사용한 로그132와
final300을 새 독립 test라고 부르지 않는다. 가능한 학습·검증 분할의 물리 객체/capture session
중복을 먼저 확인하며, 신규 독립 session은 일반화 진단에 별도로 필요하다.

설계 예산의 예는 detector180ms + primary(다수 ROI)220ms + detail/verifier180ms +
decode/후처리/HTTP120ms = **700ms**다. 이는 1초 경계에 운영 변동 여유를 두기 위한 목표이며,
측정한 p95끼리 더한 수치도 미래 성능 예측도 아니다. 최종 판단은 한 요청 전체의 실측으로 한다.

이번 조사에서 구현·학습·새 N100 측정은 수행하지 않았다. 우선 개발할 대안은
**작은 분류기 증류와 작은 detector의 GT 기반 재학습**이며, 공유 특징 경로는 큰 잠재력과
기존 실패 근거를 함께 가진 다음 연구 축으로 둔다.
