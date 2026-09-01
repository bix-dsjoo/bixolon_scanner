# Third-party model notices

이 파일은 BIXOLON Bakery AI Scanner `0.1.12` Runtime에 포함되는 모델 계열의 출처와 재배포 고지를
기록한다.

## torchvision SSDLite320 MobileNetV3-Large

- Project: `pytorch/vision`
- Source: <https://github.com/pytorch/vision>
- Architecture/revision: SSDLite320 MobileNetV3-Large, torchvision `v0.28.0`
- License: BSD 3-Clause
- Weight source: 외부 pretrained weight 없이 프로젝트 데이터로 random initialization부터 학습
- Use: 20개 빵 class-aware Detector ONNX

`TORCHVISION-LICENSE.txt`를 Runtime과 함께 배포한다. `0.1.12` Runtime은 YOLO 코드·weight를
포함하지 않으며 detector에 AGPL 의존성이 없다.

## DINOv3

- Project: `facebookresearch/dinov3`
- Source: <https://github.com/facebookresearch/dinov3>
- Models: DINOv3 ConvNeXt-Tiny와 ViT-B/16, revision
  `6876159a11b4df116f30f667f8c9888617df0751`
- License: DINOv3 License
- Use: Catalog embedding, Ridge 분류와 독립 classifier verifier

`DINOV3-LICENSE.md`를 Runtime과 함께 배포하며 사용·재배포는 해당 Meta 계약을 따른다. 이 기술
고지는 최종 배포 주체의 법무 검토를 대체하지 않는다.
