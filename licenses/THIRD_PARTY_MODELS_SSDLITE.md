# SSDLite diagnostic Runtime third-party notices

이 고지는 YOLO-free `0.1.7` SSDLite 진단 Runtime 후보에 적용된다.

## torchvision SSDLite320 MobileNetV3-Large

- Project: `pytorch/vision`
- Source: <https://github.com/pytorch/vision>
- Architecture implementation: `ssdlite320_mobilenet_v3_large`
- License: BSD 3-Clause
- Pretrained weights: 사용하지 않음
- Training data: 프로젝트 소유 실데이터 및 프로젝트에서 생성한 합성 데이터

`TORCHVISION-LICENSE.txt`를 Runtime과 함께 배포한다. 이 후보는 torchvision이 배포한
사전학습 가중치를 포함하지 않는다.

## DINOv3

Classifier와 independent classifier verifier는 기존 `0.1.7` Runtime의 DINOv3 모델을 그대로 사용하며,
`DINOV3-LICENSE.md`의 조건을 따른다.

이 기록은 최종 배포 주체의 법무 검토를 대체하지 않는다.
