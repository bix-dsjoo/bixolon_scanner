# Third-party model notices

이 파일은 YOLO-free `0.1.7` 진단 Runtime에 포함되는 모델 계열의 출처와 재배포 고지를 기록한다.

## D-FINE / HGNetv2

- Project: `Peterande/D-FINE`
- Source: <https://github.com/Peterande/D-FINE>
- Revision: `267a6da`
- Model: D-FINE-N, HGNetv2 backbone, four-model fixed ensemble
- License: Apache License 2.0
- Use: 객체 proposal 검출과 class-aware 보조 신호

`D-FINE-LICENSE.txt`와 `APACHE-2.0.txt`를 Runtime과 최종 번들에 포함한다.

## DINOv3

- Project: `facebookresearch/dinov3`
- Source: <https://github.com/facebookresearch/dinov3>
- Model: DINOv3 ConvNeXt-Tiny / ViT-S/16 계열
- License: DINOv3 License
- Use: Catalog embedding, 분류 검증, exact-count 보조 신호

`DINOV3-LICENSE.md`를 Runtime과 함께 배포한다. Detector의 Apache-2.0 전환은 별도 DINOv3
라이선스 의무를 제거하지 않는다.

이 기술 기록은 최종 배포 주체의 법무 검토를 대체하지 않는다.
