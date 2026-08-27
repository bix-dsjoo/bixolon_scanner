# Third-party model notices

이 파일은 BIXOLON Scanner `0.1.5` Runtime에 포함되는 모델 계열의 출처와 재배포 고지를 기록한다.

## YOLO26

- Project: `ultralytics/ultralytics`
- Source: <https://github.com/ultralytics/ultralytics>
- Installed release used for training and export: `8.4.102`
- Model: `YOLO26n`, one-to-many head, one-class objectness fine-tuning
- License: GNU Affero General Public License v3.0 or later (`AGPL-3.0-or-later`)
- Use: 상품 label과 무관한 단일 Detector ONNX

`AGPL-3.0.txt`를 Runtime과 최종 번들에 포함한다. 네트워크 서비스 제공과 수정본 배포를 포함한
AGPL 의무를 최종 배포 주체가 준수해야 하며, 이 고지는 상용 라이선스를 부여하지 않는다.

## DINOv3

- Project: `facebookresearch/dinov3`
- Source: <https://github.com/facebookresearch/dinov3>
- Model: DINOv3 ConvNeXt-Tiny, revision `6876159a11b4df116f30f667f8c9888617df0751`
- License: DINOv3 License
- Use: `single_objects_3` 기반 Catalog embedding과 Ridge 분류

`DINOV3-LICENSE.md`를 Runtime과 함께 배포한다. DINOv3의 사용·재배포는 Apache-2.0이 아니라 해당
Meta 계약을 따른다.

## Apache License 2.0

저장소와 최종 번들에서 사용하는 Apache-2.0 고지는 `APACHE-2.0.txt`로 함께 배포한다. 이 기술
기록은 최종 배포 주체의 법무 검토를 대체하지 않는다.
