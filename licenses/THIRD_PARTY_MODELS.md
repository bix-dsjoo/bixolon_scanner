# Third-party model notices

이 파일은 Scanner 2.0 Runtime package에 포함되는 모델 계열의 출처와 재배포 고지를 잠근다.

## D-FINE

- Project: `Peterande/D-FINE`
- Source: <https://github.com/Peterande/D-FINE>
- License: Apache License 2.0
- Use: 네 개의 locally trained/exported Detector ONNX checkpoint
- Packaged SHA-256:
  - `a3879b68dfec6bea9a43e6a0378b97ae3c9f1a5db79cd9e5bdc6e1cc82ef14e8`
  - `aed824b7b1544b76dd16704f896e6996d8e71004871026822d6bf2f50f2926a3`
  - `cf5685136023dc31e25f856970510a3af2e9f8cefbe3deee3a1782f11f62e5e2`
  - `f0d2eaf8e67821627957c3eed1462812063c32c4ad17028dda869addc5371b09`

## DINOv2

- Project: `facebookresearch/dinov2`
- Source: <https://github.com/facebookresearch/dinov2>
- Model: `dinov2-base`, revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`
- License: Apache License 2.0 for code and model weights
- Source weight SHA-256: `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`
- Packaged ONNX SHA-256: `669d290dc235d5d9d336657c9b723bb589fca3b9bc32dd9648d2ae8f1ffeac78`

## DINOv3

- Project: `facebookresearch/dinov3`
- Source: <https://github.com/facebookresearch/dinov3>
- Model: `DINOv3 ViT-B/16`, revision `6876159a11b4df116f30f667f8c9888617df0751`
- License: DINOv3 License
- Source weight SHA-256: `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`
- Use: frozen embedding backbone for Scanner `0.0.1`

`DINOV3-LICENSE.md`를 DINOv3 파생 Runtime package와 함께 배포한다. DINOv3의 사용·재배포는
Apache-2.0이 아니라 해당 Meta 계약을 따른다.

`APACHE-2.0.txt`가 이 notice와 함께 배포된다. 이 기술 검토 기록은 최종 배포 주체의 법무 승인을
대체하지 않는다.
