import '../../../shared/models/scan_models.dart';

class RecapturePresentation {
  const RecapturePresentation({
    required this.reasonCode,
    required this.title,
    required this.detail,
  });

  final String? reasonCode;
  final String title;
  final String detail;
}

RecapturePresentation presentRecaptureReasons({
  required List<String> reasonCodes,
  required InputMode inputMode,
}) {
  final reason = _reasonPriority.firstWhere(
    reasonCodes.contains,
    orElse: () => '',
  );
  final guidance = switch (reason) {
    'DETECTOR_NO_OBJECT' => (
      title: '상품을 찾지 못했어요',
      cameraDetail: '상품이 화면 안에 모두 보이도록 위치를 조정해 주세요.',
      imageDetail: '상품이 화면 안에 보이는 다른 이미지를 선택해 주세요.',
    ),
    'DETECTOR_CAPACITY_EXCEEDED' => (
      title: '상품이 너무 많거나 겹쳐 보여요',
      cameraDetail: '상품 사이를 벌리고 모두 보이도록 배치해 주세요.',
      imageDetail: '상품이 겹치지 않고 모두 보이는 다른 이미지를 선택해 주세요.',
    ),
    'DETECTOR_OBJECT_TOO_SMALL' => (
      title: '상품이 너무 작게 보여요',
      cameraDetail: '카메라를 가까이 옮겨 상품을 크게 보여 주세요.',
      imageDetail: '상품이 더 크게 보이는 다른 이미지를 선택해 주세요.',
    ),
    'DETECTOR_BORDER_CLIPPED' => (
      title: '상품 일부가 잘렸어요',
      cameraDetail: '잘린 상품을 화면 안쪽으로 옮겨 주세요.',
      imageDetail: '상품 전체가 이미지 안에 있는 다른 이미지를 선택해 주세요.',
    ),
    'DETECTOR_COUNT_MISMATCH' ||
    'DETECTOR_COUNT_UNCERTAIN' ||
    'DETECTOR_UNCERTAIN_OBJECT' => (
      title: '상품 수를 확인하기 어려워요',
      cameraDetail: '상품 사이를 벌리고 모두 보이도록 배치해 주세요.',
      imageDetail: '상품이 서로 떨어져 모두 보이는 다른 이미지를 선택해 주세요.',
    ),
    'DETECTOR_BLUR' => (
      title: '이미지가 흔들렸어요',
      cameraDetail: '카메라를 고정하고 상품이 선명하게 보이도록 조정해 주세요.',
      imageDetail: '상품이 선명한 다른 이미지를 선택해 주세요.',
    ),
    'DETECTOR_UNDEREXPOSED' => (
      title: '이미지가 너무 어두워요',
      cameraDetail: '더 밝은 곳으로 옮기거나 조명을 조정해 주세요.',
      imageDetail: '밝게 촬영된 다른 이미지를 선택해 주세요.',
    ),
    'DETECTOR_OVEREXPOSED' => (
      title: '이미지가 너무 밝아요',
      cameraDetail: '빛 반사를 줄이거나 조명을 조정해 주세요.',
      imageDetail: '빛 반사가 적은 다른 이미지를 선택해 주세요.',
    ),
    'CLASSIFIER_QUALITY_CLASS' || 'CLASSIFIER_QUALITY_REJECTED' => (
      title: '상품 상태를 확인하기 어려워요',
      cameraDetail: '상품 앞면과 모양이 선명하도록 방향을 조정해 주세요.',
      imageDetail: '상품 모양이 선명한 다른 이미지를 선택해 주세요.',
    ),
    _ => (
      title: inputMode == InputMode.camera
          ? '촬영 조건을 다시 확인해 주세요'
          : '다른 이미지를 선택해 주세요',
      cameraDetail: '상품이 화면 안에 선명하게 보이도록 위치를 조정해 주세요.',
      imageDetail: '상품이 잘 보이는 다른 이미지를 선택해 주세요.',
    ),
  };
  return RecapturePresentation(
    reasonCode: reason.isEmpty
        ? (reasonCodes.isEmpty ? null : reasonCodes.first)
        : reason,
    title: guidance.title,
    detail: inputMode == InputMode.camera
        ? guidance.cameraDetail
        : guidance.imageDetail,
  );
}

const _reasonPriority = <String>[
  'DETECTOR_NO_OBJECT',
  'DETECTOR_CAPACITY_EXCEEDED',
  'DETECTOR_OBJECT_TOO_SMALL',
  'DETECTOR_BORDER_CLIPPED',
  'DETECTOR_COUNT_MISMATCH',
  'DETECTOR_COUNT_UNCERTAIN',
  'DETECTOR_UNCERTAIN_OBJECT',
  'DETECTOR_BLUR',
  'DETECTOR_UNDEREXPOSED',
  'DETECTOR_OVEREXPOSED',
  'CLASSIFIER_QUALITY_CLASS',
  'CLASSIFIER_QUALITY_REJECTED',
];
