/// Content tokens for high-impact actions and progress states.
///
/// The same operation must keep the same label across buttons, dialogs, and
/// recovery states so operators can predict the result before activating it.
abstract final class AppActionCopy {
  static const chooseImage = '이미지 선택';
  static const chooseAnotherImage = '다른 이미지 선택';
  static const capture = '촬영하기';
  static const recapture = '다시 촬영';
  static const analyze = '분석하기';
  static const analyzing = '분석 중';
  static const analyzingAnnouncement = '분석 중. 완료될 때까지 기다려 주세요';
  static const reanalyze = '다시 분석';
  static const reconnect = '다시 연결';
  static const checkingConnection = '연결 확인 중';
  static const checkingConnectionAnnouncement = '카메라 연결 확인 중. 완료될 때까지 기다려 주세요';
  static const refresh = '새로고침';
  static const refreshing = '새로고침 중';
  static const refreshingAnnouncement = '활동 기록 새로고침 중. 완료될 때까지 기다려 주세요';
  static const resetAll = '모두 초기화';
  static const retrySave = '다시 저장';
  static const saving = '저장 중';
  static const savingAnnouncement = '저장 중. 완료될 때까지 기다려 주세요';
}

/// Content tokens that identify the image shown in the scan preview.
///
/// The preview label names the input source only. Workflow state belongs to the
/// panel header, guidance belongs to the message body, and the next operation
/// belongs to the action bar.
abstract final class AppPreviewCopy {
  static const liveCamera = '라이브 카메라';
  static const cameraPreview = '카메라 미리보기';
  static const capturedImage = '촬영 이미지';
  static const selectedImage = '선택한 이미지';

  static String semanticLabel(String source) => '입력 미리보기, $source';
}

/// Content tokens used when a saved Activity record cannot provide a value.
///
/// Stored technical values remain untouched; only the operator-facing label is
/// replaced so legacy or future log variants do not expose internal sentinels.
abstract final class AppActivityCopy {
  static const productUnavailable = '상품 정보 없음';
  static const confirmationMethodUnavailable = '확정 방식 확인 불가';
}
