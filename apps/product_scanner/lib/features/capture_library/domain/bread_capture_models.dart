import 'dart:collection';

enum BreadCaptureSide {
  normal('normal', '정상 면'),
  flipped('flipped', '뒤집은 면');

  const BreadCaptureSide(this.fileToken, this.label);

  final String fileToken;
  final String label;
}

enum BreadCapturePose {
  topLeft(
    'ground_30_dir_01',
    '왼쪽 상단',
    '빵의 기본 방향을 유지하고 3×3 기준 왼쪽 상단 구역에 놓아 주세요.',
    '좌상',
  ),
  topRight(
    'ground_30_dir_02',
    '오른쪽 상단',
    '빵의 기본 방향을 유지하고 3×3 기준 오른쪽 상단 구역에 놓아 주세요.',
    '우상',
  ),
  center('vertical', '중앙', '빵의 기본 방향을 유지하고 트레이 중앙에 놓아 주세요.', '중앙'),
  bottomLeft(
    'ground_30_dir_03',
    '왼쪽 하단',
    '빵의 기본 방향을 유지하고 3×3 기준 왼쪽 하단 구역에 놓아 주세요.',
    '좌하',
  ),
  bottomRight(
    'ground_30_dir_04',
    '오른쪽 하단',
    '빵의 기본 방향을 유지하고 3×3 기준 오른쪽 하단 구역에 놓아 주세요.',
    '우하',
  );

  const BreadCapturePose(
    this.fileToken,
    this.label,
    this.guidance,
    this.shortLabel,
  );

  final String fileToken;
  final String label;
  final String guidance;
  final String shortLabel;
}

class BreadCaptureSlot {
  const BreadCaptureSlot({required this.side, required this.pose});

  final BreadCaptureSide side;
  final BreadCapturePose pose;

  String get key => '${side.fileToken}_${pose.fileToken}';
  String get label => '${side.label} · ${pose.label}';

  String fileName(int categoryId) =>
      'bread_${categoryId.toString().padLeft(2, '0')}_$key.jpg';

  static final all = List<BreadCaptureSlot>.unmodifiable([
    for (final side in BreadCaptureSide.values)
      for (final pose in BreadCapturePose.values)
        BreadCaptureSlot(side: side, pose: pose),
  ]);

  static BreadCaptureSlot? fromKey(String key) {
    for (final slot in all) {
      if (slot.key == key) return slot;
    }
    return null;
  }
}

class BreadCaptureProduct {
  BreadCaptureProduct({
    required this.categoryId,
    required this.name,
    required this.directoryName,
    required this.createdAt,
    Map<String, String> captures = const {},
  }) : captures = UnmodifiableMapView(Map<String, String>.from(captures));

  final int categoryId;
  final String name;
  final String directoryName;
  final DateTime createdAt;
  final Map<String, String> captures;

  String get code => 'bread_${categoryId.toString().padLeft(2, '0')}';
  int get completedCaptureCount => captures.length;
  bool get isComplete => completedCaptureCount == BreadCaptureSlot.all.length;

  bool hasCapture(BreadCaptureSlot slot) => captures.containsKey(slot.key);
  String? capturePath(BreadCaptureSlot slot) => captures[slot.key];

  BreadCaptureSlot? get firstIncompleteSlot {
    for (final slot in BreadCaptureSlot.all) {
      if (!hasCapture(slot)) return slot;
    }
    return null;
  }

  BreadCaptureProduct copyWith({Map<String, String>? captures}) {
    return BreadCaptureProduct(
      categoryId: categoryId,
      name: name,
      directoryName: directoryName,
      createdAt: createdAt,
      captures: captures ?? this.captures,
    );
  }
}

class BreadCaptureLibrarySnapshot {
  const BreadCaptureLibrarySnapshot({
    required this.rootPath,
    required this.products,
  });

  final String rootPath;
  final List<BreadCaptureProduct> products;

  int get completedProductCount =>
      products.where((product) => product.isComplete).length;
  int get capturedImageCount => products.fold(
    0,
    (total, product) => total + product.completedCaptureCount,
  );
}
