import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as paths;
import 'package:product_scanner/features/capture_library/data/bread_capture_repository.dart';
import 'package:product_scanner/features/capture_library/domain/bread_capture_models.dart';

void main() {
  test('10개 촬영 슬롯은 합성기 입력 파일명과 같은 계약을 사용한다', () {
    expect(BreadCaptureSlot.all, hasLength(10));
    expect(
      BreadCaptureSlot.all.map((slot) => slot.fileName(3)),
      containsAll(<String>[
        'bread_03_normal_ground_30_dir_01.jpg',
        'bread_03_normal_ground_30_dir_02.jpg',
        'bread_03_normal_ground_30_dir_03.jpg',
        'bread_03_normal_ground_30_dir_04.jpg',
        'bread_03_normal_vertical.jpg',
        'bread_03_flipped_ground_30_dir_01.jpg',
        'bread_03_flipped_ground_30_dir_02.jpg',
        'bread_03_flipped_ground_30_dir_03.jpg',
        'bread_03_flipped_ground_30_dir_04.jpg',
        'bread_03_flipped_vertical.jpg',
      ]),
    );
  });

  test('기존 single_objects 폴더를 불러오고 추가·촬영·삭제를 관리한다', () async {
    final temporary = await Directory.systemTemp.createTemp(
      'bread-capture-repository-',
    );
    addTearDown(() => temporary.delete(recursive: true));
    final existing = Directory(
      paths.join(temporary.path, 'bread_01_walnut_donut'),
    );
    await existing.create();
    for (final slot in BreadCaptureSlot.all) {
      await File(
        paths.join(existing.path, slot.fileName(1)),
      ).writeAsBytes([1, 2, 3]);
    }
    final repository = FileBreadCaptureRepository.forRoot(temporary.path);

    var snapshot = await repository.load();

    expect(snapshot.products, hasLength(1));
    expect(snapshot.products.single.name, 'walnut donut');
    expect(snapshot.products.single.isComplete, isTrue);

    final product = await repository.addProduct('크림 소보로');
    expect(product.categoryId, 2);
    expect(product.directoryName, 'bread_02_크림_소보로');

    final firstSlot = BreadCaptureSlot.all.first;
    await repository.saveCapture(
      product,
      firstSlot,
      Uint8List.fromList([9, 8, 7]),
    );
    snapshot = await repository.load();
    final updated = snapshot.products.singleWhere((row) => row.categoryId == 2);
    expect(updated.completedCaptureCount, 1);
    expect(File(updated.capturePath(firstSlot)!).readAsBytesSync(), [9, 8, 7]);

    await repository.deleteCapture(updated, firstSlot);
    expect((await repository.load()).products.last.completedCaptureCount, 0);

    await repository.deleteProduct(product);
    snapshot = await repository.load();
    expect(snapshot.products.map((row) => row.categoryId), [1]);
    expect(
      File(paths.join(temporary.path, 'capture_catalog.json')).existsSync(),
      isTrue,
    );
  });
}
