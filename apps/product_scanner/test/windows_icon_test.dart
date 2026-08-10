import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';

void main() {
  test('Windows 아이콘은 Orange 포커스 마크의 9개 크기를 포함한다', () {
    final bytes = File(
      'windows/runner/resources/app_icon.ico',
    ).readAsBytesSync();
    final data = ByteData.sublistView(bytes);

    expect(data.getUint16(0, Endian.little), 0);
    expect(data.getUint16(2, Endian.little), 1);
    expect(data.getUint16(4, Endian.little), 9);

    final sizes = <int>[];
    Uint8List? largestPng;
    for (var index = 0; index < 9; index += 1) {
      final entry = 6 + index * 16;
      final encodedWidth = bytes[entry];
      final encodedHeight = bytes[entry + 1];
      final width = encodedWidth == 0 ? 256 : encodedWidth;
      final height = encodedHeight == 0 ? 256 : encodedHeight;
      final byteCount = data.getUint32(entry + 8, Endian.little);
      final offset = data.getUint32(entry + 12, Endian.little);

      expect(height, width);
      expect(data.getUint16(entry + 6, Endian.little), 32);
      expect(offset + byteCount, lessThanOrEqualTo(bytes.length));
      expect(bytes.sublist(offset, offset + 8), <int>[
        137,
        80,
        78,
        71,
        13,
        10,
        26,
        10,
      ]);
      sizes.add(width);
      if (width == 256) {
        largestPng = Uint8List.sublistView(bytes, offset, offset + byteCount);
      }
    }

    expect(sizes, <int>[16, 20, 24, 32, 40, 48, 64, 128, 256]);
    expect(largestPng, isNotNull);
    final image = _decodeGeneratedPng(largestPng!);
    expect(image.pixel(0, 0), <int>[0, 0, 0, 0]);
    expect(image.pixel(128, 32), <int>[0xEE, 0x72, 0x03, 0xFF]);
    expect(image.pixel(128, 128), <int>[0x17, 0x17, 0x17, 0xFF]);
  });
}

_DecodedPng _decodeGeneratedPng(Uint8List png) {
  final data = ByteData.sublistView(png);
  var offset = 8;
  var width = 0;
  var height = 0;
  final compressed = BytesBuilder(copy: false);
  while (offset < png.length) {
    final length = data.getUint32(offset);
    final type = ascii.decode(png.sublist(offset + 4, offset + 8));
    final payloadStart = offset + 8;
    if (type == 'IHDR') {
      width = data.getUint32(payloadStart);
      height = data.getUint32(payloadStart + 4);
    } else if (type == 'IDAT') {
      compressed.add(png.sublist(payloadStart, payloadStart + length));
    } else if (type == 'IEND') {
      break;
    }
    offset = payloadStart + length + 4;
  }
  final raw = Uint8List.fromList(zlib.decode(compressed.takeBytes()));
  return _DecodedPng(width, height, raw);
}

class _DecodedPng {
  const _DecodedPng(this.width, this.height, this.bytes);

  final int width;
  final int height;
  final Uint8List bytes;

  List<int> pixel(int x, int y) {
    expect(x, inInclusiveRange(0, width - 1));
    expect(y, inInclusiveRange(0, height - 1));
    final rowLength = 1 + width * 4;
    expect(bytes[y * rowLength], 0, reason: 'generator uses PNG filter 0');
    final offset = y * rowLength + 1 + x * 4;
    return bytes.sublist(offset, offset + 4);
  }
}
