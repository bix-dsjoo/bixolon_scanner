import 'dart:convert';

import 'package:flutter/services.dart';

import '../models/scan_models.dart';

class ProductCatalog {
  ProductCatalog._(this.products);

  factory ProductCatalog.fromJsonBody(String body) {
    final decoded = jsonDecode(body);
    if (decoded is! Map<String, dynamic> || decoded['products'] is! List) {
      throw const FormatException('상품 카탈로그 형식이 올바르지 않습니다.');
    }
    if (decoded['schema_version'] != '1.0') {
      throw const FormatException('지원하지 않는 상품 카탈로그 schema_version입니다.');
    }
    final products = (decoded['products'] as List)
        .map((value) {
          if (value is! Map<String, dynamic>) {
            throw const FormatException('상품 카탈로그 항목이 올바르지 않습니다.');
          }
          final classId = value['class_id'];
          final className = value['class_name'];
          if (classId is! String ||
              classId.isEmpty ||
              className is! String ||
              className.isEmpty) {
            throw const FormatException('상품 카탈로그 필수 필드가 누락되었습니다.');
          }
          return Product(
            classId: classId,
            className: className,
            displayName: className,
          );
        })
        .toList(growable: false);
    if (products.isEmpty ||
        products.map((product) => product.classId).toSet().length !=
            products.length) {
      throw const FormatException('상품 카탈로그가 비어 있거나 class_id가 중복됩니다.');
    }
    return ProductCatalog._(products);
  }

  static Future<ProductCatalog> load({AssetBundle? bundle}) async {
    final body = await (bundle ?? rootBundle).loadString(
      'assets/catalog/bread_ko.json',
    );
    return ProductCatalog.fromJsonBody(body);
  }

  final List<Product> products;

  Product localize(Product product) {
    final matched = products.where(
      (candidate) => candidate.classId == product.classId,
    );
    if (matched.isEmpty) return product;
    return product.withDisplayName(matched.first.className);
  }

  Candidate localizeCandidate(Candidate candidate) {
    final matched = products.where(
      (product) => product.classId == candidate.classId,
    );
    if (matched.isEmpty) return candidate;
    return candidate.withDisplayName(matched.first.className);
  }

  List<Product> search(String query) {
    final normalized = query.trim().toLowerCase();
    if (normalized.isEmpty) return products;
    return products
        .where(
          (product) =>
              product.displayName.toLowerCase().contains(normalized) ||
              product.className.toLowerCase().contains(normalized) ||
              product.classId.toLowerCase().contains(normalized),
        )
        .toList(growable: false);
  }
}
