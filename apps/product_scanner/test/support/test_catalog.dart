import 'package:product_scanner/catalog/product_catalog.dart';

final testCatalog = ProductCatalog.fromJsonBody('''
{
  "schema_version": "1.0",
  "products": [
    {"class_id":"bread_03","class_name":"Waffle","display_name_ko":"와플"},
    {"class_id":"bread_04","class_name":"Scon","display_name_ko":"스콘"},
    {"class_id":"bread_06","class_name":"Croissant","display_name_ko":"크루아상"},
    {"class_id":"bread_11","class_name":"Bagel","display_name_ko":"베이글"},
    {"class_id":"bread_12","class_name":"Egg Tart","display_name_ko":"에그 타르트"},
    {"class_id":"bread_13","class_name":"Muffin","display_name_ko":"머핀"}
  ]
}
''');
