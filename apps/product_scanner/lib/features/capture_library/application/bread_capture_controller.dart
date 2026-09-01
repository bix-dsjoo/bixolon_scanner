import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';

import '../../../shared/input/image_input.dart';
import '../data/bread_capture_repository.dart';
import '../domain/bread_capture_models.dart';

class BreadCaptureController extends ChangeNotifier {
  BreadCaptureController(
    this._repository,
    this._cameraGateway,
    this._directoryPicker,
  );

  final BreadCaptureRepository _repository;
  final CameraGateway _cameraGateway;
  final BreadCaptureDirectoryPicker _directoryPicker;

  BreadCaptureLibrarySnapshot? library;
  int? selectedProductId;
  BreadCaptureSlot selectedSlot = BreadCaptureSlot.all.first;
  Uint8List? pendingImageBytes;
  bool loading = false;
  bool cameraInitializing = false;
  bool capturing = false;
  bool saving = false;
  bool changingRoot = false;
  String? errorMessage;
  String? completionMessage;

  bool get isBusy =>
      loading || cameraInitializing || capturing || saving || changingRoot;
  bool get isCameraReady => _cameraGateway.isReady;
  CameraController? get cameraController => _cameraGateway.controller;
  List<BreadCaptureProduct> get products => library?.products ?? const [];
  bool get canAddProduct => products.length < 20 && !isBusy;

  BreadCaptureProduct? get selectedProduct {
    for (final product in products) {
      if (product.categoryId == selectedProductId) return product;
    }
    return null;
  }

  Future<void> initialize() async {
    if (library != null || loading) return;
    loading = true;
    errorMessage = null;
    notifyListeners();
    try {
      await _reload();
    } on Object {
      errorMessage = '촬영 목록을 불러오지 못했어요. 저장 폴더를 확인해 주세요.';
    } finally {
      loading = false;
      notifyListeners();
    }
  }

  Future<void> activate() async {
    await initialize();
    if (_cameraGateway.isReady || cameraInitializing) return;
    cameraInitializing = true;
    errorMessage = null;
    notifyListeners();
    try {
      await _cameraGateway.initialize();
    } on Object {
      errorMessage = '카메라를 사용할 수 없어요. 연결 상태를 확인해 주세요.';
    } finally {
      cameraInitializing = false;
      notifyListeners();
    }
  }

  Future<void> reconnectCamera() async {
    if (cameraInitializing || capturing) return;
    cameraInitializing = true;
    errorMessage = null;
    notifyListeners();
    try {
      await _cameraGateway.initialize();
    } on Object {
      errorMessage = '카메라를 다시 연결하지 못했어요.';
    } finally {
      cameraInitializing = false;
      notifyListeners();
    }
  }

  void selectProduct(int categoryId) {
    if (selectedProductId == categoryId) return;
    selectedProductId = categoryId;
    pendingImageBytes = null;
    completionMessage = null;
    final product = selectedProduct;
    selectedSlot = product?.firstIncompleteSlot ?? BreadCaptureSlot.all.first;
    notifyListeners();
  }

  void selectSlot(BreadCaptureSlot slot) {
    if (selectedSlot.key == slot.key && pendingImageBytes == null) return;
    selectedSlot = slot;
    pendingImageBytes = null;
    completionMessage = null;
    notifyListeners();
  }

  Future<void> addProduct(String name) async {
    if (!canAddProduct) return;
    saving = true;
    errorMessage = null;
    completionMessage = null;
    notifyListeners();
    try {
      final product = await _repository.addProduct(name);
      await _reload(preferredProductId: product.categoryId);
      selectedSlot = BreadCaptureSlot.all.first;
      completionMessage = '${product.name}을(를) 추가했어요.';
    } on ArgumentError catch (error) {
      errorMessage = error.message?.toString() ?? '빵 이름을 확인해 주세요.';
    } on StateError catch (error) {
      errorMessage = error.message;
    } on Object {
      errorMessage = '빵을 추가하지 못했어요. 저장 폴더를 확인해 주세요.';
    } finally {
      saving = false;
      notifyListeners();
    }
  }

  Future<void> deleteSelectedProduct() async {
    final product = selectedProduct;
    if (product == null || saving) return;
    saving = true;
    errorMessage = null;
    completionMessage = null;
    notifyListeners();
    try {
      await _repository.deleteProduct(product);
      await _reload();
      completionMessage = '${product.name}과(와) 촬영 사진을 삭제했어요.';
    } on Object {
      errorMessage = '빵을 삭제하지 못했어요. 폴더 사용 권한을 확인해 주세요.';
    } finally {
      saving = false;
      notifyListeners();
    }
  }

  Future<void> capture() async {
    if (selectedProduct == null || capturing || saving) return;
    if (!_cameraGateway.isReady) {
      await reconnectCamera();
      if (!_cameraGateway.isReady) return;
    }
    capturing = true;
    errorMessage = null;
    completionMessage = null;
    notifyListeners();
    try {
      final image = await _cameraGateway.capture();
      pendingImageBytes = image.bytes;
    } on CameraException {
      errorMessage = '촬영하지 못했어요. 카메라를 확인하고 다시 찍어 주세요.';
    } on Object {
      errorMessage = '촬영 이미지를 가져오지 못했어요.';
    } finally {
      capturing = false;
      notifyListeners();
    }
  }

  void discardPendingCapture() {
    if (pendingImageBytes == null || saving) return;
    pendingImageBytes = null;
    completionMessage = null;
    notifyListeners();
  }

  Future<void> acceptPendingCapture() async {
    final product = selectedProduct;
    final bytes = pendingImageBytes;
    if (product == null || bytes == null || saving) return;
    final savedSlot = selectedSlot;
    saving = true;
    errorMessage = null;
    completionMessage = null;
    notifyListeners();
    try {
      await _repository.saveCapture(product, savedSlot, bytes);
      pendingImageBytes = null;
      await _reload(preferredProductId: product.categoryId);
      final updated = selectedProduct;
      selectedSlot = updated?.firstIncompleteSlot ?? savedSlot;
      completionMessage = updated?.isComplete == true
          ? '${updated!.name}의 10장 촬영을 완료했어요.'
          : '${savedSlot.label} 사진을 저장했어요.';
    } on Object {
      errorMessage = '사진을 저장하지 못했어요. 저장 폴더 용량과 권한을 확인해 주세요.';
    } finally {
      saving = false;
      notifyListeners();
    }
  }

  Future<void> deleteCapture(BreadCaptureSlot slot) async {
    final product = selectedProduct;
    if (product == null || saving || !product.hasCapture(slot)) return;
    saving = true;
    errorMessage = null;
    completionMessage = null;
    notifyListeners();
    try {
      await _repository.deleteCapture(product, slot);
      await _reload(preferredProductId: product.categoryId);
      selectedSlot = slot;
      completionMessage = '${slot.label} 사진을 삭제했어요.';
    } on Object {
      errorMessage = '사진을 삭제하지 못했어요.';
    } finally {
      saving = false;
      notifyListeners();
    }
  }

  Future<void> chooseRoot() async {
    if (changingRoot || saving) return;
    final selected = await _directoryPicker.pick();
    if (selected == null || selected.trim().isEmpty) return;
    changingRoot = true;
    errorMessage = null;
    completionMessage = null;
    pendingImageBytes = null;
    notifyListeners();
    try {
      library = await _repository.changeRoot(selected);
      _selectAvailableProduct();
      completionMessage = '촬영 저장 폴더를 변경했어요.';
    } on Object {
      errorMessage = '선택한 폴더를 사용할 수 없어요.';
    } finally {
      changingRoot = false;
      notifyListeners();
    }
  }

  Future<void> openRoot() async {
    try {
      await _repository.openRoot();
    } on Object {
      errorMessage = '저장 폴더를 열지 못했어요.';
      notifyListeners();
    }
  }

  void clearMessage() {
    if (errorMessage == null && completionMessage == null) return;
    errorMessage = null;
    completionMessage = null;
    notifyListeners();
  }

  Future<void> _reload({int? preferredProductId}) async {
    final previous = preferredProductId ?? selectedProductId;
    library = await _repository.load();
    _selectAvailableProduct(preferredProductId: previous);
  }

  void _selectAvailableProduct({int? preferredProductId}) {
    final available = products;
    if (available.isEmpty) {
      selectedProductId = null;
      selectedSlot = BreadCaptureSlot.all.first;
      return;
    }
    final preferredExists = available.any(
      (product) => product.categoryId == preferredProductId,
    );
    selectedProductId = preferredExists
        ? preferredProductId
        : available.first.categoryId;
    final product = selectedProduct;
    if (product != null && !product.hasCapture(selectedSlot)) return;
    selectedSlot = product?.firstIncompleteSlot ?? BreadCaptureSlot.all.first;
  }
}
