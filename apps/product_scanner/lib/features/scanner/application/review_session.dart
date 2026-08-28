import '../../../shared/models/scan_models.dart';

class ReviewSession {
  final List<ReviewSnapshot> _undoStack = <ReviewSnapshot>[];
  final List<ReviewSnapshot> _redoStack = <ReviewSnapshot>[];
  ReviewSnapshot? _activeBoxEditStart;

  bool get canUndo => _undoStack.isNotEmpty;
  bool get canRedo => _redoStack.isNotEmpty;
  bool get hasActiveBoxEdit => _activeBoxEditStart != null;

  void beginBoxEdit(ReviewSnapshot snapshot) {
    _activeBoxEditStart ??= snapshot;
  }

  ReviewSnapshot? finishBoxEdit() {
    final snapshot = _activeBoxEditStart;
    _activeBoxEditStart = null;
    return snapshot;
  }

  void record(ReviewSnapshot snapshot) {
    _undoStack.add(snapshot);
    _redoStack.clear();
  }

  ReviewSnapshot? undo(ReviewSnapshot current) {
    if (_undoStack.isEmpty) return null;
    _redoStack.add(current);
    return _undoStack.removeLast();
  }

  ReviewSnapshot? redo(ReviewSnapshot current) {
    if (_redoStack.isEmpty) return null;
    _undoStack.add(current);
    return _redoStack.removeLast();
  }

  void clear() {
    _undoStack.clear();
    _redoStack.clear();
    _activeBoxEditStart = null;
  }
}

class ReviewSnapshot {
  const ReviewSnapshot({
    required this.detections,
    required this.selectedItemId,
    required this.operatorRequiresRecapture,
    required this.reviewIssueCodes,
  });

  final List<ReviewDetection> detections;
  final String? selectedItemId;
  final bool operatorRequiresRecapture;
  final Set<OperatorIssueCode> reviewIssueCodes;
}
