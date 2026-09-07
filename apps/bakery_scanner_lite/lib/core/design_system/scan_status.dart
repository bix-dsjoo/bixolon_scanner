import 'package:flutter/material.dart';
import 'tokens.dart';

Color scanStatusColor(String? status) => switch (status) {
  'APPROVED' => AppPalette.success,
  'ERROR' || 'IMAGE_RECAPTURE' || 'SEGMENT_RECAPTURE' => AppPalette.error,
  'UNKNOWN' => AppPalette.attention,
  _ => AppPalette.muted,
};
