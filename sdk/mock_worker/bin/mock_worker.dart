import 'dart:convert';
import 'dart:io';

const _version = '0.1.14';
const _allowedStatuses = <String>{
  'approved',
  'unknown',
  'segment-recapture',
  'image-recapture',
  'error',
};

Future<void> main(List<String> arguments) async {
  final port = int.parse(_argument(arguments, '--port') ?? '8000');
  final status = _argument(arguments, '--status') ?? 'approved';
  if (!_allowedStatuses.contains(status)) {
    stderr.writeln('Unsupported --status: $status');
    stderr.writeln('Allowed: ${_allowedStatuses.join(', ')}');
    exitCode = 64;
    return;
  }

  final server = await HttpServer.bind(InternetAddress.loopbackIPv4, port);
  stdout.writeln('BIXOLON mock Worker: http://127.0.0.1:$port ($status)');
  await for (final request in server) {
    await _handle(request, status);
  }
}

String? _argument(List<String> arguments, String name) {
  final index = arguments.indexOf(name);
  if (index < 0 || index + 1 >= arguments.length) return null;
  return arguments[index + 1];
}

Future<void> _handle(HttpRequest request, String status) async {
  final path = request.uri.path;
  if (request.method == 'GET' && path == '/health/live') {
    _json(request.response, HttpStatus.ok, <String, Object>{'status': 'live'});
    return;
  }
  if (request.method == 'GET' && path == '/health/ready') {
    _json(request.response, HttpStatus.ok, <String, Object?>{
      'status': 'ready',
      'provider': 'openvino',
      ..._versions(),
    });
    return;
  }
  if (request.method == 'POST' && path == '/v1/scan') {
    await request.drain<void>();
    final responseStatus = status == 'error'
        ? HttpStatus.internalServerError
        : HttpStatus.ok;
    _json(request.response, responseStatus, _scanResponse(status));
    return;
  }
  _json(request.response, HttpStatus.notFound, <String, Object>{
    'message': 'Not found',
  });
}

Map<String, Object?> _versions({bool detectorOnly = false}) =>
    <String, Object?>{
      'worker_version': _version,
      'detector_version': _version,
      'classifier_version': detectorOnly ? null : _version,
      'embedder_version': detectorOnly ? null : _version,
      'detector_policy_version': _version,
      'classifier_policy_version': detectorOnly ? null : _version,
      'catalog_version': detectorOnly ? null : _version,
    };

Map<String, Object?> _scanResponse(String status) {
  final base = <String, Object?>{
    'request_id': 'mock-${DateTime.now().microsecondsSinceEpoch}',
    'processing_time_ms': 12.34,
  };
  if (status == 'image-recapture') {
    return <String, Object?>{
      ...base,
      'status': 'IMAGE_RECAPTURE',
      'reason_codes': <String>['IMAGE_RECAPTURE_REQUIRED'],
      'segmentations': <Object>[],
      ..._versions(detectorOnly: true),
    };
  }
  if (status == 'error') {
    return <String, Object?>{
      ...base,
      'status': 'ERROR',
      'reason_codes': <String>['MOCK_SYSTEM_ERROR'],
      'segmentations': <Object>[],
      ..._versions(),
    };
  }

  final segmentStatus = switch (status) {
    'approved' => 'APPROVED',
    'unknown' => 'UNKNOWN',
    _ => 'SEGMENT_RECAPTURE',
  };
  final isApproved = segmentStatus == 'APPROVED';
  final isUnknown = segmentStatus == 'UNKNOWN';
  return <String, Object?>{
    ...base,
    'status': 'SEGMENTATION',
    'reason_codes': isUnknown
        ? <String>['SEGMENT_BELOW_APPROVAL_THRESHOLD']
        : <String>[],
    'segmentations': <Object>[
      <String, Object?>{
        'segmentation_id': 'segmentation_001',
        'bbox': <String, int>{'x': 120, 'y': 84, 'width': 310, 'height': 246},
        'status': segmentStatus,
        'reason_codes': isUnknown
            ? <String>['BELOW_APPROVAL_THRESHOLD']
            : segmentStatus == 'SEGMENT_RECAPTURE'
            ? <String>['CLASSIFIER_QUALITY_RECAPTURE']
            : <String>[],
        'prediction': isApproved
            ? <String, String>{
                'class_id': 'bread_06',
                'class_name': 'Croissant',
              }
            : null,
        'top3': isUnknown
            ? <Object>[
                <String, Object>{
                  'class_id': 'bread_02',
                  'class_name': 'Croffle',
                  'confidence': 0.521,
                },
                <String, Object>{
                  'class_id': 'bread_03',
                  'class_name': 'Waffle',
                  'confidence': 0.312,
                },
                <String, Object>{
                  'class_id': 'bread_19',
                  'class_name': 'Pastry Bread',
                  'confidence': 0.167,
                },
              ]
            : <Object>[],
        'confidence': isApproved
            ? 0.987
            : isUnknown
            ? 0.521
            : 0.0,
      },
    ],
    ..._versions(),
  };
}

void _json(HttpResponse response, int status, Map<String, Object?> value) {
  response
    ..statusCode = status
    ..headers.contentType = ContentType.json
    ..write(jsonEncode(value))
    ..close();
}
