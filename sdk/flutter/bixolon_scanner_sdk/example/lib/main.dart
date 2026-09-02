import 'package:bixolon_scanner_sdk/bixolon_scanner_sdk.dart';
import 'package:flutter/material.dart';

void main() => runApp(const SdkExampleApp());

class SdkExampleApp extends StatelessWidget {
  const SdkExampleApp({super.key});

  @override
  Widget build(BuildContext context) => const MaterialApp(home: ConnectPage());
}

class ConnectPage extends StatefulWidget {
  const ConnectPage({super.key});

  @override
  State<ConnectPage> createState() => _ConnectPageState();
}

class _ConnectPageState extends State<ConnectPage> {
  BixolonScannerSession? _session;
  String _message = '연결되지 않음';

  Future<void> _connectExisting() => _connect(BixolonScannerSession.connect);

  Future<void> _startManagedLocal() =>
      _connect(BixolonScannerSession.startLocal);

  Future<void> _connect(Future<BixolonScannerSession> Function() create) async {
    setState(() => _message = '연결 중...');
    try {
      final next = await create();
      _session?.close();
      setState(() {
        _session = next;
        _message = '준비 완료: ${next.readiness.versions.worker}';
      });
    } on BixolonScannerException catch (error) {
      setState(() => _message = '${error.code}: ${error.message}');
    }
  }

  @override
  void dispose() {
    _session?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('BIXOLON Scanner SDK')),
    body: Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(_message),
          const SizedBox(height: 16),
          FilledButton(
            onPressed: _connectExisting,
            child: const Text('기존 Worker에 연결'),
          ),
          const SizedBox(height: 8),
          OutlinedButton(
            onPressed: _startManagedLocal,
            child: const Text('로컬 Worker 시작 후 연결'),
          ),
        ],
      ),
    ),
  );
}
