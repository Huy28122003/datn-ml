import 'dart:async';
import '../../models/virus_total_report.dart';
import '../../repositories/virus_total_repository.dart';

class ScanUrlUseCase {
  final VirusTotalRepository _repository;

  ScanUrlUseCase(this._repository);

  /// Submits the [url] for analysis, then polls until the status is 'completed'.
  /// Throws a [TimeoutException] if the [timeout] is exceeded.
  Future<VirusTotalReport> call(
    String url, {
    Duration pollInterval = const Duration(seconds: 2),
    Duration timeout = const Duration(minutes: 2),
  }) async {
    final id = await _repository.submitUrl(url);
    final startTime = DateTime.now();

    while (DateTime.now().difference(startTime) < timeout) {
      final report = await _repository.getAnalysis(id);
      if (report.isCompleted) {
        return report;
      }
      await Future.delayed(pollInterval);
    }
    
    throw TimeoutException('VirusTotal analysis timed out for URL: $url');
  }
}
