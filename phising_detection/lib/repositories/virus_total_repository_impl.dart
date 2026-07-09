import '../models/virus_total_report.dart';
import '../services/virus_total_service.dart';
import 'virus_total_repository.dart';

class VirusTotalRepositoryImpl implements VirusTotalRepository {
  final VirusTotalService _service;

  VirusTotalRepositoryImpl({VirusTotalService? service})
      : _service = service ?? VirusTotalService();

  @override
  Future<String> submitUrl(String url) async {
    final id = await _service.submitUrl(url);
    if (id == null || id.isEmpty) {
      throw Exception('Received an empty analysis ID from VirusTotal service');
    }
    return id;
  }

  @override
  Future<VirusTotalReport> getAnalysis(String analysisId) async {
    final rawReport = await _service.getAnalysis(analysisId);
    return VirusTotalReport.fromJson(rawReport);
  }
}
