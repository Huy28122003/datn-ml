import '../models/virus_total_report.dart';

abstract class VirusTotalRepository {
  Future<String> submitUrl(String url);
  Future<VirusTotalReport> getAnalysis(String analysisId);
}
