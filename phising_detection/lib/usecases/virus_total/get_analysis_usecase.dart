import '../../models/virus_total_report.dart';
import '../../repositories/virus_total_repository.dart';

class GetAnalysisUseCase {
  final VirusTotalRepository _repository;

  GetAnalysisUseCase(this._repository);

  Future<VirusTotalReport> call(String analysisId) {
    return _repository.getAnalysis(analysisId);
  }
}
