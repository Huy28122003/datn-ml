import '../../repositories/virus_total_repository.dart';

class SubmitUrlUseCase {
  final VirusTotalRepository _repository;

  SubmitUrlUseCase(this._repository);

  Future<String> call(String url) {
    return _repository.submitUrl(url);
  }
}
