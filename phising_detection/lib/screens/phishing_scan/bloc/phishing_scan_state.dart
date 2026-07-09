import '../../../models/prediction_result.dart';

enum PhishingScanStatus {
  initial,
  loadingModel,
  loadModelFailure,
  ready,
  scanning,
  success,
  failure
}

class PhishingScanState {
  final PhishingScanStatus status;
  final PhishingPredictionResult? result;
  final String? error;
  final List<Map<String, dynamic>> history;

  const PhishingScanState({
    this.status = PhishingScanStatus.initial,
    this.result,
    this.error,
    this.history = const [],
  });

  PhishingScanState copyWith({
    PhishingScanStatus? status,
    PhishingPredictionResult? Function()? result,
    String? Function()? error,
    List<Map<String, dynamic>>? history,
  }) {
    return PhishingScanState(
      status: status ?? this.status,
      result: result != null ? result() : this.result,
      error: error != null ? error() : this.error,
      history: history ?? this.history,
    );
  }
}
