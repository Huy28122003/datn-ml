import 'prediction_label.dart';

class ModelVote {
  final String modelId;
  final String displayName;
  final PredictionLabel label;

  const ModelVote({
    required this.modelId,
    required this.displayName,
    required this.label,
  });
}
