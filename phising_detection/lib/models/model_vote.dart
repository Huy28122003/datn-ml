import 'prediction_label.dart';

class ModelVote {
  final String modelId;
  final String displayName;
  final PredictionLabel label;
  final String? customLabel;
  final String? subtitle;

  const ModelVote({
    required this.modelId,
    required this.displayName,
    required this.label,
    this.customLabel,
    this.subtitle,
  });
}
