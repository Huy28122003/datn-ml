import 'model_vote.dart';
import 'prediction_label.dart';

class PhishingPredictionResult {
  final String url;
  final PredictionLabel consensusLabel;
  final int phishingVotes;
  final int totalModels;
  final List<ModelVote> modelVotes;
  final bool hybridFeaturesFromLiveFetch;
  final bool isWhitelisted;
  final String? detail;
  final double? levScore;
  final String? matchedDomain;
  final PredictionLabel? rfLabel;

  const PhishingPredictionResult({
    required this.url,
    required this.consensusLabel,
    required this.phishingVotes,
    required this.totalModels,
    required this.modelVotes,
    this.hybridFeaturesFromLiveFetch = false,
    this.isWhitelisted = false,
    this.detail,
    this.levScore,
    this.matchedDomain,
    this.rfLabel,
  });

  bool get isPhishing => consensusLabel.isPhishing;
  double get confidenceRatio => phishingVotes / totalModels;
}
