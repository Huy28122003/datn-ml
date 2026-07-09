enum PredictionLabel {
  legitimate,
  phishing,
  failure,
}

extension PredictionLabelX on PredictionLabel {
  bool get isPhishing => this == PredictionLabel.phishing;
}
