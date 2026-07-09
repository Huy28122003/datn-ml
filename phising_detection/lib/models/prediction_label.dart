enum PredictionLabel {
  legitimate,
  phishing,
}

extension PredictionLabelX on PredictionLabel {
  bool get isPhishing => this == PredictionLabel.phishing;
}
