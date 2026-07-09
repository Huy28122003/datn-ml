import 'package:flutter/material.dart';

import '../models/model_vote.dart';
import '../models/prediction_label.dart';
import '../models/prediction_result.dart';
import '../theme/app_theme.dart';

extension PredictionLabelUi on PredictionLabel {
  String get titleVi {
    switch (this) {
      case PredictionLabel.phishing:
        return 'Nguy cơ phishing';
      case PredictionLabel.legitimate:
        return 'Trang web có độ an toàn cao';
    }
  }

  String get subtitleVi {
    switch (this) {
      case PredictionLabel.phishing:
        return 'Đa số mô hình phát hiện dấu hiệu đáng ngờ';
      case PredictionLabel.legitimate:
        return 'Hệ thống không phát hiện dấu hiệu đáng ngờ';
    }
  }

  Color get color => isPhishing ? AppColors.danger : AppColors.safe;

  Color get glowColor => isPhishing ? AppColors.dangerGlow : AppColors.safeGlow;

  IconData get icon =>
      isPhishing ? Icons.gpp_bad_rounded : Icons.verified_user_rounded;
}

extension PhishingResultUi on PhishingPredictionResult {
  String get votesSummary =>
      '$phishingVotes / $totalModels mô hình báo phishing';
}

extension ModelVoteUi on ModelVote {
  String get labelVi => label.isPhishing ? 'Phishing' : 'An toàn';
}
