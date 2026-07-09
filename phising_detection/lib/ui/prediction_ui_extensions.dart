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
      case PredictionLabel.failure:
        return 'Không phân tích được dữ liệu';
    }
  }

  String get subtitleVi {
    switch (this) {
      case PredictionLabel.phishing:
        return 'Đa số mô hình phát hiện dấu hiệu đáng ngờ';
      case PredictionLabel.legitimate:
        return 'Hệ thống không phát hiện dấu hiệu đáng ngờ';
      case PredictionLabel.failure:
        return 'Không phân tích được dữ liệu trang web';
    }
  }

  Color get color {
    if (this == PredictionLabel.failure) return Colors.grey;
    return isPhishing ? AppColors.danger : AppColors.safe;
  }

  Color get glowColor {
    if (this == PredictionLabel.failure) return Colors.grey.withOpacity(0.2);
    return isPhishing ? AppColors.dangerGlow : AppColors.safeGlow;
  }

  IconData get icon {
    if (this == PredictionLabel.failure) return Icons.error_outline_rounded;
    return isPhishing ? Icons.gpp_bad_rounded : Icons.verified_user_rounded;
  }
}

extension PhishingResultUi on PhishingPredictionResult {
  String get votesSummary =>
      '$phishingVotes / $totalModels mô hình báo phishing';
}

extension ModelVoteUi on ModelVote {
  String get labelVi {
    if (label == PredictionLabel.failure) return 'Lỗi kết nối';
    return label.isPhishing ? 'Phishing' : 'An toàn';
  }
}
