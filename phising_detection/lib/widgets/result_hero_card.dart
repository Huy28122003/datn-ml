import 'package:flutter/material.dart';

import '../models/prediction_result.dart';
import '../theme/app_theme.dart';
import '../ui/prediction_ui_extensions.dart';

class ResultHeroCard extends StatelessWidget {
  final PhishingPredictionResult result;

  const ResultHeroCard({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final isOffline =
        !result.hybridFeaturesFromLiveFetch && !result.isWhitelisted;

    if (isOffline) {
      const color = Colors.grey;
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(24),
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              color.withOpacity(0.15),
              AppColors.surface,
            ],
          ),
          border: Border.all(color: color.withOpacity(0.35)),
          boxShadow: [
            BoxShadow(
              color: color.withOpacity(0.15),
              blurRadius: 28,
              offset: const Offset(0, 12),
            ),
          ],
        ),
        child: Column(
          children: [
            Container(
              width: 72,
              height: 72,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: color.withOpacity(0.12),
                border: Border.all(color: color.withOpacity(0.35)),
              ),
              child: Icon(Icons.cloud_off_rounded, size: 40, color: color),
            ),
            const SizedBox(height: 16),
            const Text(
              'Trang web đã bị sập',
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            const Text(
              'Không thể kết nối hoặc tải dữ liệu từ trang web này (Offline).',
              style: TextStyle(
                fontSize: 14,
                color: Colors.white70,
                height: 1.4,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      );
    }

    final label = result.rfLabel ?? result.consensusLabel;
    final color = label.color;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(24),
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            color.withOpacity(0.22),
            AppColors.surface,
          ],
        ),
        border: Border.all(color: color.withOpacity(0.45)),
        boxShadow: [
          BoxShadow(
            color: color.withOpacity(0.25),
            blurRadius: 28,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: Column(
        children: [
          if (result.rfLabel != null) ...[
            const Text(
              'MÔ HÌNH RANDOM FOREST (HYBRID)',
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w700,
                color: Colors.white70,
                letterSpacing: 0.5,
              ),
            ),
            const SizedBox(height: 16),
          ],
          Container(
            width: 72,
            height: 72,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: color.withOpacity(0.18),
              border: Border.all(color: color.withOpacity(0.5)),
            ),
            child: Icon(label.icon, size: 40, color: color),
          ),
          const SizedBox(height: 16),
          Text(
            label.titleVi,
            style: const TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 8),
          Text(
            result.rfLabel != null
                ? label.subtitleVi
                : (result.detail ?? label.subtitleVi),
            style: const TextStyle(
              fontSize: 14,
              color: Colors.white70,
              height: 1.4,
            ),
            textAlign: TextAlign.center,
          ),
          if (result.rfLabel == null) ...[
            const SizedBox(height: 20),
            _VoteMeter(
              phishingVotes: result.phishingVotes,
              total: result.totalModels,
              color: color,
            ),
            const SizedBox(height: 12),
            Text(
              result.votesSummary,
              style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w500,
                color: color.withOpacity(0.9),
              ),
            ),
          ],
          if (result.hybridFeaturesFromLiveFetch) ...[
            const SizedBox(height: 14),
            const _InfoChip(
              icon: Icons.wifi_tethering_rounded,
              text: 'Đã phân tích HTML trực tiếp',
            ),
          ],
        ],
      ),
    );
  }
}

class _VoteMeter extends StatelessWidget {
  final int phishingVotes;
  final int total;
  final Color color;

  const _VoteMeter({
    required this.phishingVotes,
    required this.total,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: List.generate(total, (index) {
        final isPhishingVote = index < phishingVotes;
        return Expanded(
          child: Container(
            height: 8,
            margin: EdgeInsets.only(left: index == 0 ? 0 : 6),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(8),
              color: isPhishingVote ? color : AppColors.border.withOpacity(0.8),
            ),
          ),
        );
      }),
    );
  }
}

class _InfoChip extends StatelessWidget {
  final IconData icon;
  final String text;
  final bool isWarning;

  const _InfoChip({
    required this.icon,
    required this.text,
    this.isWarning = false,
  });

  @override
  Widget build(BuildContext context) {
    final bgColor =
        isWarning ? AppColors.danger.withOpacity(0.12) : AppColors.surfaceLight;
    final borderColor =
        isWarning ? AppColors.danger.withOpacity(0.35) : AppColors.border;
    final textColor = isWarning ? AppColors.danger : Colors.white70;
    final iconColor = isWarning ? AppColors.danger : AppColors.accent;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: borderColor),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: iconColor),
          const SizedBox(width: 8),
          Text(
            text,
            style: TextStyle(
              fontSize: 12,
              color: textColor,
              fontWeight: isWarning ? FontWeight.w500 : FontWeight.normal,
            ),
          ),
        ],
      ),
    );
  }
}
