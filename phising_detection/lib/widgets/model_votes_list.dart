import 'package:flutter/material.dart';

import '../models/model_vote.dart';
import '../models/prediction_label.dart';
import '../theme/app_theme.dart';
import '../ui/prediction_ui_extensions.dart';

class ModelVotesList extends StatelessWidget {
  final List<ModelVote> votes;

  const ModelVotesList({super.key, required this.votes});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Chi tiết từng mô hình',
          style: TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w600,
            color: Colors.white,
          ),
        ),
        const SizedBox(height: 12),
        ...votes.map((vote) => Padding(
              padding: const EdgeInsets.only(bottom: 10),
              child: _ModelVoteTile(vote: vote),
            )),
      ],
    );
  }
}

class _ModelVoteTile extends StatelessWidget {
  final ModelVote vote;

  const _ModelVoteTile({required this.vote});

  @override
  Widget build(BuildContext context) {
    final isPhishing = vote.label.isPhishing;
    final color = vote.label.color;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        children: [
          Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: color.withOpacity(0.15),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(
              isPhishing ? Icons.warning_amber_rounded : Icons.check_circle_outline,
              color: color,
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  vote.displayName,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  vote.subtitle ?? vote.modelId,
                  style: const TextStyle(
                    fontSize: 12,
                    color: Colors.white70,
                  ),
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: color.withOpacity(0.15),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: color.withOpacity(0.35)),
            ),
            child: Text(
              vote.labelVi,
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: color,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
