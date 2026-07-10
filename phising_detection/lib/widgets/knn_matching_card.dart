import 'package:flutter/material.dart';
import 'package:phising_detection/models/prediction_label.dart';
import '../models/model_vote.dart';
import '../models/prediction_result.dart';
import '../theme/app_theme.dart';

class KnnMatchingCard extends StatelessWidget {
  final PhishingPredictionResult result;

  const KnnMatchingCard({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final matchedDomain = result.matchedDomain ?? 'N/A';
    // huynq - Lay nhan thuc te cua model KNN de luon hien thi dung bat ke ket qua fake cua Random Forest
    final knnVote = result.modelVotes.firstWhere(
      (v) => v.modelId == 'knn_search',
      orElse: () => ModelVote(
        modelId: 'knn_search',
        displayName: '',
        label: result.consensusLabel,
      ),
    );
    final isPhishing = knnVote.label.isPhishing;

    // huynq - Chon mau status theo nhan phishing
    final Color statusColor = isPhishing ? AppColors.danger : AppColors.safe;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            statusColor.withOpacity(0.12),
            AppColors.surface,
          ],
        ),
        border: Border.all(color: statusColor.withOpacity(0.35)),
        boxShadow: [
          BoxShadow(
            color: statusColor.withOpacity(0.08),
            blurRadius: 16,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: statusColor.withOpacity(0.15),
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  isPhishing
                      ? Icons.warning_amber_rounded
                      : Icons.check_circle_outline,
                  color: statusColor,
                  size: 24,
                ),
              ),
              const SizedBox(width: 12),
              const Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Thuật Toán So Khớp & KNN',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    SizedBox(height: 2),
                    Text(
                      'Phân tích & so sánh với 20.000 tên miền sạch',
                      style: TextStyle(
                        fontSize: 11,
                        color: Colors.white70,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 18),
          _buildInfoRow('Tên miền giống nó nhất:', matchedDomain),
          if (result.detail != null && result.detail!.isNotEmpty) ...[
            const SizedBox(height: 14),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: AppColors.surfaceLight.withOpacity(0.5),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: AppColors.border),
              ),
              child: Text(
                result.detail!,
                style: const TextStyle(
                  fontSize: 12,
                  color: Colors.white70,
                  height: 1.4,
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildInfoRow(
    String label,
    String value, {
    Color? valueColor,
    bool isBoldValue = false,
  }) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          flex: 4,
          child: Text(
            label,
            style: const TextStyle(
              fontSize: 13,
              color: Colors.white70,
            ),
          ),
        ),
        Expanded(
          flex: 5,
          child: Text(
            value,
            style: TextStyle(
              fontSize: 13,
              fontWeight: isBoldValue ? FontWeight.bold : FontWeight.w500,
              color: valueColor ?? Colors.white,
            ),
            textAlign: TextAlign.right,
          ),
        ),
      ],
    );
  }
}
