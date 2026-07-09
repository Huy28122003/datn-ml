import 'package:auto_route/auto_route.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../phishing_scan/bloc/phishing_scan_bloc.dart';
import '../phishing_scan/bloc/phishing_scan_event.dart';
import '../phishing_scan/bloc/phishing_scan_state.dart';
import '../../theme/app_theme.dart';

@RoutePage()
class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<PhishingScanBloc, PhishingScanState>(
      builder: (context, state) {
        return Scaffold(
          backgroundColor: AppColors.background,
          appBar: AppBar(
            backgroundColor: Colors.transparent,
            elevation: 0,
            leading: IconButton(
              icon: const Icon(Icons.arrow_back_ios,
                  color: AppColors.textPrimary),
              onPressed: () => context.router.back(),
            ),
            title: const Text(
              'Lịch sử quét URL',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: AppColors.textPrimary,
              ),
            ),
            actions: [
              if (state.history.isNotEmpty)
                IconButton(
                  icon: const Icon(
                    Icons.delete_sweep_outlined,
                    color: AppColors.danger,
                    size: 24,
                  ),
                  tooltip: 'Xóa toàn bộ lịch sử',
                  onPressed: () => _confirmClearHistory(context),
                ),
            ],
          ),
          body: SafeArea(
            child: _buildBody(context, state),
          ),
        );
      },
    );
  }

  void _confirmClearHistory(BuildContext context) {
    showDialog(
      context: context,
      builder: (dialogCtx) => AlertDialog(
        backgroundColor: AppColors.surface,
        title: const Text('Xóa lịch sử?',
            style: TextStyle(color: AppColors.textPrimary)),
        content: const Text(
          'Bạn có chắc chắn muốn xóa toàn bộ lịch sử kiểm tra URL không?',
          style: TextStyle(color: AppColors.textSecondary),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogCtx).pop(),
            child: const Text('Hủy',
                style: TextStyle(color: AppColors.textSecondary)),
          ),
          TextButton(
            onPressed: () {
              context
                  .read<PhishingScanBloc>()
                  .add(PhishingScanHistoryCleared());
              Navigator.of(dialogCtx).pop();
            },
            child: const Text('Xóa', style: TextStyle(color: AppColors.danger)),
          ),
        ],
      ),
    );
  }

  Widget _buildBody(BuildContext context, PhishingScanState state) {
    if (state.history.isEmpty) {
      return Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(28),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: AppColors.surfaceLight,
                  shape: BoxShape.circle,
                  border: Border.all(color: AppColors.border),
                ),
                child: Icon(
                  Icons.folder_open_outlined,
                  color: AppColors.textSecondary.withOpacity(0.5),
                  size: 64,
                ),
              ),
              const SizedBox(height: 20),
              const Text(
                'Lịch sử trống',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: AppColors.textPrimary,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'Các liên kết bạn đã kiểm tra sẽ xuất hiện ở đây.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 14,
                  color: AppColors.textSecondary.withOpacity(0.8),
                ),
              ),
            ],
          ),
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      itemCount: state.history.length,
      itemBuilder: (context, index) {
        final item = state.history[index];
        final url = item['url'] as String? ?? '';
        final timestampStr = item['timestamp'] as String? ?? '';
        final resultType = item['resultType'] as String? ?? 'no_data';
        final detail = item['detail'] as String?;

        // Format timestamp
        String timeFormatted = '';
        try {
          final dt = DateTime.parse(timestampStr).toLocal();
          final hour = dt.hour.toString().padLeft(2, '0');
          final minute = dt.minute.toString().padLeft(2, '0');
          final day = dt.day.toString().padLeft(2, '0');
          final month = dt.month.toString().padLeft(2, '0');
          timeFormatted = '$hour:$minute - $day/$month';
        } catch (_) {
          timeFormatted = timestampStr;
        }

        // Determine styling based on type
        Color typeColor;
        IconData iconData;
        String labelText;

        switch (resultType) {
          case 'safe':
            typeColor = AppColors.safe;
            iconData = Icons.shield_outlined;
            labelText = 'An toàn';
            break;
          case 'phishing':
            typeColor = AppColors.danger;
            iconData = Icons.gpp_maybe_outlined;
            labelText = 'Lừa đảo';
            break;
          case 'no_data':
          default:
            typeColor = AppColors.accent;
            iconData = Icons.info_outline;
            labelText = 'Ko có dữ liệu';
            break;
        }

        return Padding(
          padding: const EdgeInsets.only(bottom: 12),
          child: Container(
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: AppColors.border),
              boxShadow: [
                BoxShadow(
                  color: typeColor.withOpacity(0.02),
                  blurRadius: 10,
                  offset: const Offset(0, 4),
                ),
              ],
            ),
            child: Material(
              color: Colors.transparent,
              child: InkWell(
                borderRadius: BorderRadius.circular(16),
                onTap: () {
                  context.router.maybePop(url);
                },
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(10),
                        decoration: BoxDecoration(
                          color: typeColor.withOpacity(0.12),
                          shape: BoxShape.circle,
                        ),
                        child: Icon(
                          iconData,
                          color: typeColor,
                          size: 22,
                        ),
                      ),
                      const SizedBox(width: 14),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              url,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(
                                color: AppColors.textPrimary,
                                fontSize: 14,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            const SizedBox(height: 6),
                            Row(
                              children: [
                                Text(
                                  timeFormatted,
                                  style: TextStyle(
                                    color: AppColors.textSecondary
                                        .withOpacity(0.7),
                                    fontSize: 11,
                                  ),
                                ),
                                if (detail != null && detail.isNotEmpty) ...[
                                  const SizedBox(width: 8),
                                  Container(
                                    width: 3,
                                    height: 3,
                                    decoration: BoxDecoration(
                                      shape: BoxShape.circle,
                                      color: AppColors.textSecondary
                                          .withOpacity(0.5),
                                    ),
                                  ),
                                  const SizedBox(width: 8),
                                  Expanded(
                                    child: Text(
                                      detail,
                                      maxLines: 1,
                                      overflow: TextOverflow.ellipsis,
                                      style: TextStyle(
                                        color: AppColors.textSecondary
                                            .withOpacity(0.7),
                                        fontSize: 11,
                                      ),
                                    ),
                                  ),
                                ],
                              ],
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(width: 10),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 6),
                        decoration: BoxDecoration(
                          color: typeColor.withOpacity(0.12),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                            color: typeColor.withOpacity(0.35),
                            width: 1.0,
                          ),
                        ),
                        child: Text(
                          labelText,
                          style: TextStyle(
                            fontSize: 10,
                            fontWeight: FontWeight.w700,
                            color: typeColor,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}
