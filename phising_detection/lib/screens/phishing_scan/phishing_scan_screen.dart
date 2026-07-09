import 'package:auto_route/auto_route.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../../routes/app_router.gr.dart';
import '../../theme/app_theme.dart' show AppColors;
import 'bloc/phishing_scan_bloc.dart';
import 'bloc/phishing_scan_event.dart';
import 'bloc/phishing_scan_state.dart';
import '../../widgets/gradient_background.dart';
import '../../widgets/model_votes_list.dart';
import '../../widgets/result_hero_card.dart';
import '../../widgets/knn_matching_card.dart';

@RoutePage()
class PhishingScanScreen extends StatefulWidget {
  const PhishingScanScreen({super.key});

  @override
  State<PhishingScanScreen> createState() => _PhishingScanScreenState();
}

class _PhishingScanScreenState extends State<PhishingScanScreen> {
  final _urlController = TextEditingController();
  final _urlFocus = FocusNode();

  @override
  void initState() {
    super.initState();
    // huynq - Khoi tao model ONNX qua BLoC
    context.read<PhishingScanBloc>().add(PhishingScanBootstrap());
  }

  void _onScan({required bool? rs}) {
    FocusScope.of(context).unfocus();
    final url = _urlController.text.trim();
    if (url.isEmpty) {
      context.read<PhishingScanBloc>().add(PhishingScanSubmitted('', true));
      return;
    }
    if (rs == null) {
      context.read<PhishingScanBloc>().add(PhishingScanVirusTotalSubmitted(url));
    } else {
      context.read<PhishingScanBloc>().add(PhishingScanSubmitted(url, rs));
    }
  }

  void _fillExample(String url) {
    _urlController.text = url;
    context.read<PhishingScanBloc>().add(PhishingScanClear());
  }

  @override
  void dispose() {
    _urlController.dispose();
    _urlFocus.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<PhishingScanBloc, PhishingScanState>(
      builder: (context, state) {
        final initializing = state.status == PhishingScanStatus.loadingModel ||
            state.status == PhishingScanStatus.initial;
        final initError = state.status == PhishingScanStatus.loadModelFailure
            ? state.error
            : null;

        return Scaffold(
          body: GradientBackground(
            child: SafeArea(
              child: initializing
                  ? _buildLoadingState()
                  : initError != null
                      ? _buildInitErrorState(initError)
                      : _buildMainContent(state),
            ),
          ),
        );
      },
    );
  }

  Widget _buildLoadingState() {
    return const Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(
            width: 48,
            height: 48,
            child: CircularProgressIndicator(
              strokeWidth: 3,
              color: AppColors.accent,
            ),
          ),
          SizedBox(height: 20),
          Text(
            'Đang tải mô hình AI...',
            style: TextStyle(
              color: AppColors.textPrimary,
              fontSize: 16,
              fontWeight: FontWeight.w500,
            ),
          ),
          SizedBox(height: 8),
          Text(
            'Random Forest (Hybrid)',
            style: TextStyle(color: AppColors.textSecondary, fontSize: 13),
          ),
        ],
      ),
    );
  }

  Widget _buildInitErrorState(String error) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(28),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.error_outline, size: 56, color: AppColors.danger),
          const SizedBox(height: 16),
          Text(
            error,
            textAlign: TextAlign.center,
            style: const TextStyle(color: AppColors.textSecondary, height: 1.5),
          ),
          const SizedBox(height: 24),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: () {
                context.read<PhishingScanBloc>().add(PhishingScanBootstrap());
              },
              icon: const Icon(Icons.refresh),
              label: const Text('Thử lại'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMainContent(PhishingScanState state) {
    final scanning = state.status == PhishingScanStatus.scanning;
    final result =
        state.status == PhishingScanStatus.success ? state.result : null;
    final scanError =
        state.status == PhishingScanStatus.failure ? state.error : null;

    return CustomScrollView(
      slivers: [
        SliverToBoxAdapter(child: _buildHeader()),
        SliverToBoxAdapter(child: _buildInputSection(scanning)),
        if (scanError != null)
          SliverToBoxAdapter(child: _buildErrorBanner(scanError)),
        if (result != null) ...[
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 8, 20, 0),
              child: KnnMatchingCard(result: result),
            ),
          ),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 0),
              child: ResultHeroCard(result: result),
            ),
          ),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 20, 20, 8),
              child: _buildUrlAnalyzedChip(result.url),
            ),
          ),
          if (!(result.totalModels == 1 && !result.hybridFeaturesFromLiveFetch))
            SliverToBoxAdapter(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(20, 8, 20, 16),
                child: ModelVotesList(votes: result.modelVotes),
              ),
            ),
        ],
        SliverToBoxAdapter(child: _buildHistorySection(state)),
        const SliverToBoxAdapter(child: SizedBox(height: 32)),
      ],
    );
  }

  Widget _buildHeader() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 12, 20, 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(
                    color: AppColors.primary.withOpacity(0.35),
                  ),
                ),
                child: const Icon(
                  Icons.shield_outlined,
                  color: AppColors.accent,
                  size: 28,
                ),
              ),
              const SizedBox(width: 14),
              const Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'PhishGuard',
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: AppColors.textPrimary,
                        letterSpacing: -0.5,
                      ),
                    ),
                    Text(
                      'Phát hiện URL lừa đảo — offline',
                      style: TextStyle(
                        fontSize: 13,
                        color: AppColors.textSecondary,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildInputSection(bool scanning) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Nhập URL cần kiểm tra',
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w500,
              color: AppColors.textSecondary,
            ),
          ),
          const SizedBox(height: 10),
          TextField(
            controller: _urlController,
            focusNode: _urlFocus,
            keyboardType: TextInputType.url,
            textInputAction: TextInputAction.go,
            autocorrect: false,
            style: const TextStyle(color: AppColors.textPrimary),
            decoration: InputDecoration(
              hintText: 'https://example.com/login',
              prefixIcon: const Icon(
                Icons.link_rounded,
                color: AppColors.textSecondary,
              ),
              suffixIcon: _urlController.text.isNotEmpty
                  ? IconButton(
                      icon: const Icon(Icons.clear, size: 20),
                      color: AppColors.textSecondary,
                      onPressed: () {
                        _urlController.clear();
                        context
                            .read<PhishingScanBloc>()
                            .add(PhishingScanClear());
                      },
                    )
                  : null,
            ),
            // onSubmitted: (_) => scanning ? null : _onScan(),
            onChanged: (text) {
              // Clear previous results or errors as user types new URL
              context.read<PhishingScanBloc>().add(PhishingScanClear());
            },
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              _ExampleChip(
                label: 'Google',
                onTap: () => _fillExample('https://google.com'),
              ),
              _ExampleChip(
                label: 'GitHub',
                onTap: () => _fillExample('https://github.com'),
              ),
            ],
          ),
          const SizedBox(height: 16),
          SizedBox(
            width: double.infinity,
            child: Stack(
              children: [
                Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    color: Colors.blue,
                    borderRadius: BorderRadius.circular(14),
                  ),
                  padding: const EdgeInsets.symmetric(
                      horizontal: 16, vertical: 10),
                  child: scanning
                      ? const Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            SizedBox(
                              height: 20,
                              width: 20,
                              child: CircularProgressIndicator(
                                strokeWidth: 2.5,
                                color: Colors.white,
                              ),
                            ),
                            SizedBox(width: 10),
                            Text(
                              'Đang phân tích...',
                              style: TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        )
                      : const Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.radar_rounded, size: 22, color: Colors.white),
                            SizedBox(width: 10),
                            Text(
                              'Quét URL',
                              style: TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        ),
                ),
                if (!scanning)
                  Positioned.fill(
                    child: Row(
                      children: [
                        Expanded(
                          child: GestureDetector(
                            behavior: HitTestBehavior.translucent,
                            onTap: () {
                              _onScan(rs: true);
                            },
                          ),
                        ),
                        Expanded(
                          child: GestureDetector(
                            behavior: HitTestBehavior.translucent,
                            onTap: () {
                              _onScan(rs: null);
                            },
                          ),
                        ),
                        Expanded(
                          child: GestureDetector(
                            behavior: HitTestBehavior.translucent,
                            onTap: () {
                              _onScan(rs: false);
                            },
                          ),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          const Center(
            child: Text(
              'Mô hình Random Forest (Hybrid) phân tích',
              style: TextStyle(fontSize: 12, color: AppColors.textSecondary),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildErrorBanner(String message) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 4, 20, 8),
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: AppColors.danger.withOpacity(0.12),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppColors.danger.withOpacity(0.35)),
        ),
        child: Row(
          children: [
            const Icon(Icons.info_outline, color: AppColors.danger, size: 20),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                message,
                style: const TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 13,
                  height: 1.35,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildUrlAnalyzedChip(String url) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.language, size: 18, color: AppColors.textSecondary),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'URL đã phân tích',
                  style: TextStyle(
                    fontSize: 11,
                    color: AppColors.textSecondary,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  url,
                  style: const TextStyle(
                    fontSize: 13,
                    color: AppColors.textPrimary,
                    height: 1.35,
                  ),
                ),
              ],
            ),
          ),
          IconButton(
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(),
            icon: const Icon(Icons.copy_rounded, size: 18),
            color: AppColors.accent,
            onPressed: () {
              Clipboard.setData(ClipboardData(text: url));
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Đã sao chép URL')),
              );
            },
          ),
        ],
      ),
    );
  }

  Widget _buildHistorySection(PhishingScanState state) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  Icon(
                    Icons.history_rounded,
                    color: AppColors.accent.withOpacity(0.9),
                    size: 20,
                  ),
                  const SizedBox(width: 8),
                  const Text(
                    'Lịch sử kiểm tra',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: AppColors.textPrimary,
                    ),
                  ),
                ],
              ),
              if (state.history.isNotEmpty)
                TextButton.icon(
                  onPressed: () async {
                    final selectedUrl =
                        await context.router.push(const HistoryRoute());
                    if (selectedUrl is String && selectedUrl.isNotEmpty) {
                      _urlController.text = selectedUrl;
                      // _onScan();
                    }
                  },
                  icon: const Icon(Icons.arrow_forward_rounded,
                      size: 16, color: AppColors.accent),
                  label: const Text(
                    'Xem tất cả',
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: AppColors.accent,
                    ),
                  ),
                  style: TextButton.styleFrom(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    minimumSize: Size.zero,
                    tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  ),
                ),
            ],
          ),
          const SizedBox(height: 12),
          if (state.history.isEmpty)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 16),
              decoration: BoxDecoration(
                color: AppColors.surface,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: AppColors.border),
              ),
              child: Column(
                children: [
                  Icon(
                    Icons.folder_open_outlined,
                    color: AppColors.textSecondary.withOpacity(0.5),
                    size: 36,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Không có dữ liệu lịch sử',
                    style: TextStyle(
                      color: AppColors.textSecondary.withOpacity(0.8),
                      fontSize: 13,
                    ),
                  ),
                ],
              ),
            )
          else
            Column(
              children: state.history.take(3).map((item) {
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
                  padding: const EdgeInsets.only(bottom: 10),
                  child: Container(
                    decoration: BoxDecoration(
                      color: AppColors.surface,
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: AppColors.border),
                    ),
                    child: Material(
                      color: Colors.transparent,
                      child: InkWell(
                        borderRadius: BorderRadius.circular(16),
                        onTap: () {
                          _urlController.text = url;
                          // _onScan();
                        },
                        child: Padding(
                          padding: const EdgeInsets.all(14),
                          child: Row(
                            children: [
                              Container(
                                padding: const EdgeInsets.all(8),
                                decoration: BoxDecoration(
                                  color: typeColor.withOpacity(0.12),
                                  shape: BoxShape.circle,
                                ),
                                child: Icon(
                                  iconData,
                                  color: typeColor,
                                  size: 20,
                                ),
                              ),
                              const SizedBox(width: 12),
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
                                        fontWeight: FontWeight.w500,
                                      ),
                                    ),
                                    const SizedBox(height: 4),
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
                                        if (detail != null &&
                                            detail.isNotEmpty) ...[
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
                              const SizedBox(width: 8),
                              Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 10, vertical: 4),
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
              }).toList(),
            ),
        ],
      ),
    );
  }
}

class _ExampleChip extends StatelessWidget {
  final String label;
  final VoidCallback onTap;

  const _ExampleChip({required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return ActionChip(
      label: Text(label),
      labelStyle: const TextStyle(fontSize: 12, color: AppColors.textSecondary),
      backgroundColor: AppColors.surface,
      side: const BorderSide(color: AppColors.border),
      onPressed: onTap,
    );
  }
}
