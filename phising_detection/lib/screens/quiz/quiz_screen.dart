import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:auto_route/auto_route.dart';
import '../phishing_scan/bloc/quiz_bloc.dart';
import '../phishing_scan/bloc/quiz_event.dart';
import '../phishing_scan/bloc/quiz_state.dart';
import '../../theme/app_theme.dart';

@RoutePage()
class QuizScreen extends StatefulWidget {
  const QuizScreen({super.key});

  @override
  State<QuizScreen> createState() => _QuizScreenState();
}

class _QuizScreenState extends State<QuizScreen> {
  @override
  void initState() {
    super.initState();
    context.read<QuizBloc>().add(QuizStarted());
  }

  @override
  Widget build(BuildContext context) {
    return const QuizView();
  }
}

class QuizView extends StatelessWidget {
  const QuizView({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<QuizBloc, QuizState>(
      builder: (context, state) {
        return Scaffold(
          backgroundColor: AppColors.background,
          appBar: AppBar(
            backgroundColor: Colors.transparent,
            elevation: 0,
            leading: Container(
              margin: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: AppColors.accent.withOpacity(0.15),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(
                Icons.videogame_asset_outlined,
                color: AppColors.accent,
                size: 20,
              ),
            ),
            title: const Text(
              'Thử thách Bảo mật',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: AppColors.textPrimary,
              ),
            ),
            actions: [
              IconButton(
                icon: const Icon(Icons.close, color: AppColors.textSecondary),
                onPressed: () => context.router.back(),
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

  Widget _buildBody(BuildContext context, QuizState state) {
    if (state.status == QuizStatus.inProgress) {
      return _buildQuestionView(context, state);
    } else if (state.status == QuizStatus.completed) {
      return _buildCompletionView(context, state);
    }
    return const Center(
      child: CircularProgressIndicator(color: AppColors.accent),
    );
  }

  Widget _buildQuestionView(BuildContext context, QuizState state) {
    final question = state.currentQuestion!;
    final progress = (state.currentQuestionIndex + 1) / state.questions.length;

    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Linear Progress indicator
          Row(
            children: [
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: progress,
                    backgroundColor: AppColors.surfaceLight,
                    valueColor: const AlwaysStoppedAnimation<Color>(AppColors.accent),
                    minHeight: 6,
                  ),
                ),
              ),
              const SizedBox(width: 14),
              Text(
                '${state.currentQuestionIndex + 1}/${state.questions.length}',
                style: const TextStyle(
                  color: AppColors.textSecondary,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),
          
          // Question text card
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: AppColors.border),
              boxShadow: [
                BoxShadow(
                  color: AppColors.primary.withOpacity(0.04),
                  blurRadius: 15,
                  offset: const Offset(0, 5),
                ),
              ],
            ),
            child: Text(
              question.text,
              style: const TextStyle(
                color: AppColors.textPrimary,
                fontSize: 16,
                fontWeight: FontWeight.w600,
                height: 1.5,
              ),
            ),
          ),
          const SizedBox(height: 24),
          
          // Options List
          Column(
            children: List.generate(question.options.length, (index) {
              return _buildOptionButton(context, state, index);
            }),
          ),
          
          // Explanation Box (shows after answered)
          if (state.hasAnswered) ...[
            const SizedBox(height: 20),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: state.isAnswerCorrect 
                    ? AppColors.safe.withOpacity(0.08) 
                    : AppColors.danger.withOpacity(0.08),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(
                  color: state.isAnswerCorrect 
                      ? AppColors.safe.withOpacity(0.35) 
                      : AppColors.danger.withOpacity(0.35),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        state.isAnswerCorrect ? Icons.check_circle_outline : Icons.info_outline,
                        color: state.isAnswerCorrect ? AppColors.safe : AppColors.danger,
                        size: 20,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        state.isAnswerCorrect ? 'Chính xác!' : 'Giải thích học thuật:',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          color: state.isAnswerCorrect ? AppColors.safe : AppColors.danger,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    question.explanation,
                    style: const TextStyle(
                      color: AppColors.textPrimary,
                      fontSize: 13,
                      height: 1.45,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 32),
            
            // Next Button
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () {
                  context.read<QuizBloc>().add(QuizNextQuestionRequested());
                },
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(state.isLastQuestion ? 'Xem kết quả' : 'Câu tiếp theo'),
                    const SizedBox(width: 8),
                    const Icon(Icons.arrow_forward, size: 18),
                  ],
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildOptionButton(
    BuildContext context, 
    QuizState state, 
    int index,
  ) {
    final question = state.currentQuestion!;
    final isSelected = state.selectedAnswerIndex == index;
    final isCorrect = question.correctIndex == index;
    final hasAnswered = state.hasAnswered;

    Color buttonColor = AppColors.surface;
    Color borderColor = AppColors.border;
    Color textColor = AppColors.textPrimary;
    Widget? icon;

    if (hasAnswered) {
      if (isCorrect) {
        buttonColor = AppColors.safe.withOpacity(0.12);
        borderColor = AppColors.safe;
        textColor = AppColors.safeGlow;
        icon = const Icon(Icons.check_circle, color: AppColors.safe, size: 18);
      } else if (isSelected) {
        buttonColor = AppColors.danger.withOpacity(0.12);
        borderColor = AppColors.danger;
        textColor = AppColors.dangerGlow;
        icon = const Icon(Icons.cancel, color: AppColors.danger, size: 18);
      } else {
        // Dim other options
        textColor = AppColors.textSecondary.withOpacity(0.5);
      }
    }

    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: hasAnswered
              ? null
              : () {
                  context.read<QuizBloc>().add(QuizAnswerSubmitted(index));
                },
          child: Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
            decoration: BoxDecoration(
              color: buttonColor,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: borderColor, width: isSelected || (hasAnswered && isCorrect) ? 2.0 : 1.0),
            ),
            child: Row(
              children: [
                Expanded(
                  child: Text(
                    question.options[index],
                    style: TextStyle(
                      color: textColor,
                      fontSize: 14,
                      fontWeight: isSelected || (hasAnswered && isCorrect) ? FontWeight.bold : FontWeight.normal,
                      height: 1.35,
                    ),
                  ),
                ),
                if (icon != null) ...[
                  const SizedBox(width: 8),
                  icon,
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildCompletionView(BuildContext context, QuizState state) {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(28),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Celebration Badge Icon
            Container(
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                color: AppColors.accent.withOpacity(0.1),
                shape: BoxShape.circle,
                border: Border.all(color: AppColors.accent.withOpacity(0.25), width: 2),
              ),
              child: Text(
                state.badgeEmoji,
                style: const TextStyle(fontSize: 64),
              ),
            ),
            const SizedBox(height: 24),
            
            // Score Display
            Text(
              'Điểm số: ${state.score}/${state.questions.length}',
              style: const TextStyle(
                fontSize: 18,
                color: AppColors.textSecondary,
                fontWeight: FontWeight.w500,
              ),
            ),
            const SizedBox(height: 8),
            
            // Badge Name
            Text(
              state.badgeName,
              style: const TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.w800,
                color: AppColors.accent,
                letterSpacing: 0.5,
              ),
            ),
            const SizedBox(height: 16),
            
            // Badge Description Card
            Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                color: AppColors.surface,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: AppColors.border),
              ),
              child: Text(
                state.badgeDescription,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 14,
                  height: 1.45,
                ),
              ),
            ),
            const SizedBox(height: 48),
            
            // Buttons
            Row(
              children: [
                Expanded(
                  child: OutlinedButton(
                    onPressed: () {
                      context.read<QuizBloc>().add(QuizStarted());
                    },
                    style: OutlinedButton.styleFrom(
                      side: const BorderSide(color: AppColors.border),
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                    ),
                    child: const Text('Chơi lại', style: TextStyle(color: AppColors.textPrimary)),
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: ElevatedButton(
                    onPressed: () => context.router.back(),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                    child: const Text('Đóng'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
