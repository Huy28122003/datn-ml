import '../../../../models/quiz_question.dart';

enum QuizStatus { initial, inProgress, completed }

class QuizState {
  final QuizStatus status;
  final List<QuizQuestion> questions;
  final int currentQuestionIndex;
  final int score;
  final int? selectedAnswerIndex;
  final bool isAnswerCorrect;

  // Badge fields
  final String badgeName;
  final String badgeEmoji;
  final String badgeDescription;

  const QuizState({
    this.status = QuizStatus.initial,
    this.questions = const [],
    this.currentQuestionIndex = 0,
    this.score = 0,
    this.selectedAnswerIndex,
    this.isAnswerCorrect = false,
    this.badgeName = "",
    this.badgeEmoji = "",
    this.badgeDescription = "",
  });

  QuizQuestion? get currentQuestion => 
      (status == QuizStatus.inProgress && currentQuestionIndex < questions.length) 
          ? questions[currentQuestionIndex] 
          : null;
  bool get hasAnswered => selectedAnswerIndex != null;
  bool get isLastQuestion => currentQuestionIndex == questions.length - 1;

  QuizState copyWith({
    QuizStatus? status,
    List<QuizQuestion>? questions,
    int? currentQuestionIndex,
    int? score,
    int? Function()? selectedAnswerIndex,
    bool? isAnswerCorrect,
    String? badgeName,
    String? badgeEmoji,
    String? badgeDescription,
  }) {
    return QuizState(
      status: status ?? this.status,
      questions: questions ?? this.questions,
      currentQuestionIndex: currentQuestionIndex ?? this.currentQuestionIndex,
      score: score ?? this.score,
      selectedAnswerIndex: selectedAnswerIndex != null ? selectedAnswerIndex() : this.selectedAnswerIndex,
      isAnswerCorrect: isAnswerCorrect ?? this.isAnswerCorrect,
      badgeName: badgeName ?? this.badgeName,
      badgeEmoji: badgeEmoji ?? this.badgeEmoji,
      badgeDescription: badgeDescription ?? this.badgeDescription,
    );
  }
}
