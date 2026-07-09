abstract class QuizEvent {}

class QuizStarted extends QuizEvent {}

class QuizAnswerSubmitted extends QuizEvent {
  final int answerIndex;
  QuizAnswerSubmitted(this.answerIndex);
}

class QuizNextQuestionRequested extends QuizEvent {}
