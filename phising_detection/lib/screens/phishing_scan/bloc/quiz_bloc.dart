import 'package:flutter_bloc/flutter_bloc.dart';
import '../../../../models/quiz_question.dart';
import 'quiz_event.dart';
import 'quiz_state.dart';

class QuizBloc extends Bloc<QuizEvent, QuizState> {
  static const List<QuizQuestion> _demoQuestions = [
    QuizQuestion(
      text: "Bạn nhận được tin nhắn SMS từ 'Vietcombank' thông báo tài khoản bị khóa khẩn cấp và yêu cầu bạn click link 'http://vietcombanh-login.com' để kích hoạt lại. Bạn nên làm gì?",
      options: [
        "Click link để xử lý ngay vì thông báo khẩn cấp.",
        "Bỏ qua SMS, gọi trực tiếp hotline ngân hàng để xác nhận hoặc mở app chính chủ.",
        "Nhập thông tin tài khoản ngân hàng và mã OTP vào web đó xem sao."
      ],
      correctIndex: 1,
      explanation: "Không bao giờ click vào link lạ từ tin nhắn SMS thương hiệu giả mạo. Link ngân hàng thật thường sử dụng giao thức HTTPS và domain chính thức (ví dụ: vietcombank.com.vn) chứ không dùng các domain biến thể (vietcombanh-login.com).",
    ),
    QuizQuestion(
      text: "Để thực hiện tấn công lừa đảo, kẻ xấu thường sử dụng kỹ thuật Typosquatting (tên miền gần giống). Trong 3 tên miền dưới đây, đâu là tên miền giả mạo dịch vụ PayPal?",
      options: [
        "https://www.paypal.com",
        "https://www.paypaI.com (với chữ 'I' viết hoa thay chữ 'l')",
        "Cả hai đều là chính chủ."
      ],
      correctIndex: 1,
      explanation: "Chữ 'I' (i viết hoa) trông cực kỳ giống chữ 'l' (L viết thường) trên một số phông chữ. Kẻ xấu lợi dụng điều này để đăng ký tên miền lừa đảo paypaI.com hòng dụ người dùng đăng nhập.",
    ),
    QuizQuestion(
      text: "Email từ địa chỉ 'it-support@congty-gmail.com' gửi cho bạn yêu cầu nhập lại mật khẩu email công ty qua Google Form để nâng cấp bảo mật hệ thống. Hành vi này có an toàn?",
      options: [
        "An toàn, vì do bộ phận IT Support của công ty gửi.",
        "Không an toàn. IT thực sự của công ty sẽ dùng email có đuôi domain công ty và không bao giờ dùng Google Form để thu thập mật khẩu.",
        "An toàn, miễn là đường link Google Form có giao thức HTTPS bảo mật."
      ],
      correctIndex: 1,
      explanation: "Các tổ chức, doanh nghiệp nghiêm túc không bao giờ sử dụng Google Form để yêu cầu nhân viên nhập thông tin nhạy cảm như mật khẩu. Hơn nữa, đuôi email giả mạo domain công ty (@congty-gmail.com) là dấu hiệu rõ ràng của phishing.",
    ),
    QuizQuestion(
      text: "Một website mua sắm có thanh toán trực tuyến sử dụng giao thức bảo mật 'https://' và có biểu tượng ổ khóa màu xanh lá. Điều này có chứng minh website đó 100% an toàn không?",
      options: [
        "Có, HTTPS đảm bảo website đó hoàn toàn đáng tin cậy và không lừa đảo.",
        "Không, HTTPS chỉ chứng minh kết nối giữa trình duyệt và máy chủ đã được mã hóa. Kẻ lừa đảo hoàn toàn có thể đăng ký chứng chỉ SSL/HTTPS cho trang web giả mạo của chúng.",
        "Có, ổ khóa chứng minh trang web đã được cảnh sát mạng phê duyệt."
      ],
      correctIndex: 1,
      explanation: "HTTPS và biểu tượng ổ khóa chỉ đảm bảo rằng dữ liệu truyền đi được mã hóa an toàn, không bị nghe lén trên đường truyền. Nó KHÔNG chứng minh website đó là thật hay giả. Rất nhiều trang web lừa đảo hiện nay vẫn có chứng chỉ HTTPS.",
    )
  ];

  QuizBloc() : super(const QuizState()) {
    on<QuizStarted>(_onQuizStarted);
    on<QuizAnswerSubmitted>(_onAnswerSubmitted);
    on<QuizNextQuestionRequested>(_onNextQuestionRequested);
  }

  void _onQuizStarted(QuizStarted event, Emitter<QuizState> emit) {
    emit(QuizState(
      status: QuizStatus.inProgress,
      questions: _demoQuestions,
      currentQuestionIndex: 0,
      score: 0,
    ));
  }

  void _onAnswerSubmitted(QuizAnswerSubmitted event, Emitter<QuizState> emit) {
    if (state.status == QuizStatus.inProgress) {
      if (state.hasAnswered) return;

      final currentQuestion = state.currentQuestion!;
      final isCorrect = event.answerIndex == currentQuestion.correctIndex;
      final newScore = isCorrect ? state.score + 1 : state.score;

      emit(state.copyWith(
        score: newScore,
        selectedAnswerIndex: () => event.answerIndex,
        isAnswerCorrect: isCorrect,
      ));
    }
  }

  void _onNextQuestionRequested(QuizNextQuestionRequested event, Emitter<QuizState> emit) {
    if (state.status == QuizStatus.inProgress) {
      if (!state.hasAnswered) return;

      if (state.isLastQuestion) {
        String badgeName;
        String badgeEmoji;
        String badgeDescription;

        final score = state.score;
        final total = state.questions.length;

        if (score == total) {
          badgeName = "Hiệp Sĩ Hoàng Kim";
          badgeEmoji = "🛡️👹🛡️";
          badgeDescription = "Xuất sắc! Bạn sở hữu Hỏa Nhãn Kim Tinh, nhìn thấu mọi yêu quái giả mạo trên không gian mạng!";
        } else if (score >= 2) {
          badgeName = "Thám Tử Tập Sự";
          badgeEmoji = "🕵️‍♂️🔍";
          badgeDescription = "Khá tốt! Bạn có nhận thức tốt về an ninh mạng, nhưng vẫn có nguy cơ bị sập các bẫy tinh vi.";
        } else {
          badgeName = "Cá Vàng Ngây Thơ";
          badgeEmoji = "🐠🎣";
          badgeDescription = "Rất nguy hiểm! Bạn rất dễ tin người và là mục tiêu béo bở của kẻ xấu. Hãy nâng cao kiến thức bảo mật!";
        }

        emit(state.copyWith(
          status: QuizStatus.completed,
          badgeName: badgeName,
          badgeEmoji: badgeEmoji,
          badgeDescription: badgeDescription,
        ));
      } else {
        emit(state.copyWith(
          currentQuestionIndex: state.currentQuestionIndex + 1,
          selectedAnswerIndex: () => null,
          isAnswerCorrect: false,
        ));
      }
    }
  }
}
