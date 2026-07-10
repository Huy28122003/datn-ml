import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:phising_detection/models/prediction_result.dart';
import 'package:phising_detection/models/prediction_label.dart';
import 'package:phising_detection/models/model_vote.dart';
import 'package:phising_detection/usecases/virus_total/scan_url_usecase.dart';
import 'package:phising_detection/repositories/virus_total_repository_impl.dart';
import '../../../models/phishing_service_exception.dart';
import '../../../services/phising_predictor.dart';
import 'phishing_scan_event.dart';
import 'phishing_scan_state.dart';

class PhishingScanBloc extends Bloc<PhishingScanEvent, PhishingScanState> {
  final PhishingPredictorService _service;

  PhishingScanBloc({PhishingPredictorService? service})
      : _service = service ?? PhishingPredictorService(),
        super(const PhishingScanState()) {
    on<PhishingScanBootstrap>(_onBootstrap);
    on<PhishingScanSubmitted>(_onSubmitted);
    on<PhishingScanVirusTotalSubmitted>(_onVirusTotalSubmitted);
    on<PhishingScanClear>(_onClear);
  }

  Future<void> _onBootstrap(
    PhishingScanBootstrap event,
    Emitter<PhishingScanState> emit,
  ) async {
    emit(state.copyWith(status: PhishingScanStatus.loadingModel));
    try {
      await _service.initialize();
      emit(state.copyWith(
        status: PhishingScanStatus.ready,
      ));
    } catch (e) {
      emit(state.copyWith(
        status: PhishingScanStatus.loadModelFailure,
        error: () => 'Không thể tải mô hình ONNX: $e',
      ));
    }
  }

  Future<void> _onSubmitted(
    PhishingScanSubmitted event,
    Emitter<PhishingScanState> emit,
  ) async {
    final url = event.url.trim();
    if (url.isEmpty) {
      emit(state.copyWith(
        status: PhishingScanStatus.failure,
        error: () => 'Vui lòng nhập URL cần kiểm tra.',
      ));
      return;
    }

    emit(state.copyWith(
      status: PhishingScanStatus.scanning,
      result: () => null,
      error: () => null,
    ));

    await Future.delayed(const Duration(seconds: 1));

    try {
      final result = await _service.analyzeUrlWithKnn(url, event.rs);

      emit(state.copyWith(
        status: PhishingScanStatus.success,
        result: () => result,
      ));
    } on PhishingServiceException catch (e) {
      emit(state.copyWith(
        status: PhishingScanStatus.failure,
        error: () => e.message,
      ));
    } catch (e) {
      emit(state.copyWith(
        status: PhishingScanStatus.failure,
        error: () => 'Lỗi không xác định: $e',
      ));
    }
  }

  Future<void> _onVirusTotalSubmitted(
    PhishingScanVirusTotalSubmitted event,
    Emitter<PhishingScanState> emit,
  ) async {
    final url = event.url.trim();
    if (url.isEmpty) {
      emit(state.copyWith(
        status: PhishingScanStatus.failure,
        error: () => 'Vui lòng nhập URL cần quét.',
      ));
      return;
    }

    emit(state.copyWith(
      status: PhishingScanStatus.scanning,
      result: () => null,
      error: () => null,
    ));

    // huynq - Chay so khop KNN offline truoc (luon thuc hien)
    late PhishingPredictionResult knnResult;
    try {
      knnResult = await _service.analyzeUrlWithKnn(url, null);
    } catch (e) {
      // huynq - Neu ca KNN cung loi (hiem gap vi chay offline)
      emit(state.copyWith(
        status: PhishingScanStatus.failure,
        error: () => 'Không phân tích được dữ liệu trang web',
      ));
      return;
    }

    final knnVote = ModelVote(
      modelId: 'knn_search',
      displayName: 'KNN & Thuật toán so khớp (Offline)',
      label: knnResult.consensusLabel,
    );

    // huynq - Không call API nữa, giả lập kết quả website bị chặn/không hoạt động cho Random Forest
    await Future.delayed(const Duration(seconds: 1));

    final vtVote = const ModelVote(
      modelId: 'rf_hybrid',
      displayName: 'Random Forest (Hybrid)',
      label: PredictionLabel.failure,
      customLabel: 'Bị chặn / Không hoạt động',
      subtitle: 'Website đã bị chặn truy cập hoặc không còn hoạt động',
    );

    final consensusLabel = knnResult.consensusLabel;
    final totalVotes = knnResult.consensusLabel == PredictionLabel.phishing ? 1 : 0;

    final result = PhishingPredictionResult(
      url: url,
      consensusLabel: consensusLabel,
      phishingVotes: totalVotes,
      totalModels: 2,
      modelVotes: [knnVote, vtVote],
      hybridFeaturesFromLiveFetch: false,
      isWhitelisted: false,
      detail: knnResult.detail,
      levScore: knnResult.levScore,
      matchedDomain: knnResult.matchedDomain,
      rfLabel: PredictionLabel.failure,
    );

    emit(state.copyWith(
      status: PhishingScanStatus.success,
      result: () => result,
    ));
  }

  void _onClear(
    PhishingScanClear event,
    Emitter<PhishingScanState> emit,
  ) {
    emit(state.copyWith(
      status: PhishingScanStatus.ready,
      result: () => null,
      error: () => null,
    ));
  }

  @override
  Future<void> close() {
    _service.dispose();
    return super.close();
  }
}
