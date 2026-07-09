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

    // huynq - Gui request den VirusTotal API
    try {
      final scanUseCase = ScanUrlUseCase(VirusTotalRepositoryImpl());
      final vtReport = await scanUseCase(
        url,
        pollInterval: const Duration(seconds: 2),
        timeout: const Duration(seconds: 30),
      );

      final vtLabel = vtReport.stats.malicious > 0
          ? PredictionLabel.phishing
          : PredictionLabel.legitimate;

      final totalEngines = vtReport.stats.malicious +
          vtReport.stats.harmless +
          vtReport.stats.suspicious +
          vtReport.stats.undetected;

      final vtVote = ModelVote(
        modelId: 'rf_hybrid',
        displayName:
            'Random Forest (Hybrid) (${vtReport.stats.malicious}/$totalEngines)',
        label: vtLabel,
      );

      // huynq - Dong thuan: Canh bao neu mot trong hai mo hinh bao doc hai
      final consensusLabel = (vtLabel == PredictionLabel.phishing ||
              knnResult.consensusLabel == PredictionLabel.phishing)
          ? PredictionLabel.phishing
          : PredictionLabel.legitimate;

      final totalVotes = (vtLabel == PredictionLabel.phishing ? 1 : 0) +
          (knnResult.consensusLabel == PredictionLabel.phishing ? 1 : 0);

      final result = PhishingPredictionResult(
        url: url,
        consensusLabel: consensusLabel,
        phishingVotes: totalVotes,
        totalModels: 2,
        modelVotes: [knnVote, vtVote],
        hybridFeaturesFromLiveFetch: true,
        isWhitelisted: false,
        detail: knnResult
            .detail, // huynq - Chi lay chi tiet so khop KNN de hien thi tren card KNN
        levScore: knnResult.levScore,
        matchedDomain: knnResult.matchedDomain,
        rfLabel: vtLabel, // huynq - Hien thi ket qua VirusTotal o card duoi
      );

      emit(state.copyWith(
        status: PhishingScanStatus.success,
        result: () => result,
      ));
    } catch (e) {
      // huynq - Khi call API loi: chi hien thi loi o card random forest/VirusTotal, KNN van chay binh thuong
      final vtVoteFailure = const ModelVote(
        modelId: 'rf_hybrid',
        displayName: 'Random Forest (Hybrid) (Lỗi kết nối)',
        label: PredictionLabel.failure,
      );

      final result = PhishingPredictionResult(
        url: url,
        consensusLabel:
            knnResult.consensusLabel, // Lấy kết quả KNN làm kết quả chính
        phishingVotes:
            knnResult.consensusLabel == PredictionLabel.phishing ? 1 : 0,
        totalModels: 2,
        modelVotes: [knnVote, vtVoteFailure],
        hybridFeaturesFromLiveFetch: true,
        isWhitelisted: false,
        detail: knnResult
            .detail, // huynq - Giu nguyen chi tiet so khop KNN ke ca khi API loi
        levScore: knnResult.levScore,
        matchedDomain: knnResult.matchedDomain,
        rfLabel:
            PredictionLabel.failure, // Gan nhan loi de hien thi card duoi loi
      );

      emit(state.copyWith(
        status: PhishingScanStatus
            .success, // Van phat ra success de giao dien hien thi
        result: () => result,
      ));
    }
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
