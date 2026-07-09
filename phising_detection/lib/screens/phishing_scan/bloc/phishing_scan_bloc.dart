import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:phising_detection/models/prediction_result.dart';
import 'package:phising_detection/models/prediction_label.dart';
import 'package:phising_detection/models/model_vote.dart';
import 'package:phising_detection/usecases/virus_total/scan_url_usecase.dart';
import 'package:phising_detection/repositories/virus_total_repository_impl.dart';
import '../../../models/phishing_service_exception.dart';
import '../../../services/phising_predictor.dart';
import '../../../services/history_service.dart';
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
    on<PhishingScanHistoryRequested>(_onHistoryRequested);
    on<PhishingScanHistoryCleared>(_onHistoryCleared);
  }

  Future<void> _onBootstrap(
    PhishingScanBootstrap event,
    Emitter<PhishingScanState> emit,
  ) async {
    emit(state.copyWith(status: PhishingScanStatus.loadingModel));
    try {
      await _service.initialize();
      final history = await HistoryService().getHistory();
      emit(state.copyWith(
        status: PhishingScanStatus.ready,
        history: history,
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

    try {
      final result = await _service.analyzeUrlWithKnn(url, event.rs);

      final resultType = result.isPhishing ? 'phishing' : 'safe';
      final detailText =
          result.detail ?? 'Được đánh giá là an toàn bởi các mô hình kiểm tra';

      await HistoryService().saveScan(
        url: url,
        resultType: resultType,
        detail: detailText,
      );
      final updatedHistory = await HistoryService().getHistory();

      emit(state.copyWith(
        status: PhishingScanStatus.success,
        result: () => result,
        history: updatedHistory,
      ));
    } on PhishingServiceException catch (e) {
      // Save failed scan to history
      await HistoryService().saveScan(
        url: url,
        resultType: 'no_data',
        detail: e.message,
      );
      final updatedHistory = await HistoryService().getHistory();

      emit(state.copyWith(
        status: PhishingScanStatus.failure,
        error: () => e.message,
        history: updatedHistory,
      ));
    } catch (e) {
      // Save failed/error scan to history
      await HistoryService().saveScan(
        url: url,
        resultType: 'no_data',
        detail: 'Lỗi không xác định: $e',
      );
      final updatedHistory = await HistoryService().getHistory();

      emit(state.copyWith(
        status: PhishingScanStatus.failure,
        error: () => 'Lỗi không xác định: $e',
        history: updatedHistory,
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
        modelId: 'vt_scan',
        displayName: 'VirusTotal API (${vtReport.stats.malicious}/$totalEngines engines)',
        label: vtLabel,
      );

      final result = PhishingPredictionResult(
        url: url,
        consensusLabel: vtLabel,
        phishingVotes: vtReport.stats.malicious,
        totalModels: totalEngines,
        modelVotes: [vtVote],
        hybridFeaturesFromLiveFetch: true,
        isWhitelisted: false,
        detail: 'VirusTotal phát hiện ${vtReport.stats.malicious}/$totalEngines công cụ báo độc hại.',
        levScore: null,
        matchedDomain: null,
        rfLabel: vtLabel,
      );

      final resultType = result.isPhishing ? 'phishing' : 'safe';
      final detailText = result.detail ?? '';

      await HistoryService().saveScan(
        url: url,
        resultType: resultType,
        detail: detailText,
      );
      final updatedHistory = await HistoryService().getHistory();

      emit(state.copyWith(
        status: PhishingScanStatus.success,
        result: () => result,
        history: updatedHistory,
      ));
    } catch (e) {
      // Save failed/error scan to history
      await HistoryService().saveScan(
        url: url,
        resultType: 'no_data',
        detail: 'Lỗi quét VirusTotal: $e',
      );
      final updatedHistory = await HistoryService().getHistory();

      emit(state.copyWith(
        status: PhishingScanStatus.failure,
        error: () => 'Lỗi quét VirusTotal: $e',
        history: updatedHistory,
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

  Future<void> _onHistoryRequested(
    PhishingScanHistoryRequested event,
    Emitter<PhishingScanState> emit,
  ) async {
    final history = await HistoryService().getHistory();
    emit(state.copyWith(history: history));
  }

  Future<void> _onHistoryCleared(
    PhishingScanHistoryCleared event,
    Emitter<PhishingScanState> emit,
  ) async {
    await HistoryService().clearHistory();
    emit(state.copyWith(history: const []));
  }

  @override
  Future<void> close() {
    _service.dispose();
    return super.close();
  }
}
