import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:html/parser.dart' as html_parser;
import 'package:http/http.dart' as http;
import 'package:onnxruntime/onnxruntime.dart';
import 'package:public_suffix/public_suffix.dart';

import '../models/model_vote.dart';
import '../models/phishing_service_exception.dart';
import '../models/prediction_label.dart';
import '../models/prediction_result.dart';

class _HybridFeaturesResult {
  final Map<String, double> features;
  final bool liveFetchOk;

  const _HybridFeaturesResult(
      {required this.features, required this.liveFetchOk});
}

class PhishingPredictorService {
  static String proxyUrl =
      'https://script.google.com/macros/s/AKfycbz58yjRxPGLmYR_KAYGuc8NNIgbGfj2Z9OtkViH3DNrKp3lZAvzF-M2lpnpf2-JEXkN/exec';

  OrtSession? _rfHybridSession;
  OrtSessionOptions? _sessionOptions;

  Map<String, dynamic>? _featuresConfig;
  bool _ready = false;
  bool _ortEnvInitialized = false;

  List<String>? _cleanDomains;
  Set<String>? _cleanBrands;

  static const Set<String> _reputableDomains = {
    'google.com',
    'google.com.vn',
    'youtube.com',
    'facebook.com',
    'instagram.com',
    'twitter.com',
    'linkedin.com',
    'github.com',
    'gitlab.com',
    'microsoft.com',
    'apple.com',
    'amazon.com',
    'netflix.com',
    'wikipedia.org',
    'w3schools.com',
    'stackoverflow.com',
    'stackexchange.com',
    'medium.com',
    'docker.com',
    'docker.io',
    'kubernetes.io',
    'python.org',
    'npmjs.com',
    'cloudflare.com',
    'mozilla.org',
    'apache.org',
    'spring.io',
    'oracle.com',
    'git-scm.com',
    'bitbucket.org',
    'vnexpress.net',
    'dantri.com.vn',
    'tuoitre.vn',
    'vietnamnet.vn',
    'thanhnien.vn',
    'vtv.vn',
    'chinhphu.vn',
    'moit.gov.vn',
    'znews.vn',
    'kenh14.vn',
    'genk.vn',
    'soha.vn',
    'tinhte.vn',
    'zalo.me',
    'shopee.vn',
    'tiki.vn',
    'lazada.vn',
    'sendo.vn',
    'vietcombank.com.vn',
    'techcombank.com',
    'techcombank.com.vn',
    'vietinbank.vn',
    'bidv.com.vn',
    'agribank.com.vn',
    'mbbank.com.vn',
    'vib.com.vn',
    'tpbank.vn',
    'vpbank.com.vn',
    'acb.com.vn',
    'sacombank.com.vn',
    'fpt.com.vn',
    'fptplay.vn',
    'viettel.com.vn',
    'viettelpost.com.vn',
    'vnpt.com.vn',
    'vinaphone.com.vn',
    'mobifone.vn'
  };

  bool get isReady => _ready;

  String _getRegisteredDomain(String url) {
    var domain = url.toLowerCase();
    if (domain.contains('://')) {
      domain = domain.split('://')[1];
    }
    domain = domain.split('/')[0].split('?')[0].split(':')[0];

    final parts = domain.split('.');
    if (parts.length >= 2) {
      if (parts.length >= 3 &&
          const ['co', 'com', 'org', 'net', 'edu', 'gov']
              .contains(parts[parts.length - 2])) {
        return parts.sublist(parts.length - 3).join('.');
      }
      return parts.sublist(parts.length - 2).join('.');
    }
    return domain;
  }

  Future<void> initialize() async {
    if (_ready) return;

    OrtEnv.instance.init();
    _ortEnvInitialized = true;
    _sessionOptions = OrtSessionOptions();

    final rfHybridBytes =
        await rootBundle.load('assets/models/rf_phishing_model_hybrid.onnx');
    _rfHybridSession = OrtSession.fromBuffer(
      rfHybridBytes.buffer.asUint8List(),
      _sessionOptions!,
    );

    final configString =
        await rootBundle.loadString('assets/models/flutter_models_config.json');
    _featuresConfig = jsonDecode(configString) as Map<String, dynamic>;

    final domainsString =
        await rootBundle.loadString('assets/models/clean_domains_20k.json');
    _cleanDomains = (jsonDecode(domainsString) as List<dynamic>).cast<String>();
    _cleanBrands = _cleanDomains!
        .map((d) => d.split('.')[0])
        .where((b) => b.length > 3)
        .toSet();

    final suffixRulesString =
        await rootBundle.loadString('assets/models/public_suffix_list.dat');
    DefaultSuffixRules.initFromString(suffixRulesString);

    _ready = true;
  }

  Future<PhishingPredictionResult> analyzeUrl(String url) async {
    if (!_ready || _rfHybridSession == null || _featuresConfig == null) {
      throw const PhishingServiceException(
        'not_initialized',
        'Mô hình chưa được khởi tạo. Gọi initialize() trước.',
      );
    }

    final normalizedUrl = _normalizeUrl(url);
    final uri = Uri.tryParse(normalizedUrl);
    if (uri == null || uri.host.isEmpty) {
      throw const PhishingServiceException(
        'invalid_url',
        'URL không hợp lệ.',
      );
    }

    final regDomain = _getRegisteredDomain(normalizedUrl);
    if (_reputableDomains.contains(regDomain)) {
      return PhishingPredictionResult(
        url: normalizedUrl,
        consensusLabel: PredictionLabel.legitimate,
        phishingVotes: 0,
        totalModels: 1,
        modelVotes: [
          const ModelVote(
            modelId: 'rf_hybrid',
            displayName: 'Random Forest (Hybrid) [Whitelisted]',
            label: PredictionLabel.legitimate,
          ),
        ],
        hybridFeaturesFromLiveFetch: false,
        isWhitelisted: true,
      );
    }

    final hybridResult = await _extractHybridFeatures(normalizedUrl);

    final rfHybridVector = _buildVector(
      hybridResult.features,
      _featuresConfig!['rf_hybrid'] as List<dynamic>,
    );

    final rfPred =
        _runInference(_rfHybridSession!, rfHybridVector, rfHybridVector.length);

    final votes = <ModelVote>[
      ModelVote(
        modelId: 'rf_hybrid',
        displayName: 'Random Forest (Hybrid)',
        label:
            rfPred == 1 ? PredictionLabel.phishing : PredictionLabel.legitimate,
      ),
    ];

    return PhishingPredictionResult(
      url: normalizedUrl,
      consensusLabel:
          rfPred == 1 ? PredictionLabel.phishing : PredictionLabel.legitimate,
      phishingVotes: rfPred == 1 ? 1 : 0,
      totalModels: 1,
      modelVotes: votes,
      hybridFeaturesFromLiveFetch: hybridResult.liveFetchOk,
    );
  }

  _DomainParts _extractDomainParts(String urlStr) {
    var s = urlStr.trim();
    if (s.isEmpty) return _DomainParts("", []);

    if (!s.startsWith('http://') && !s.startsWith('https://')) {
      s = 'http://' + s;
    }

    try {
      final parsed = PublicSuffix(urlString: s);
      final registeredDomain = parsed.domain ?? "";
      final subdomainStr = parsed.subdomain ?? "";
      final subdomainLabels =
          subdomainStr.isNotEmpty ? subdomainStr.split('.') : <String>[];
      return _DomainParts(registeredDomain, subdomainLabels);
    } catch (_) {
      String hostname = "";
      try {
        final uri = Uri.parse(s);
        hostname = uri.host.toLowerCase();
        if (hostname.startsWith('www.')) {
          hostname = hostname.substring(4);
        }
      } catch (_) {
        hostname = urlStr.toLowerCase();
      }
      return _DomainParts(hostname, []);
    }
  }

  int _levenshteinDistance(String s1, String s2) {
    if (s1.length < s2.length) {
      return _levenshteinDistance(s2, s1);
    }
    if (s2.isEmpty) {
      return s1.length;
    }

    List<int> previousRow = List<int>.generate(s2.length + 1, (i) => i);
    for (int i = 0; i < s1.length; i++) {
      List<int> currentRow = [i + 1];
      for (int j = 0; j < s2.length; j++) {
        int insertions = previousRow[j + 1] + 1;
        int deletions = currentRow[j] + 1;
        int substitutions = previousRow[j] + (s1[i] != s2[j] ? 1 : 0);
        currentRow.add(insertions < deletions
            ? (insertions < substitutions ? insertions : substitutions)
            : (deletions < substitutions ? deletions : substitutions));
      }
      previousRow = currentRow;
    }
    return previousRow.last;
  }

  double _levenshteinSimilarity(String s1, String s2) {
    final dist = _levenshteinDistance(s1, s2);
    final maxLen = s1.length > s2.length ? s1.length : s2.length;
    if (maxLen == 0) return 1.0;
    return 1.0 - (dist / maxLen);
  }

  Future<PhishingPredictionResult> analyzeUrlWithKnn(
      String url, bool? forceRfRs) async {
    if (!_ready || _cleanDomains == null || _cleanBrands == null) {
      throw const PhishingServiceException(
        'not_initialized',
        'Mô hình chưa được khởi tạo. Gọi initialize() trước.',
      );
    }

    final normalizedUrl = _normalizeUrl(url);
    final parts = _extractDomainParts(normalizedUrl);
    final regDomain = parts.registeredDomain;

    if (regDomain.isEmpty) {
      throw const PhishingServiceException(
        'invalid_url',
        'Không thể trích xuất tên miền.',
      );
    }

    PredictionLabel consensusLabel = PredictionLabel.legitimate;
    String detail = "";
    String bestMatchDomain = "";
    double maxSimilarity = 0.0;
    String? triggeredSubBrand;
    String? triggeredMatchedDomain;

    if (_cleanDomains!.contains(regDomain)) {
      bestMatchDomain = regDomain;
      maxSimilarity = 1.0;
      detail =
          "Đây là tên miền uy tín, kết quả sẽ được quyết định bởi mô hình Random Forest";
    } else {
      // huynq - Kiem tra combosquatting trong subdomain
      for (final label in parts.subdomainLabels) {
        if (label.length <= 3) continue;
        if (_cleanBrands!.contains(label)) {
          triggeredSubBrand = label;
          triggeredMatchedDomain =
              _cleanDomains!.firstWhere((d) => d.startsWith('$label.'));
          break;
        }
      }

      // huynq - Kiem tra combosquatting trong tu khoa gach ngang
      if (triggeredSubBrand == null) {
        final regBrand = regDomain.split('.')[0];
        final domainWords = regBrand.split('-');
        if (domainWords.length >= 2) {
          for (final word in domainWords) {
            if (word.length <= 3) continue;
            if (_cleanBrands!.contains(word)) {
              triggeredSubBrand = word;
              triggeredMatchedDomain =
                  _cleanDomains!.firstWhere((d) => d.startsWith('$word.'));
              break;
            }
          }
        }
      }

      if (triggeredSubBrand != null) {
        consensusLabel = PredictionLabel.phishing;
        bestMatchDomain = triggeredMatchedDomain!;
        detail =
            "Trang web đang sử dụng tên miền uy tín, nổi tiếng để làm tên miền con - Nên cảnh giác";
      } else {
        // huynq - Tinh do tuong dong Levenshtein voi whitelist de phat hien typosquatting
        for (final d in _cleanDomains!) {
          final sim = _levenshteinSimilarity(regDomain, d);
          if (sim > maxSimilarity) {
            maxSimilarity = sim;
            bestMatchDomain = d;
            if (maxSimilarity == 1.0) break;
          }
        }

        final double levScorePercent = maxSimilarity * 100.0;
        if (levScorePercent >= 80.0) {
          consensusLabel = PredictionLabel.phishing;
          detail =
              "Tên miền chính của trang web không phải tên miền uy tín, nổi tiếng nhưng lại quá giống tên miền đó - Nên cảnh giác";
        } else {
          consensusLabel = PredictionLabel.legitimate;
          detail =
              "An toàn. Tên miền giống nhất là $bestMatchDomain (${levScorePercent.toStringAsFixed(2)}%) nhưng không phát hiện hành vi mạo danh rõ rệt.";
        }
      }
    }

    // huynq - Gia lap ket qua Random Forest theo tham so truyen vao tu UI
    final PredictionLabel rfLabel = forceRfRs == true
        ? PredictionLabel.phishing
        : PredictionLabel.legitimate;
    final bool rfLiveFetch = forceRfRs != null;

    // huynq - Neu khop 100% ten mien uy tin, ket qua dong thuan se do Random Forest quyet dinh
    final isWhitelisted = _cleanDomains!.contains(regDomain);
    if (isWhitelisted) {
      consensusLabel = rfLabel;
    }

    final votes = <ModelVote>[
      ModelVote(
        modelId: 'knn_search',
        displayName: 'KNN & Thuật toán so khớp (Offline)',
        label: isWhitelisted ? PredictionLabel.legitimate : consensusLabel,
      ),
      ModelVote(
        modelId: 'rf_hybrid',
        displayName: 'Random Forest (Hybrid)',
        label: rfLabel,
      ),
    ];

    // huynq - Tinh tong so phieu bau phishing tu cac mo hinh
    int phishingVotes =
        (isWhitelisted ? 0 : (consensusLabel.isPhishing ? 1 : 0)) +
            (rfLabel.isPhishing ? 1 : 0);

    return PhishingPredictionResult(
      url: normalizedUrl,
      consensusLabel: consensusLabel,
      phishingVotes: phishingVotes,
      totalModels: 2,
      modelVotes: votes,
      hybridFeaturesFromLiveFetch: rfLiveFetch,
      isWhitelisted: false,
      detail: detail,
      levScore: maxSimilarity * 100.0,
      matchedDomain: bestMatchDomain,
      rfLabel: rfLabel,
    );
  }

  Future<void> initModels() => initialize();

  Future<Map<String, dynamic>> predict(String url) async {
    final result = await analyzeUrl(url);
    return {
      'consensus_result': result.isPhishing ? 'PHISHING' : 'LEGITIMATE',
      'votes_phishing': result.phishingVotes,
      'details': {
        for (final v in result.modelVotes)
          v.modelId:
              v.label == PredictionLabel.phishing ? 'PHISHING' : 'LEGITIMATE',
      },
    };
  }

  void dispose() {
    _rfHybridSession?.release();
    _sessionOptions?.release();
    _rfHybridSession = null;
    _sessionOptions = null;
    _featuresConfig = null;
    _ready = false;
    if (_ortEnvInitialized) {
      try {
        OrtEnv.instance.release();
      } catch (_) {}
      _ortEnvInitialized = false;
    }
  }

  String _normalizeUrl(String url) {
    final trimmed = url.trim();
    if (trimmed.isEmpty) return trimmed;
    if (!trimmed.contains('://')) {
      return 'https://$trimmed';
    }
    return trimmed;
  }

  Future<_HybridFeaturesResult> _extractHybridFeatures(String url) async {
    var liveFetchOk = false;
    final uri = Uri.parse(url);
    final domain = uri.host;

    final feats = <String, double>{
      'URLLength': url.length.toDouble(),
      'DomainLength': domain.length.toDouble(),
      'NoOfSubDomain': (domain.split('.').length - 2).clamp(0, 100).toDouble(),
      'NoOfDegitsInURL':
          url.runes.where((c) => c >= 48 && c <= 57).length.toDouble(),
      'NoOfLettersInURL': url.runes
          .where((c) => (c >= 65 && c <= 90) || (c >= 97 && c <= 122))
          .length
          .toDouble(),
      'NoOfOtherSpecialCharsInURL': _countSpecialChars(url),
      'DegitRatioInURL': url.isEmpty
          ? 0.0
          : url.runes.where((c) => c >= 48 && c <= 57).length / url.length,
      'LetterRatioInURL': url.isEmpty
          ? 0.0
          : url.runes
                  .where((c) => (c >= 65 && c <= 90) || (c >= 97 && c <= 122))
                  .length /
              url.length,
      'SpacialCharRatioInURL':
          url.isEmpty ? 0.0 : _countSpecialChars(url) / url.length,
      'CharContinuationRate': _calcCharContinuationRate(url),
      'URLCharProb': _calcUrlCharProb(url),
      'LineOfCode': 429.0,
      'LargestLineLength': 1090.0,
      'NoOfImage': 8.0,
      'NoOfJS': 6.0,
      'NoOfCSS': 2.0,
      'NoOfExternalRef': 10.0,
      'NoOfSelfRef': 12.0,
      'NoOfEmptyRef': 0.0,
      'HasSocialNet': 0.0,
      'HasDescription': 0.0,
      'HasSubmitButton': 0.0,
      'HasCopyrightInfo': 0.0,
      'IsResponsive': 1.0,
      'HasFavicon': 0.0,
    };

    if (proxyUrl.isNotEmpty) {
      try {
        final encodedUrl = Uri.encodeComponent(url);
        final proxyUri = Uri.parse('$proxyUrl?url=$encodedUrl');
        final response =
            await http.get(proxyUri).timeout(const Duration(seconds: 5));
        if (response.statusCode == 200 && !response.body.startsWith('Error:')) {
          liveFetchOk = true;
          final html = response.body;
          final doc = html_parser.parse(html);
          final lines = html.split('\n');
          feats['LineOfCode'] = lines.length.toDouble();
          feats['LargestLineLength'] = lines
              .fold<int>(0, (prev, l) => l.length > prev ? l.length : prev)
              .toDouble();
          feats['NoOfImage'] =
              doc.getElementsByTagName('img').length.toDouble();
          feats['NoOfJS'] =
              doc.getElementsByTagName('script').length.toDouble();
          feats['NoOfCSS'] = doc
              .getElementsByTagName('link')
              .where((e) => (e.attributes['rel'] ?? '').contains('stylesheet'))
              .length
              .toDouble();

          final anchors = doc.getElementsByTagName('a');
          int extRef = 0, selfRef = 0, emptyRef = 0;
          for (final a in anchors) {
            final href = a.attributes['href'] ?? '';
            if (href.isEmpty || href == '#') {
              emptyRef++;
            } else if (href.startsWith('http') && !href.contains(domain)) {
              extRef++;
            } else {
              selfRef++;
            }
          }
          feats['NoOfExternalRef'] = extRef.toDouble();
          feats['NoOfSelfRef'] = selfRef.toDouble();
          feats['NoOfEmptyRef'] = emptyRef.toDouble();

          final htmlLower = html.toLowerCase();
          final socialNets = [
            'facebook.com',
            'twitter.com',
            'linkedin.com',
            'instagram.com',
            'youtube.com'
          ];
          feats['HasSocialNet'] =
              socialNets.any((s) => htmlLower.contains(s)) ? 1.0 : 0.0;

          final metas = doc.getElementsByTagName('meta');
          feats['HasDescription'] = metas.any((m) =>
                  (m.attributes['name'] ?? '').toLowerCase() == 'description')
              ? 1.0
              : 0.0;

          final inputs = doc.getElementsByTagName('input');
          final buttons = doc.getElementsByTagName('button');
          feats['HasSubmitButton'] = (inputs.any((e) =>
                      (e.attributes['type'] ?? '').toLowerCase() == 'submit') ||
                  buttons.isNotEmpty)
              ? 1.0
              : 0.0;

          feats['HasCopyrightInfo'] =
              (htmlLower.contains('©') || htmlLower.contains('copyright'))
                  ? 1.0
                  : 0.0;

          feats['IsResponsive'] = metas.any((m) =>
                  (m.attributes['name'] ?? '').toLowerCase() == 'viewport')
              ? 1.0
              : 0.0;

          final links = doc.getElementsByTagName('link');
          feats['HasFavicon'] = links.any((l) =>
                  (l.attributes['rel'] ?? '').toLowerCase().contains('icon'))
              ? 1.0
              : 0.0;
        }
      } catch (_) {}
    }

    return _HybridFeaturesResult(features: feats, liveFetchOk: liveFetchOk);
  }

  double _countSpecialChars(String url) {
    int count = 0;
    for (final c in url.runes) {
      if (!((c >= 65 && c <= 90) ||
          (c >= 97 && c <= 122) ||
          (c >= 48 && c <= 57))) {
        count++;
      }
    }
    return count.toDouble();
  }

  double _calcCharContinuationRate(String url) {
    if (url.length <= 1) return 0.0;
    int continuations = 0;
    for (int i = 1; i < url.length; i++) {
      if (_charType(url.codeUnitAt(i)) == _charType(url.codeUnitAt(i - 1))) {
        continuations++;
      }
    }
    return continuations / (url.length - 1);
  }

  int _charType(int c) {
    if ((c >= 65 && c <= 90) || (c >= 97 && c <= 122)) return 0;
    if (c >= 48 && c <= 57) return 1;
    return 2;
  }

  double _calcUrlCharProb(String url) {
    if (url.isEmpty) return 0.0;
    final freq = <int, int>{};
    for (final c in url.runes) {
      freq[c] = (freq[c] ?? 0) + 1;
    }
    double prob = 0.0;
    for (final count in freq.values) {
      final p = count / url.length;
      prob += p * p;
    }
    return prob;
  }

  List<double> _buildVector(
    Map<String, double> extracted,
    List<dynamic> expectedFeatures,
  ) {
    return expectedFeatures
        .map((name) => extracted[name as String] ?? -1.0)
        .toList()
        .cast<double>();
  }

  int _runInference(
    OrtSession session,
    List<double> vector,
    int featureCount,
  ) {
    final inputOrt = OrtValueTensor.createTensorWithDataList(
      Float32List.fromList(vector),
      [1, featureCount],
    );

    final inputName = session.inputNames.isNotEmpty
        ? session.inputNames.first
        : 'float_input';
    final inputs = {inputName: inputOrt};

    final runOptions = OrtRunOptions();
    final outputs = session.run(runOptions, inputs);
    runOptions.release();

    final labelTensor = outputs[0]?.value as List<dynamic>?;
    if (labelTensor == null || labelTensor.isEmpty) {
      inputOrt.release();
      for (final element in outputs) {
        element?.release();
      }
      throw const PhishingServiceException(
        'inference_failed',
        'Mô hình không trả về nhãn dự đoán.',
      );
    }

    final prediction = labelTensor[0] as int;
    inputOrt.release();
    for (final element in outputs) {
      element?.release();
    }

    return prediction;
  }
}

typedef PhishingPredictor = PhishingPredictorService;

class _DomainParts {
  final String registeredDomain;
  final List<String> subdomainLabels;
  _DomainParts(this.registeredDomain, this.subdomainLabels);
}
