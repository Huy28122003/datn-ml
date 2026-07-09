class VirusTotalReport {
  final String id;
  final String status;
  final int? date;
  final VirusTotalStats stats;
  final Map<String, VirusTotalEngineResult> results;

  const VirusTotalReport({
    required this.id,
    required this.status,
    this.date,
    required this.stats,
    required this.results,
  });

  factory VirusTotalReport.fromJson(Map<String, dynamic> json) {
    final data = json['data'] as Map<String, dynamic>? ?? {};
    final attributes = data['attributes'] as Map<String, dynamic>? ?? {};
    final statsMap = attributes['stats'] as Map<String, dynamic>? ?? {};
    final resultsMap = attributes['results'] as Map<String, dynamic>? ?? {};

    final parsedResults = resultsMap.map((key, value) {
      return MapEntry(
        key,
        VirusTotalEngineResult.fromJson(value as Map<String, dynamic>),
      );
    });

    return VirusTotalReport(
      id: data['id'] as String? ?? '',
      status: attributes['status'] as String? ?? 'unknown',
      date: attributes['date'] as int?,
      stats: VirusTotalStats.fromJson(statsMap),
      results: parsedResults,
    );
  }

  bool get isCompleted => status == 'completed';
}

class VirusTotalStats {
  final int harmless;
  final int malicious;
  final int suspicious;
  final int undetected;
  final int timeout;

  const VirusTotalStats({
    this.harmless = 0,
    this.malicious = 0,
    this.suspicious = 0,
    this.undetected = 0,
    this.timeout = 0,
  });

  factory VirusTotalStats.fromJson(Map<String, dynamic> json) {
    return VirusTotalStats(
      harmless: json['harmless'] as int? ?? 0,
      malicious: json['malicious'] as int? ?? 0,
      suspicious: json['suspicious'] as int? ?? 0,
      undetected: json['undetected'] as int? ?? 0,
      timeout: json['timeout'] as int? ?? 0,
    );
  }
}

class VirusTotalEngineResult {
  final String category;
  final String engineName;
  final String result;
  final String method;

  const VirusTotalEngineResult({
    required this.category,
    required this.engineName,
    required this.result,
    required this.method,
  });

  factory VirusTotalEngineResult.fromJson(Map<String, dynamic> json) {
    return VirusTotalEngineResult(
      category: json['category'] as String? ?? '',
      engineName: json['engine_name'] as String? ?? '',
      result: json['result'] as String? ?? '',
      method: json['method'] as String? ?? '',
    );
  }
}
