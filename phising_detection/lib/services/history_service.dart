import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class HistoryService {
  static const String _historyKey = 'phishing_scan_history';

  // Singleton instance
  static final HistoryService _instance = HistoryService._internal();
  factory HistoryService() => _instance;
  HistoryService._internal();

  Future<void> saveScan({
    required String url,
    required String resultType, // 'phishing', 'safe', 'no_data'
    String? detail,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    final List<String> historyJson = prefs.getStringList(_historyKey) ?? [];

    final Map<String, dynamic> newItem = {
      'url': url,
      'timestamp': DateTime.now().toIso8601String(),
      'resultType': resultType,
      'detail': detail,
    };

    // Remove duplicates of the same URL to keep history tidy
    historyJson.removeWhere((item) {
      try {
        final decoded = json.decode(item) as Map<String, dynamic>;
        return decoded['url'] == url;
      } catch (_) {
        return false;
      }
    });

    // Insert new item at the top of the list
    historyJson.insert(0, json.encode(newItem));

    // Limit history size to 50 items
    if (historyJson.length > 50) {
      historyJson.removeLast();
    }

    await prefs.setStringList(_historyKey, historyJson);
  }

  Future<List<Map<String, dynamic>>> getHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final List<String> historyJson = prefs.getStringList(_historyKey) ?? [];
    return historyJson.map((item) {
      try {
        return json.decode(item) as Map<String, dynamic>;
      } catch (_) {
        return <String, dynamic>{};
      }
    }).toList();
  }

  Future<void> clearHistory() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_historyKey);
  }
}
