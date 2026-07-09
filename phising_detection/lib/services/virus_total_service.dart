import 'dart:convert';
import 'package:http/http.dart' as http;

class VirusTotalService {
  static const String _defaultApiKey = 'c652949ac51579c29c4d0bc1a0373a7e9f25882981cd55fadbcc81face8103bd';
  
  final String _apiKey;
  final http.Client _client;

  VirusTotalService({
    String apiKey = _defaultApiKey,
    http.Client? client,
  })  : _apiKey = apiKey,
        _client = client ?? http.Client();

  Future<String?> submitUrl(String url) async {
    final response = await _client.post(
      Uri.parse('https://www.virustotal.com/api/v3/urls'),
      headers: {
        'x-apikey': _apiKey,
      },
      body: {
        'url': url,
      },
    );

    if (response.statusCode == 200) {
      final json = jsonDecode(response.body);
      return json['data']['id'] as String?;
    }

    throw Exception(response.body);
  }

  Future<Map<String, dynamic>> getAnalysis(String id) async {
    final response = await _client.get(
      Uri.parse('https://www.virustotal.com/api/v3/analyses/$id'),
      headers: {
        'x-apikey': _apiKey,
      },
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body) as Map<String, dynamic>;
    }

    throw Exception(response.body);
  }
}
