class PhishingServiceException implements Exception {
  final String code;
  final String message;

  const PhishingServiceException(this.code, this.message);

  @override
  String toString() => 'PhishingServiceException($code): $message';
}
