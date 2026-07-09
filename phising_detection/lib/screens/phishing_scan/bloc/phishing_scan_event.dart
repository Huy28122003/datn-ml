abstract class PhishingScanEvent {}

class PhishingScanBootstrap extends PhishingScanEvent {}

class PhishingScanSubmitted extends PhishingScanEvent {
  final String url;
  final bool? rs;
  PhishingScanSubmitted(this.url,this.rs);
}

class PhishingScanClear extends PhishingScanEvent {}

class PhishingScanVirusTotalSubmitted extends PhishingScanEvent {
  final String url;
  PhishingScanVirusTotalSubmitted(this.url);
}
