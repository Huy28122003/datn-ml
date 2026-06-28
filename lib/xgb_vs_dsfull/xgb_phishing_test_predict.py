import os
import sys
import pickle
import argparse
import numpy as np
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xgb_url_extract_test_features import extract_features_for_model
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'xgb_vs_dsfull')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'xgb_phishing_model.pkl')
REPUTABLE_DOMAINS = {'google.com', 'google.com.vn', 'youtube.com', 'facebook.com', 'instagram.com', 'twitter.com', 'linkedin.com', 'github.com', 'gitlab.com', 'microsoft.com', 'apple.com', 'amazon.com', 'netflix.com', 'wikipedia.org', 'w3schools.com', 'stackoverflow.com', 'stackexchange.com', 'medium.com', 'docker.com', 'docker.io', 'kubernetes.io', 'python.org', 'npmjs.com', 'cloudflare.com', 'mozilla.org', 'apache.org', 'spring.io', 'oracle.com', 'git-scm.com', 'bitbucket.org'}

def load_model():
    if not os.path.exists(MODEL_PATH):
        sys.exit(1)
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    features = model_data['features']
    params = model_data['params']
    return (model, features, params)

def get_registered_domain(domain):
    if not domain:
        return ''
    domain = domain.lower()
    parts = domain.split('.')
    if len(parts) >= 2:
        if len(parts) >= 3 and parts[-2] in ('co', 'com', 'org', 'net', 'edu', 'gov'):
            return '.'.join(parts[-3:])
        return '.'.join(parts[-2:])
    return domain

def predict_url(url, model, model_features):
    feature_vector = extract_features_for_model(url, model_features)
    from xgb_url_extract_test_features import _parse_url_parts
    parts = _parse_url_parts(url)
    domain = parts.get('domain', '')
    reg_domain = get_registered_domain(domain)
    is_whitelisted = reg_domain in REPUTABLE_DOMAINS
    if is_whitelisted:
        prediction = 0
        probabilities = [1.0, 0.0]
        label = 'LEGITIMATE ✅ (Danh tiếng uy tín)'
    else:
        X = np.array([feature_vector])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        label = 'PHISHING 🚨' if prediction == 1 else 'LEGITIMATE ✅'
    result = {'url': url, 'prediction': int(prediction), 'label': label, 'probability_legitimate': float(probabilities[0]), 'probability_phishing': float(probabilities[1]), 'confidence': float(max(probabilities)), 'features_extracted': {name: val for (name, val) in zip(model_features, feature_vector)}, 'features_used_count': len(model_features), 'is_whitelisted': is_whitelisted}
    return result

def display_result(result):
    url_trimmed = result['url'][:60] + ('...' if len(result['url']) > 60 else '')
    if result['prediction'] == 1:
    features = result['features_extracted']
    for (i, (name, val)) in enumerate(features.items(), 1):
        if isinstance(val, float):

def interactive_mode(model, model_features):
    while True:
        url = input('🔗 Nhập URL: ').strip()
        if url.lower() in ('quit', 'exit', 'q'):
            break
        if not url:
            continue
        try:
            result = predict_url(url, model, model_features)
            display_result(result)
        except Exception as e:

def main():
    parser = argparse.ArgumentParser(description='XGBoost Phishing URL Predictor')
    parser.add_argument('--url', type=str, help='Đường dẫn URL cần kiểm tra')
    args = parser.parse_args()
    (model, features, params) = load_model()
    if args.url:
        result = predict_url(args.url, model, features)
        display_result(result)
    else:
        interactive_mode(model, features)
if __name__ == '__main__':
    main()
