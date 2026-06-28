import os
import sys
import re
import json
import pickle
import warnings
import urllib.request
import urllib.parse
import numpy as np
warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(BASE_DIR) == 'lib':
    BASE_DIR = os.path.dirname(BASE_DIR)
elif os.path.basename(BASE_DIR) in ('rf_vs_dsfull', 'xgb_vs_dsfull', 'rf_vs_hybrid'):
    BASE_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.insert(0, os.path.join(BASE_DIR, 'lib', 'rf_vs_dsfull'))
sys.path.insert(0, os.path.join(BASE_DIR, 'lib', 'xgb_vs_dsfull'))
try:
    import rf_url_extract_test_features as rf_ext
    import xgb_url_extract_test_features as xgb_ext
except ImportError as e:
    sys.exit(1)
RF_PURE_PATH = os.path.join(BASE_DIR, 'output', 'rf_vs_dsfull', 'rf_phishing_model.pkl')
XGB_PURE_PATH = os.path.join(BASE_DIR, 'output', 'xgb_vs_dsfull', 'xgb_phishing_model.pkl')
RF_HYBRID_PATH = os.path.join(BASE_DIR, 'output', 'rf_vs_hybrid', 'rf_phishing_model.pkl')
REPUTABLE_DOMAINS = {'google.com', 'google.com.vn', 'youtube.com', 'facebook.com', 'instagram.com', 'twitter.com', 'linkedin.com', 'github.com', 'gitlab.com', 'microsoft.com', 'apple.com', 'amazon.com', 'netflix.com', 'wikipedia.org', 'w3schools.com', 'stackoverflow.com', 'stackexchange.com', 'medium.com', 'docker.com', 'docker.io', 'kubernetes.io', 'python.org', 'npmjs.com', 'cloudflare.com', 'mozilla.org', 'apache.org', 'spring.io', 'oracle.com', 'git-scm.com', 'bitbucket.org'}

def load_model(path, name):
    if not os.path.exists(path):
        sys.exit(1)
    with open(path, 'rb') as f:
        return pickle.load(f)

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

def extract_hybrid_features(url, domain):
    feats = {'URLLength': len(url), 'DomainLength': len(domain) if domain else 20.0, 'IsDomainIP': 0.0, 'URLSimilarityIndex': 100.0, 'CharContinuationRate': 1.0, 'TLDLegitimateProb': 0.08, 'URLCharProb': 0.058, 'TLDLength': 3.0, 'NoOfSubDomain': 1.0, 'HasObfuscation': 0.0, 'NoOfObfuscatedChar': 0.0, 'ObfuscationRatio': 0.0, 'NoOfLettersInURL': 14.0, 'LetterRatioInURL': 0.519, 'NoOfDegitsInURL': 0.0, 'DegitRatioInURL': 0.0, 'NoOfEqualsInURL': 0.0, 'NoOfQMarkInURL': 0.0, 'NoOfAmpersandInURL': 0.0, 'NoOfOtherSpecialCharsInURL': 1.0, 'SpacialCharRatioInURL': 0.05, 'IsHTTPS': 1.0 if url.startswith('https') else 0.0, 'LineOfCode': 429.0, 'LargestLineLength': 1090.0, 'HasTitle': 1.0, 'DomainTitleMatchScore': 75.0, 'URLTitleMatchScore': 100.0, 'HasFavicon': 0.0, 'Robots': 0.0, 'IsResponsive': 1.0, 'NoOfURLRedirect': 0.0, 'NoOfSelfRedirect': 0.0, 'HasDescription': 0.0, 'NoOfPopup': 0.0, 'NoOfiFrame': 0.0, 'HasExternalFormSubmit': 0.0, 'HasSocialNet': 0.0, 'HasSubmitButton': 0.0, 'HasHiddenFields': 0.0, 'HasPasswordField': 0.0, 'Bank': 0.0, 'Pay': 0.0, 'Crypto': 0.0, 'HasCopyrightInfo': 0.0, 'NoOfImage': 8.0, 'NoOfCSS': 2.0, 'NoOfJS': 6.0, 'NoOfSelfRef': 12.0, 'NoOfEmptyRef': 0.0, 'NoOfExternalRef': 10.0}
    try:
        ip_pattern = re.compile('^\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}$')
        feats['IsDomainIP'] = 1.0 if ip_pattern.match(domain) else 0.0
        dots = domain.count('.')
        feats['NoOfSubDomain'] = float(max(0, dots - 1))
        letters = sum((1 for c in url if c.isalpha()))
        digits = sum((1 for c in url if c.isdigit()))
        feats['NoOfLettersInURL'] = float(letters)
        feats['LetterRatioInURL'] = letters / len(url) if len(url) > 0 else 0.5
        feats['NoOfDegitsInURL'] = float(digits)
        feats['DegitRatioInURL'] = digits / len(url) if len(url) > 0 else 0.0
        specials = sum((1 for c in url if not c.isalnum() and c not in ('/', '.', ':', '-')))
        feats['NoOfOtherSpecialCharsInURL'] = float(specials)
        feats['SpacialCharRatioInURL'] = specials / len(url) if len(url) > 0 else 0.0
    except Exception:
        pass
    is_online = False
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        with urllib.request.urlopen(req, timeout=3.0) as response:
            html = response.read().decode('utf-8', errors='ignore')
            is_online = True
            lines = html.split('\n')
            feats['LineOfCode'] = float(len(lines))
            feats['LargestLineLength'] = float(max((len(line) for line in lines)) if lines else 0)
            feats['NoOfImage'] = float(len(re.findall('<img\\s+', html, re.I)))
            feats['NoOfJS'] = float(len(re.findall('<script\\s+', html, re.I)))
            feats['NoOfCSS'] = float(len(re.findall('<link\\s+[^>]*rel=["\\\']stylesheet["\\\']|<style\\s+', html, re.I)))
            all_links = re.findall('href=["\\\'](https?://[^"\\\']+|/[^"\\\']+|#[^"\\\']*)["\\\']', html, re.I)
            external = 0
            self_ref = 0
            empty = 0
            domain_escaped = re.escape(domain) if domain else ''
            for link in all_links:
                if link.startswith('#') or link == '':
                    empty += 1
                elif link.startswith('/') or (domain_escaped and re.search(domain_escaped, link)):
                    self_ref += 1
                else:
                    external += 1
            feats['NoOfExternalRef'] = float(external)
            feats['NoOfSelfRef'] = float(self_ref)
            feats['NoOfEmptyRef'] = float(empty)
            feats['HasDescription'] = 1.0 if re.search('<meta\\s+name=["\\\']description["\\\']', html, re.I) else 0.0
            feats['HasFavicon'] = 1.0 if re.search('rel=["\\\'](icon|shortcut icon)["\\\']', html, re.I) else 0.0
            feats['IsResponsive'] = 1.0 if re.search('name=["\\\']viewport["\\\']', html, re.I) else 0.0
            feats['HasSocialNet'] = 1.0 if any((s in html for s in ['facebook.com', 'twitter.com', 'instagram.com'])) else 0.0
            feats['HasSubmitButton'] = 1.0 if re.search('<input\\s+[^>]*type=["\\\']submit["\\\']|<button', html, re.I) else 0.0
            feats['HasHiddenFields'] = 1.0 if re.search('type=["\\\']hidden["\\\']', html, re.I) else 0.0
            feats['HasCopyrightInfo'] = 1.0 if any((c in html.lower() for c in ['copyright', '©', '&copy;'])) else 0.0
    except Exception as e:
    return (feats, is_online)

def main():
    rf_pure = load_model(RF_PURE_PATH, 'Random Forest (Pure URL)')
    xgb_pure = load_model(XGB_PURE_PATH, 'XGBoost (Pure URL)')
    rf_hybrid = load_model(RF_HYBRID_PATH, 'Random Forest (Hybrid)')
    while True:
        url = input('\n🔗 Nhập URL để quét: ').strip()
        if url.lower() in ('quit', 'exit', 'q'):
            break
        if not url:
            continue
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc
            if ':' in domain:
                domain = domain.split(':')[0]
            reg_domain = get_registered_domain(domain)
            is_whitelisted = reg_domain in REPUTABLE_DOMAINS
            rf_features = rf_pure['features']
            rf_feat_vector = rf_ext.extract_features_for_model(url, rf_features)
            if is_whitelisted:
                rf_pred = 0
                rf_prob = 0.0
                rf_label = 'LEGITIMATE ✅ (Whitelist)'
            else:
                rf_pred = rf_pure['model'].predict([rf_feat_vector])[0]
                rf_prob = rf_pure['model'].predict_proba([rf_feat_vector])[0][1]
                rf_label = '🚨 PHISHING' if rf_pred == 1 else '✅ LEGITIMATE'
            xgb_features = xgb_pure['features']
            xgb_feat_vector = xgb_ext.extract_features_for_model(url, xgb_features)
            if is_whitelisted:
                xgb_pred = 0
                xgb_prob = 0.0
                xgb_label = 'LEGITIMATE ✅ (Whitelist)'
            else:
                xgb_pred = xgb_pure['model'].predict([xgb_feat_vector])[0]
                xgb_prob = xgb_pure['model'].predict_proba([xgb_feat_vector])[0][1]
                xgb_label = '🚨 PHISHING' if xgb_pred == 1 else '✅ LEGITIMATE'
            hybrid_features = rf_hybrid['features']
            (hybrid_extracted, is_online) = extract_hybrid_features(url, domain)
            hybrid_vector = [hybrid_extracted[name] for name in hybrid_features]
            if is_whitelisted:
                hybrid_pred = 0
                hybrid_prob = 0.0
                hybrid_label = 'LEGITIMATE ✅ (Whitelist)'
            else:
                hybrid_pred = rf_hybrid['model'].predict([hybrid_vector])[0]
                hybrid_prob = rf_hybrid['model'].predict_proba([hybrid_vector])[0][1]
                hybrid_label = '🚨 PHISHING' if hybrid_pred == 1 else '✅ LEGITIMATE'
            url_trimmed = url[:75] + ('...' if len(url) > 75 else '')
            rf_conf = rf_prob if rf_pred == 1 else 1.0 - rf_prob
            xgb_conf = xgb_prob if xgb_pred == 1 else 1.0 - xgb_prob
            hybrid_conf = hybrid_prob if hybrid_pred == 1 else 1.0 - hybrid_prob
            votes = [rf_pred, xgb_pred, hybrid_pred]
            consensus_pred = 1 if sum(votes) >= 2 else 0
            consensus_label = '🚨 PHISHING (Nguy cơ cao)' if consensus_pred == 1 else '✅ LEGITIMATE (An sau / An toàn)'
            if not is_online:
        except Exception as e:
if __name__ == '__main__':
    main()
