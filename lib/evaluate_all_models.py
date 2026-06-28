import os
import sys
import json
import pickle
import warnings
import time
import re
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data_set')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'model_comparison')
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATASET_FULL_PATH = os.path.join(DATA_DIR, 'dataset_full.csv')
HYBRID_PATH = os.path.join(DATA_DIR, 'hybrid.csv')
DL_TEST_PATH = os.path.join(DATA_DIR, 'dl_dataset', 'processed', 'test_processed.csv')
PHISHING_URL_PATH = os.path.join(DATA_DIR, 'phishing_url.csv')
RF_DSFULL_MODEL = os.path.join(BASE_DIR, 'output', 'rf_vs_dsfull', 'rf_phishing_model.pkl')
RF_HYBRID_MODEL = os.path.join(BASE_DIR, 'output', 'rf_vs_hybrid', 'rf_phishing_model.pkl')
XGB_DSFULL_MODEL = os.path.join(BASE_DIR, 'output', 'xgb_vs_dsfull', 'xgb_phishing_model.pkl')
KNN_HYBRID_MODEL = os.path.join(BASE_DIR, 'output', 'knn_vs_hybrid', 'knn_model.pkl')
DL_MODEL_PTH = os.path.join(BASE_DIR, 'output', 'dl', 'url_phishing_dl_model.pth')
RF_DSFULL_FEATURES = os.path.join(BASE_DIR, 'output', 'rf_vs_dsfull', 'rf_selected_features.json')
XGB_DSFULL_FEATURES = os.path.join(BASE_DIR, 'output', 'xgb_vs_dsfull', 'xgb_selected_features.json')
RF_HYBRID_FEATURES = os.path.join(BASE_DIR, 'output', 'rf_vs_hybrid', 'rf_selected_features.json')

def extract_url_features_for_dsfull(url):
    features = {}
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or ''
        path = parsed.path or ''
        query = parsed.query or ''
        fragment = parsed.fragment or ''
    except:
        parsed = None
        domain = ''
        path = ''
        query = ''
        fragment = ''
    path_parts = path.rsplit('/', 1)
    directory = path_parts[0] if len(path_parts) > 1 else path
    file_part = path_parts[1] if len(path_parts) > 1 else ''
    params = query
    special_chars = {'dot': '.', 'hyphen': '-', 'underline': '_', 'slash': '/', 'questionmark': '?', 'equal': '=', 'at': '@', 'and': '&', 'exclamation': '!', 'space': ' ', 'tilde': '~', 'comma': ',', 'plus': '+', 'asterisk': '*', 'hashtag': '#', 'dollar': '$', 'percent': '%'}
    for (name, char) in special_chars.items():
        features[f'qty_{name}_url'] = url.count(char)
    for (name, char) in special_chars.items():
        features[f'qty_{name}_domain'] = domain.count(char)
    for (name, char) in special_chars.items():
        features[f'qty_{name}_directory'] = directory.count(char)
    for (name, char) in special_chars.items():
        features[f'qty_{name}_file'] = file_part.count(char)
    for (name, char) in special_chars.items():
        features[f'qty_{name}_params'] = params.count(char) if params else -1
    features['length_url'] = len(url)
    features['domain_length'] = len(domain)
    features['directory_length'] = len(directory)
    features['file_length'] = len(file_part)
    features['params_length'] = len(params) if params else -1
    features['qty_vowels_domain'] = sum((1 for c in domain.lower() if c in 'aeiou'))
    features['domain_in_ip'] = 1 if re.match('^\\d+\\.\\d+\\.\\d+\\.\\d+', domain) else 0
    features['qty_tld_url'] = 1 if '.' in domain else 0
    features['url_shortened'] = 1 if any((s in domain.lower() for s in ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'buff.ly'])) else 0
    features['qty_params'] = params.count('&') + 1 if params else 0
    features['email_in_url'] = 1 if re.search('[\\w.+-]+@[\\w-]+\\.[\\w.]+', url) else 0
    for col in ['time_response', 'domain_spf', 'asn_ip', 'time_domain_activation', 'time_domain_expiration', 'qty_ip_resolved', 'qty_nameservers', 'qty_mx_servers', 'ttl_hostname', 'tls_ssl_certificate', 'qty_redirects', 'url_google_index', 'domain_google_index', 'server_client_domain', 'tld_present_params']:
        features[col] = -1
    return features
MAX_LEN = 200
CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~:/?#[]@!$&'()*+,;= "
CHAR_TO_IDX = {char: idx + 2 for (idx, char) in enumerate(CHAR_VOCAB)}
CHAR_TO_IDX['<pad>'] = 0
CHAR_TO_IDX['<unk>'] = 1
VOCAB_SIZE = len(CHAR_TO_IDX)

def tokenize_url(url, max_len=MAX_LEN):
    if not isinstance(url, str):
        url = ''
    tokenized = []
    for char in url[:max_len]:
        tokenized.append(CHAR_TO_IDX.get(char, 1))
    if len(tokenized) < max_len:
        tokenized += [0] * (max_len - len(tokenized))
    return np.array(tokenized, dtype=np.int64)

def load_all_models():
    models = {}
    with open(RF_DSFULL_MODEL, 'rb') as f:
        rf_data = pickle.load(f)
    rf_model = rf_data['model']
    rf_model.n_jobs = 1
    models['RF_Pure_URL'] = {'model': rf_model, 'features': rf_data['features'], 'type': 'sklearn'}
    with open(XGB_DSFULL_MODEL, 'rb') as f:
        xgb_data = pickle.load(f)
    xgb_model = xgb_data['model']
    xgb_model.n_jobs = 1
    models['XGBoost_Pure_URL'] = {'model': xgb_model, 'features': xgb_data['features'], 'type': 'sklearn'}
    with open(RF_HYBRID_MODEL, 'rb') as f:
        rf_hybrid_data = pickle.load(f)
    rf_h_model = rf_hybrid_data['model']
    rf_h_model.n_jobs = 1
    models['RF_Hybrid'] = {'model': rf_h_model, 'features': rf_hybrid_data['features'], 'type': 'sklearn_hybrid'}
    with open(KNN_HYBRID_MODEL, 'rb') as f:
        knn_data = pickle.load(f)
    knn_model = knn_data['knn']
    knn_model.n_jobs = 1
    models['KNN_Hybrid'] = {'knn': knn_model, 'scaler': knn_data['scaler'], 'features': knn_data['features'], 'type': 'knn_unsupervised'}
    try:
        import torch
        import torch.nn as nn
        torch.set_num_threads(1)

        class PhishingCNNBiLSTM(nn.Module):

            def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=64, cnn_filters=128, kernel_size=5, lstm_hidden=64, dropout=0.5):
                super(PhishingCNNBiLSTM, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=cnn_filters, kernel_size=kernel_size, padding=kernel_size // 2)
                self.relu = nn.ReLU()
                self.max_pool = nn.MaxPool1d(kernel_size=2)
                self.bilstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden, num_layers=1, bidirectional=True, batch_first=True)
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(lstm_hidden * 2, 1)

            def forward(self, x):
                embedded = self.embedding(x).transpose(1, 2)
                conv_out = self.relu(self.conv1d(embedded))
                pooled = self.max_pool(conv_out).transpose(1, 2)
                (lstm_out, _) = self.bilstm(pooled)
                repr_vec = lstm_out[:, -1, :]
                out = self.dropout(repr_vec)
                logits = self.fc(out)
                return logits.squeeze(1)
        device = torch.device('cpu')
        dl_model = PhishingCNNBiLSTM()
        dl_model.load_state_dict(torch.load(DL_MODEL_PTH, map_location=device))
        dl_model.eval()
        models['DL_CNN_BiLSTM'] = {'model': dl_model, 'type': 'pytorch'}
    except Exception as e:
    return models

def evaluate_on_holdout(models):
    results = {}
    df_full = pd.read_csv(DATASET_FULL_PATH)
    y_full = df_full['phishing']
    (_, X_test_idx, _, y_test) = train_test_split(df_full.index, y_full, test_size=0.2, random_state=42, stratify=y_full)
    df_test_full = df_full.loc[X_test_idx]
    rf_model = models['RF_Pure_URL']
    X_test_rf = df_test_full[rf_model['features']]
    y_pred_rf = rf_model['model'].predict(X_test_rf)
    y_proba_rf = rf_model['model'].predict_proba(X_test_rf)[:, 1]
    results['RF_Pure_URL'] = {'y_true': y_test.values, 'y_pred': y_pred_rf, 'y_proba': y_proba_rf, 'dataset': 'dataset_full (hold-out 20%)'}
    xgb_model = models['XGBoost_Pure_URL']
    X_test_xgb = df_test_full[xgb_model['features']]
    y_pred_xgb = xgb_model['model'].predict(X_test_xgb)
    y_proba_xgb = xgb_model['model'].predict_proba(X_test_xgb)[:, 1]
    results['XGBoost_Pure_URL'] = {'y_true': y_test.values, 'y_pred': y_pred_xgb, 'y_proba': y_proba_xgb, 'dataset': 'dataset_full (hold-out 20%)'}
    df_hybrid = pd.read_csv(HYBRID_PATH)
    y_hybrid = df_hybrid['label']
    (_, X_test_h_idx, _, y_test_h) = train_test_split(df_hybrid.index, y_hybrid, test_size=0.2, random_state=42)
    df_test_hybrid = df_hybrid.loc[X_test_h_idx]
    rf_hybrid = models['RF_Hybrid']
    X_test_rf_h = df_test_hybrid[rf_hybrid['features']]
    y_pred_rf_h = rf_hybrid['model'].predict(X_test_rf_h)
    y_proba_rf_h = rf_hybrid['model'].predict_proba(X_test_rf_h)[:, 1]
    results['RF_Hybrid'] = {'y_true': y_test_h.values, 'y_pred': y_pred_rf_h, 'y_proba': y_proba_rf_h, 'dataset': 'hybrid (hold-out 20%)'}
    if 'DL_CNN_BiLSTM' in models:
        import torch
        df_dl_test = pd.read_csv(DL_TEST_PATH)
        df_dl_test = df_dl_test.dropna(subset=['url', 'result'])
        urls_dl = df_dl_test['url'].values
        y_true_dl = df_dl_test['result'].values.astype(np.float32)
        dl_model = models['DL_CNN_BiLSTM']['model']
        batch_size = 256
        all_preds = []
        all_probas = []
        for i in range(0, len(urls_dl), batch_size):
            batch_urls = urls_dl[i:i + batch_size]
            tokenized = np.array([tokenize_url(url) for url in batch_urls])
            with torch.no_grad():
                inputs = torch.tensor(tokenized)
                logits = dl_model(inputs)
                probas = torch.sigmoid(logits).numpy()
                preds = (probas >= 0.5).astype(np.float32)
                all_preds.extend(preds)
                all_probas.extend(probas)
        results['DL_CNN_BiLSTM'] = {'y_true': y_true_dl, 'y_pred': np.array(all_preds), 'y_proba': np.array(all_probas), 'dataset': 'dl_dataset (test_processed)'}
    return results

def extract_hybrid_features_fallback(url):
    feats = {'URLLength': len(url), 'DomainLength': 20.0, 'IsDomainIP': 0.0, 'URLSimilarityIndex': 100.0, 'CharContinuationRate': 1.0, 'TLDLegitimateProb': 0.08, 'URLCharProb': 0.058, 'TLDLength': 3.0, 'NoOfSubDomain': 1.0, 'HasObfuscation': 0.0, 'NoOfObfuscatedChar': 0.0, 'ObfuscationRatio': 0.0, 'NoOfLettersInURL': 14.0, 'LetterRatioInURL': 0.519, 'NoOfDegitsInURL': 0.0, 'DegitRatioInURL': 0.0, 'NoOfEqualsInURL': 0.0, 'NoOfQMarkInURL': 0.0, 'NoOfAmpersandInURL': 0.0, 'NoOfOtherSpecialCharsInURL': 1.0, 'SpacialCharRatioInURL': 0.05, 'IsHTTPS': 1.0 if url.startswith('https') else 0.0, 'LineOfCode': 429.0, 'LargestLineLength': 1090.0, 'HasTitle': 1.0, 'DomainTitleMatchScore': 75.0, 'URLTitleMatchScore': 100.0, 'HasFavicon': 0.0, 'Robots': 0.0, 'IsResponsive': 1.0, 'NoOfURLRedirect': 0.0, 'NoOfSelfRedirect': 0.0, 'HasDescription': 0.0, 'NoOfPopup': 0.0, 'NoOfiFrame': 0.0, 'HasExternalFormSubmit': 0.0, 'HasSocialNet': 0.0, 'HasSubmitButton': 0.0, 'HasHiddenFields': 0.0, 'HasPasswordField': 0.0, 'Bank': 0.0, 'Pay': 0.0, 'Crypto': 0.0, 'HasCopyrightInfo': 0.0, 'NoOfImage': 8.0, 'NoOfCSS': 2.0, 'NoOfJS': 6.0, 'NoOfSelfRef': 12.0, 'NoOfEmptyRef': 0.0, 'NoOfExternalRef': 10.0}
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or ''
        if ':' in domain:
            domain = domain.split(':')[0]
        feats['DomainLength'] = len(domain)
        feats['IsDomainIP'] = 1.0 if re.match('^\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}$', domain) else 0.0
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
    except:
        pass
    return feats

def evaluate_on_external(models):
    results_ext = {}
    if not os.path.exists(PHISHING_URL_PATH):
        return results_ext
    df_phish = pd.read_csv(PHISHING_URL_PATH)
    if 'verified' in df_phish.columns:
        df_phish = df_phish[df_phish['verified'] == 'yes']
    if 'online' in df_phish.columns:
        df_phish = df_phish[df_phish['online'] == 'yes']
    if len(df_phish) > 5000:
        df_phish = df_phish.sample(n=5000, random_state=42)
    phishing_urls = df_phish['url'].dropna().values
    n_samples = len(phishing_urls)
    y_true = np.ones(n_samples, dtype=int)
    legit_urls = ['https://www.google.com', 'https://www.facebook.com', 'https://www.youtube.com', 'https://www.amazon.com', 'https://www.wikipedia.org', 'https://www.twitter.com', 'https://www.github.com', 'https://www.stackoverflow.com', 'https://www.microsoft.com', 'https://www.apple.com', 'https://www.netflix.com', 'https://www.linkedin.com', 'https://www.reddit.com', 'https://www.yahoo.com', 'https://www.bing.com', 'https://www.ebay.com', 'https://www.instagram.com', 'https://www.twitch.tv', 'https://www.spotify.com', 'https://www.medium.com', 'https://docs.python.org', 'https://www.bbc.com/news', 'https://www.cnn.com', 'https://www.nytimes.com', 'https://mail.google.com', 'https://drive.google.com', 'https://www.dropbox.com', 'https://www.paypal.com', 'https://www.shopify.com', 'https://www.wordpress.org']
    all_urls = list(phishing_urls) + legit_urls
    y_true_ext = np.concatenate([y_true, np.zeros(len(legit_urls), dtype=int)])
    features_list = []
    for (i, url) in enumerate(all_urls):
        if i % 500 == 0:
        features_list.append(extract_url_features_for_dsfull(url))
    df_features = pd.DataFrame(features_list)
    rf_model = models['RF_Pure_URL']
    rf_feats = rf_model['features']
    for f in rf_feats:
        if f not in df_features.columns:
            df_features[f] = -1
    X_ext_rf = df_features[rf_feats].values
    y_pred_rf = rf_model['model'].predict(X_ext_rf)
    y_proba_rf = rf_model['model'].predict_proba(X_ext_rf)[:, 1]
    results_ext['RF_Pure_URL'] = {'y_true': y_true_ext, 'y_pred': y_pred_rf, 'y_proba': y_proba_rf, 'dataset': 'PhishTank External'}
    xgb_model = models['XGBoost_Pure_URL']
    xgb_feats = xgb_model['features']
    for f in xgb_feats:
        if f not in df_features.columns:
            df_features[f] = -1
    X_ext_xgb = df_features[xgb_feats].values
    y_pred_xgb = xgb_model['model'].predict(X_ext_xgb)
    y_proba_xgb = xgb_model['model'].predict_proba(X_ext_xgb)[:, 1]
    results_ext['XGBoost_Pure_URL'] = {'y_true': y_true_ext, 'y_pred': y_pred_xgb, 'y_proba': y_proba_xgb, 'dataset': 'PhishTank External'}
    if 'RF_Hybrid' in models:
        rf_h_model = models['RF_Hybrid']
        rf_h_feats = rf_h_model['features']
        features_h_list = []
        for url in all_urls:
            features_h_list.append(extract_hybrid_features_fallback(url))
        df_h_features = pd.DataFrame(features_h_list)
        for f in rf_h_feats:
            if f not in df_h_features.columns:
                df_h_features[f] = 0.0
        X_ext_rf_h = df_h_features[rf_h_feats].values
        y_pred_rf_h = rf_h_model['model'].predict(X_ext_rf_h)
        y_proba_rf_h = rf_h_model['model'].predict_proba(X_ext_rf_h)[:, 1]
        results_ext['RF_Hybrid'] = {'y_true': y_true_ext, 'y_pred': y_pred_rf_h, 'y_proba': y_proba_rf_h, 'dataset': 'PhishTank External'}
    if 'DL_CNN_BiLSTM' in models:
        import torch
        dl_model = models['DL_CNN_BiLSTM']['model']
        batch_size = 256
        all_preds_dl = []
        all_probas_dl = []
        for i in range(0, len(all_urls), batch_size):
            batch_urls = all_urls[i:i + batch_size]
            tokenized = np.array([tokenize_url(url) for url in batch_urls])
            with torch.no_grad():
                inputs = torch.tensor(tokenized)
                logits = dl_model(inputs)
                probas = torch.sigmoid(logits).numpy()
                preds = (probas >= 0.5).astype(np.float32)
                all_preds_dl.extend(preds)
                all_probas_dl.extend(probas)
        results_ext['DL_CNN_BiLSTM'] = {'y_true': y_true_ext, 'y_pred': np.array(all_preds_dl), 'y_proba': np.array(all_probas_dl), 'dataset': 'PhishTank External'}
    return results_ext

def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred, zero_division=0), 'recall': recall_score(y_true, y_pred, zero_division=0), 'f1_score': f1_score(y_true, y_pred, zero_division=0)}
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        (tn, fp, fn, tp) = cm.ravel()
        metrics['true_positive'] = int(tp)
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        metrics['total_errors'] = int(fp + fn)
        metrics['error_rate'] = (fp + fn) / len(y_true)
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['roc_auc'] = None
    return metrics

def print_comparison_table(all_results, title=''):
    header = f"{'Mô hình':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'AUC-ROC':>10} {'Errors':>10} {'Error%':>10}"
    rows = []
    for (model_name, res) in all_results.items():
        metrics = compute_metrics(res['y_true'], res['y_pred'], res.get('y_proba'))
        roc_str = f"{metrics['roc_auc']:.4f}" if metrics.get('roc_auc') is not None else 'N/A'
        row = f"{model_name:<20} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1_score']:>10.4f} {roc_str:>10} {metrics.get('total_errors', 'N/A'):>10} {metrics.get('error_rate', 0) * 100:>9.2f}%"
        rows.append((model_name, metrics))
    if rows:
        best_model = max(rows, key=lambda x: x[1]['f1_score'])
        worst_model = min(rows, key=lambda x: x[1]['f1_score'])
        most_errors = max(rows, key=lambda x: x[1].get('total_errors', 0))
    return rows

def plot_comparison_charts(all_results, suffix=''):
    sns.set_theme(style='whitegrid')
    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    error_rates = []
    for (model_name, res) in all_results.items():
        metrics = compute_metrics(res['y_true'], res['y_pred'], res.get('y_proba'))
        model_names.append(model_name.replace('_', '\n'))
        accuracies.append(metrics['accuracy'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1_scores.append(metrics['f1_score'])
        error_rates.append(metrics.get('error_rate', 0) * 100)
    (fig, axes) = plt.subplots(1, 2, figsize=(18, 7))
    x = np.arange(len(model_names))
    width = 0.2
    ax1 = axes[0]
    bars1 = ax1.bar(x - 1.5 * width, accuracies, width, label='Accuracy', color='#3498db', edgecolor='white')
    bars2 = ax1.bar(x - 0.5 * width, precisions, width, label='Precision', color='#2ecc71', edgecolor='white')
    bars3 = ax1.bar(x + 0.5 * width, recalls, width, label='Recall', color='#e67e22', edgecolor='white')
    bars4 = ax1.bar(x + 1.5 * width, f1_scores, width, label='F1-Score', color='#e74c3c', edgecolor='white')
    ax1.set_xlabel('Mô hình', fontsize=12)
    ax1.set_ylabel('Giá trị', fontsize=12)
    ax1.set_title(f'So sánh Metrics các mô hình {suffix}', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=9)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=7)
    ax2 = axes[1]
    colors = ['#e74c3c' if er == max(error_rates) else '#3498db' for er in error_rates]
    bars = ax2.bar(model_names, error_rates, color=colors, edgecolor='white', width=0.6)
    ax2.set_xlabel('Mô hình', fontsize=12)
    ax2.set_ylabel('Tỷ lệ lỗi (%)', fontsize=12)
    ax2.set_title(f'Tỷ lệ dự đoán sai {suffix}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for (bar, er) in zip(bars, error_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.2, f'{er:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    plt.tight_layout()
    suffix_clean = suffix.replace(' ', '_').replace('(', '').replace(')', '').lower().strip('_')
    filename = f'model_comparison_{suffix_clean}.png' if suffix_clean else 'model_comparison.png'
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    n_models = len(all_results)
    (fig, axes) = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    for (idx, (model_name, res)) in enumerate(all_results.items()):
        cm = confusion_matrix(res['y_true'], res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], xticklabels=['Legit', 'Phishing'], yticklabels=['Legit', 'Phishing'], annot_kws={'size': 12, 'weight': 'bold'})
        axes[idx].set_title(f'{model_name}', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Dự đoán', fontsize=10)
        axes[idx].set_ylabel('Thực tế', fontsize=10)
    plt.suptitle(f'Confusion Matrix - Tất cả mô hình {suffix}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    suffix_clean2 = suffix.replace(' ', '_').replace('(', '').replace(')', '').lower().strip('_')
    filename_cm = f'confusion_matrices_{suffix_clean2}.png' if suffix_clean2 else 'confusion_matrices.png'
    save_path_cm = os.path.join(OUTPUT_DIR, filename_cm)
    plt.savefig(save_path_cm, dpi=150, bbox_inches='tight')
    plt.close()

def analyze_errors(all_results, suffix=''):
    error_details = {}
    for (model_name, res) in all_results.items():
        y_true = res['y_true']
        y_pred = res['y_pred']
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            (tn, fp, fn, tp) = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        total = len(y_true)
        total_errors = fp + fn
        if fn > 0:
        if fp > 0:
        error_details[model_name] = {'total': total, 'errors': total_errors, 'false_positive': fp, 'false_negative': fn, 'error_rate': total_errors / total}
    if error_details:
        sorted_models = sorted(error_details.items(), key=lambda x: x[1]['error_rate'])
        for (rank, (name, details)) in enumerate(sorted_models, 1):
            emoji = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else '  '
    return error_details

def main():
    start_time = time.time()
    models = load_all_models()
    holdout_results = evaluate_on_holdout(models)
    print_comparison_table(holdout_results, 'BẢNG SO SÁNH - TẬP TEST HOLD-OUT')
    analyze_errors(holdout_results, '(Hold-out)')
    external_results = evaluate_on_external(models)
    if external_results:
        print_comparison_table(external_results, 'BẢNG SO SÁNH - TẬP DỮ LIỆU NGOẠI (PhishTank)')
        analyze_errors(external_results, '(External)')
    all_summary = {}
    for (model_name, res) in holdout_results.items():
        metrics = compute_metrics(res['y_true'], res['y_pred'], res.get('y_proba'))
        all_summary[model_name] = {'holdout': metrics, 'dataset': res['dataset']}
    for (model_name, res) in external_results.items():
        metrics = compute_metrics(res['y_true'], res['y_pred'], res.get('y_proba'))
        if model_name in all_summary:
            all_summary[model_name]['external'] = metrics
        else:
            all_summary[model_name] = {'external': metrics}
    summary_path = os.path.join(OUTPUT_DIR, 'evaluation_summary.json')

    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    summary_json = {}
    for (model_name, data) in all_summary.items():
        summary_json[model_name] = {}
        for (key, val) in data.items():
            if isinstance(val, dict):
                summary_json[model_name][key] = {k: convert_numpy(v) for (k, v) in val.items()}
            else:
                summary_json[model_name][key] = convert_numpy(val)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False, default=str)
    elapsed = time.time() - start_time
if __name__ == '__main__':
    main()
