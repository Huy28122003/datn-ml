import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score
from xgboost import XGBClassifier
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'data_set', 'dataset_full.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'xgb_vs_dsfull')
os.makedirs(OUTPUT_DIR, exist_ok=True)
SELECTED_FEATURES_PATH = os.path.join(OUTPUT_DIR, 'xgb_selected_features.json')
BEST_HYPERPARAMS_PATH = os.path.join(OUTPUT_DIR, 'xgb_best_hyperparameters.json')
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'xgb_phishing_model.pkl')
TRAINING_SUMMARY_PATH = os.path.join(OUTPUT_DIR, 'xgb_model_training_summary.json')

def main():
    if not os.path.exists(SELECTED_FEATURES_PATH) or not os.path.exists(BEST_HYPERPARAMS_PATH):
        sys.exit(1)
    with open(SELECTED_FEATURES_PATH, 'r') as f:
        selected_features = json.load(f)
    with open(BEST_HYPERPARAMS_PATH, 'r') as f:
        hyperparams = json.load(f)
    if not os.path.exists(DATASET_PATH):
        sys.exit(1)
    df = pd.read_csv(DATASET_PATH)
    X = df[selected_features]
    y = df['phishing']
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(**hyperparams)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_txt = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'], annot_kws={'size': 14, 'weight': 'bold'})
    plt.title('XGBoost - Ma trận nhầm lẫn (Confusion Matrix)', fontsize=14, pad=15)
    plt.xlabel('Nhãn dự đoán bởi Model', fontsize=12)
    plt.ylabel('Nhãn thực tế (Ground Truth)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'xgb_confusion_matrix.png'), dpi=150)
    plt.close()
    (fpr, tpr, _) = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('XGBoost - Đường cong ROC', fontsize=14, pad=15)
    plt.xlabel('Tỷ lệ cảnh báo sai (False Positive Rate)', fontsize=12)
    plt.ylabel('Tỷ lệ phát hiện đúng (True Positive Rate)', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'xgb_roc_curve.png'), dpi=150)
    plt.close()
    (precision, recall, _) = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2.5, label=f'Precision-Recall (AP = {avg_precision:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('XGBoost - Đường cong Precision-Recall', fontsize=14, pad=15)
    plt.xlabel('Tỷ lệ phát hiện đúng (Recall)', fontsize=12)
    plt.ylabel('Độ chính xác lớp cảnh báo (Precision)', fontsize=12)
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'xgb_precision_recall_curve.png'), dpi=150)
    plt.close()
    plt.figure(figsize=(8, 5))
    folds = [f'Fold {i + 1}' for i in range(5)]
    sns.barplot(x=folds, y=cv_scores, palette='coolwarm')
    plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', lw=1.5, label=f'Trung bình = {cv_scores.mean():.4%}')
    plt.ylim([min(cv_scores) - 0.01, 1.0])
    plt.title('XGBoost - Tính ổn định qua 5-Fold CV', fontsize=14, pad=15)
    plt.ylabel('Độ chính xác (Accuracy)', fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'xgb_cross_validation_scores.png'), dpi=150)
    plt.close()
    (train_sizes, train_scores, val_scores) = learning_curve(model, X, y, cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy', random_state=42)
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    plt.figure(figsize=(9, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Độ chính xác tập Train', lw=2)
    plt.plot(train_sizes, val_scores_mean, 's-', color='g', label='Độ chính xác tập Validation', lw=2)
    plt.title('XGBoost - Đường cong học tập (Learning Curve)', fontsize=14, pad=15)
    plt.xlabel('Mẫu huấn luyện', fontsize=12)
    plt.ylabel('Độ chính xác (Accuracy)', fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'xgb_learning_curve.png'), dpi=150)
    plt.close()
    plt.figure(figsize=(8, 4))
    sns.heatmap(pd.DataFrame(report_dict).iloc[:-1, :2].T, annot=True, cmap='YlGnBu', cbar=False, annot_kws={'size': 13, 'weight': 'bold'})
    plt.title('XGBoost - Classification Report', fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'xgb_classification_report.png'), dpi=150)
    plt.close()
    model_data = {'model': model, 'features': selected_features, 'params': hyperparams}
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    summary_data = {'accuracy': float(acc), 'precision_phishing': float(report_dict['1']['precision']), 'recall_phishing': float(report_dict['1']['recall']), 'f1_score_phishing': float(report_dict['1']['f1-score']), 'cross_validation_mean': float(cv_scores.mean()), 'cross_validation_std': float(cv_scores.std()), 'auc_roc': float(roc_auc), 'features_count': len(selected_features), 'hyperparameters': hyperparams}
    with open(TRAINING_SUMMARY_PATH, 'w') as f:
        json.dump(summary_data, f, indent=4)
if __name__ == '__main__':
    main()
