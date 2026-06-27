"""
rf_model_train_final.py
========================
Bước 3: Huấn luyện mô hình Random Forest cuối cùng trên tập dữ liệu Hybrid.
  - Đọc danh sách các đặc trưng chọn lọc và siêu tham số tối ưu.
  - Huấn luyện mô hình trên toàn bộ 235,794 mẫu của tập dữ liệu Hybrid.
  - Đánh giá trên tập kiểm thử Test Set (20% - 47,159 mẫu) và vẽ 6 biểu đồ kỹ thuật.
  - Lưu file mô hình binary: output/rf_vs_hybrid/rf_phishing_model.pkl.
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, 
    precision_recall_curve, average_precision_score, accuracy_score
)

# Xác định đường dẫn gốc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Cấu hình đường dẫn
DATASET_PATH = os.path.join(BASE_DIR, 'data_set', 'hybrid.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'rf_vs_hybrid')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SELECTED_FEATURES_PATH = os.path.join(OUTPUT_DIR, 'rf_selected_features.json')
BEST_HYPERPARAMS_PATH = os.path.join(OUTPUT_DIR, 'rf_best_hyperparameters.json')
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'rf_phishing_model.pkl')
TRAINING_SUMMARY_PATH = os.path.join(OUTPUT_DIR, 'rf_model_training_summary.json')


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  BƯỚC 3 (HYBRID): HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH CUỐI     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if not os.path.exists(SELECTED_FEATURES_PATH) or not os.path.exists(BEST_HYPERPARAMS_PATH):
        print("❌ LỖI: Thiếu tệp cấu hình đặc trưng hoặc tham số tối ưu!")
        sys.exit(1)

    with open(SELECTED_FEATURES_PATH, 'r') as f:
        selected_features = json.load(f)
    with open(BEST_HYPERPARAMS_PATH, 'r') as f:
        hyperparams = json.load(f)

    print(f"\n📖 Đã load {len(selected_features)} đặc trưng chọn lọc và tham số Random Forest.")
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ LỖI: Không tìm thấy tập dữ liệu tại {DATASET_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)

    # Loại bỏ FILENAME và cột text
    if 'FILENAME' in df.columns:
        df = df.drop(columns=['FILENAME'])
    text_cols = ['URL', 'Domain', 'TLD', 'Title']
    available_text_cols = [c for c in text_cols if c in df.columns]
    if available_text_cols:
        df = df.drop(columns=available_text_cols)

    X = df[selected_features]
    y = df['label']

    # Chia tập dữ liệu thành Train (80%) và Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  - Mẫu tập Train: {X_train.shape[0]:,}")
    print(f"  - Mẫu tập Test:  {X_test.shape[0]:,}")

    # 2. Huấn luyện mô hình Random Forest chính thức
    print("\n🌲 Đang huấn luyện mô hình Random Forest cuối cùng trên tập Hybrid...")
    model = RandomForestClassifier(**hyperparams)
    model.fit(X_train, y_train)
    print("  ✓ Huấn luyện hoàn tất!")

    # 3. Đánh giá chất lượng mô hình trên tập Test
    print("\n📊 Đang đánh giá hiệu năng trên tập Test...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Tính toán các chỉ số
    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_txt = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"\n📈 CHỈ SỐ CHẤT LƯỢNG MÔ HÌNH RANDOM FOREST (HYBRID):")
    print(f"  - Accuracy: {acc:.4%}")
    print(f"  - Precision (Phishing): {report_dict['1']['precision']:.4%}")
    print(f"  - Recall (Phishing):    {report_dict['1']['recall']:.4%}")
    print(f"  - F1-Score (Phishing):  {report_dict['1']['f1-score']:.4%}")

    # 4. Kiểm định chéo 5-Fold Cross Validation
    # Sử dụng 60,000 dòng để 5-Fold chạy cực nhanh nhưng vẫn bảo đảm tính đồng nhất
    print("\n🔄 Đang thực hiện 5-Fold Cross Validation trên mẫu đại diện...")
    df_cv = df.sample(n=60000, random_state=42)
    X_cv = df_cv[selected_features]
    y_cv = df_cv['label']
    cv_scores = cross_val_score(model, X_cv, y_cv, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"  ✓ Cross-Validation Scores: {cv_scores}")
    print(f"  ✓ Độ chính xác trung bình: {cv_scores.mean():.4%} (± {cv_scores.std():.4%})")

    # 5. Vẽ và xuất bản các đồ thị đánh giá chuyên nghiệp
    print("\n🎨 Đang tạo các biểu đồ kỹ thuật...")
    sns.set_theme(style="whitegrid")

    # Đồ thị 1: Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        cbar=False,
        xticklabels=['Legitimate', 'Phishing'],
        yticklabels=['Legitimate', 'Phishing'],
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    plt.title('Hybrid RF - Ma trận nhầm lẫn (Confusion Matrix)', fontsize=14, pad=15)
    plt.xlabel('Nhãn dự đoán bởi Model', fontsize=12)
    plt.ylabel('Nhãn thực tế (Ground Truth)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rf_confusion_matrix.png'), dpi=150)
    plt.close()

    # Đồ thị 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Hybrid RF - Đường cong ROC', fontsize=14, pad=15)
    plt.xlabel('Tỷ lệ cảnh báo sai (False Positive Rate)', fontsize=12)
    plt.ylabel('Tỷ lệ phát hiện đúng (True Positive Rate)', fontsize=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rf_roc_curve.png'), dpi=150)
    plt.close()

    # Đồ thị 3: Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2.5, label=f'Precision-Recall (AP = {avg_precision:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Hybrid RF - Đường cong Precision-Recall', fontsize=14, pad=15)
    plt.xlabel('Tỷ lệ phát hiện đúng (Recall)', fontsize=12)
    plt.ylabel('Độ chính xác lớp cảnh báo (Precision)', fontsize=12)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rf_precision_recall_curve.png'), dpi=150)
    plt.close()

    # Đồ thị 4: Cross-Validation Scores
    plt.figure(figsize=(8, 5))
    folds = [f'Fold {i+1}' for i in range(5)]
    sns.barplot(x=folds, y=cv_scores, palette='coolwarm')
    plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', lw=1.5, label=f'Trung bình = {cv_scores.mean():.4%}')
    plt.ylim([min(cv_scores) - 0.01, 1.0])
    plt.title('Hybrid RF - Tính ổn định qua 5-Fold CV', fontsize=14, pad=15)
    plt.ylabel('Độ chính xác (Accuracy)', fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rf_cross_validation_scores.png'), dpi=150)
    plt.close()

    # Đồ thị 5: Learning Curve (Dùng mẫu 50,000 để vẽ cực nhanh)
    print("  → Đang tính toán Learning Curve...")
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_cv, y_cv, cv=3, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), 
        scoring='accuracy', random_state=42
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    
    plt.figure(figsize=(9, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Độ chính xác tập Train", lw=2)
    plt.plot(train_sizes, val_scores_mean, 's-', color="g", label="Độ chính xác tập Validation", lw=2)
    plt.title("Hybrid RF - Đường cong học tập", fontsize=14, pad=15)
    plt.xlabel("Mẫu huấn luyện", fontsize=12)
    plt.ylabel("Độ chính xác", fontsize=12)
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rf_learning_curve.png'), dpi=150)
    plt.close()

    # Đồ thị 6: Classification Report
    plt.figure(figsize=(8, 4))
    sns.heatmap(
        pd.DataFrame(report_dict).iloc[:-1, :2].T, 
        annot=True, 
        fmt=".4f",
        cmap="YlGnBu", 
        cbar=False,
        annot_kws={'size': 13, 'weight': 'bold'}
    )
    plt.title('Hybrid RF - Báo cáo phân loại (Classification Report)', fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rf_classification_report.png'), dpi=150)
    plt.close()

    # 6. Lưu file mô hình binary hoàn chỉnh
    print("\n📦 Đang lưu trữ tệp mô hình...")
    model_data = {
        'model': model,
        'features': selected_features,
        'params': hyperparams
    }
    
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"  ✓ Đã lưu file mô hình tại: {MODEL_SAVE_PATH}")

    # 7. Lưu báo cáo tóm tắt dưới dạng JSON
    summary_data = {
        'accuracy': float(acc),
        'precision_phishing': float(report_dict['1']['precision']),
        'recall_phishing': float(report_dict['1']['recall']),
        'f1_score_phishing': float(report_dict['1']['f1-score']),
        'cross_validation_mean': float(cv_scores.mean()),
        'cross_validation_std': float(cv_scores.std()),
        'auc_roc': float(roc_auc),
        'features_count': len(selected_features),
        'hyperparameters': hyperparams
    }
    with open(TRAINING_SUMMARY_PATH, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"  ✓ Đã lưu tóm tắt báo cáo huấn luyện tại: {TRAINING_SUMMARY_PATH}")

    print("\n🎉 HUẦN LUYỆN THÀNH CÔNG MÔ HÌNH RANDOM FOREST (HYBRID)!")


if __name__ == '__main__':
    main()
