import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, 
    precision_recall_curve, average_precision_score, accuracy_score, f1_score, recall_score
)
from xgboost import XGBClassifier

# Add current project root to system path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, BASE_DIR)

def load_data(processed_dir):
    """Loads train, validation, and test sets."""
    print("Loading optimized datasets...")
    train_path = os.path.join(processed_dir, "train_processed.csv")
    val_path = os.path.join(processed_dir, "val_processed.csv")
    test_path = os.path.join(processed_dir, "test_processed.csv")
    
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Processed datasets not found! Run pipeline.py first.")
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Loaded train ({len(train_df)}), val ({len(val_df)}), test ({len(test_df)}) samples.")
    return train_df, val_df, test_df

def extract_features(train_df, val_df, test_df):
    """
    Extracts char-level n-grams from URLs using TF-IDF.
    This serves as a solid baseline resembling character-level deep learning models.
    """
    print("\nExtracting character-level n-gram features (TF-IDF)...")
    # Character n-grams of size 3 to 5
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=5000)
    
    X_train = vectorizer.fit_transform(train_df['url'])
    X_val = vectorizer.transform(val_df['url'])
    X_test = vectorizer.transform(test_df['url'])
    
    y_train = train_df['result'].values
    y_val = val_df['result'].values
    y_test = test_df['result'].values
    
    print(f"Feature matrix shape: {X_train.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer

def train_and_evaluate(processed_dir, output_dir):
    """Trains a baseline model (XGBoost) and evaluates it on the test set."""
    train_df, val_df, test_df = load_data(processed_dir)
    
    # Downsample datasets if they are too large for a fast demonstration run
    # (Optional: remove this if you want to run on the complete dataset with full computing power)
    MAX_SAMPLES = 50000
    if len(train_df) > MAX_SAMPLES:
        print(f"Downsampling training set to {MAX_SAMPLES} samples for quick demonstration.")
        train_df = train_df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
    if len(val_df) > 10000:
        val_df = val_df.sample(n=10000, random_state=42).reset_index(drop=True)
    if len(test_df) > 10000:
        test_df = test_df.sample(n=10000, random_state=42).reset_index(drop=True)
        
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = extract_features(train_df, val_df, test_df)
    
    print("\n⚡ Training XGBoost baseline classifier...")
    # Use fast parameters for quick demo
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.15,
        verbosity=1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    print("✓ Training completed!")
    
    print("\nEvaluating model on the Test Set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate performance metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Benign', 'Malicious'])
    
    print("\n================ EVALUATION METRICS ================")
    print(f"Accuracy:  {acc:.4%}")
    print(f"Recall:    {recall:.4%}")
    print(f"F1-Score:  {f1:.4%}")
    print("\nClassification Report:")
    print(report)
    print("====================================================")
    
    # Save evaluation plots
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Confusion Matrix
    plt.figure(figsize=(7, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malicious'], 
                yticklabels=['Benign', 'Malicious'])
    plt.title('Confusion Matrix on Disjoint Test Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eval_confusion_matrix.png"), dpi=150)
    plt.close()
    
    # Plot 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eval_roc_curve.png"), dpi=150)
    plt.close()
    
    # Plot 3: Precision-Recall Curve
    precision, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(recall_vals, precision, color='purple', lw=2, label=f'PR curve (AP = {ap:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eval_precision_recall_curve.png"), dpi=150)
    plt.close()
    
    print(f"\n✓ Evaluation plots saved to {output_dir}")

if __name__ == '__main__':
    processed_dir = os.path.join(BASE_DIR, "data_set", "processed")
    eval_output_dir = os.path.join(BASE_DIR, "output", "dataset_eval")
    
    train_and_evaluate(processed_dir, eval_output_dir)
