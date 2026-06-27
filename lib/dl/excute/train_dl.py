import os
import sys
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the workspace root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("\n❌ LỖI: Chưa cài đặt thư viện PyTorch (torch)!")
    print("Vui lòng chạy lệnh sau để cài đặt PyTorch trước:")
    print("  ./venv/bin/pip install torch")
    sys.exit(1)

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, 
    precision_recall_curve, average_precision_score, accuracy_score, f1_score, recall_score
)

# ----------------- Cấu hình chung -----------------
DATASET_DIR = os.path.join(BASE_DIR, 'data_set', 'processed') # hoặc 'data_set/dl_dataset/processed'
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'dl')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'url_phishing_dl_model.pth')
METRICS_SAVE_PATH = os.path.join(OUTPUT_DIR, 'dl_metrics.json')

# Siêu tham số mặc định
MAX_LEN = 200        # Độ dài tối đa của URL
BATCH_SIZE = 128     # Kích thước Batch
EPOCHS = 15          # Số Epoch tối đa
LEARNING_RATE = 0.001
EMBEDDING_DIM = 64
CNN_FILTERS = 128
CNN_KERNEL_SIZE = 5
LSTM_HIDDEN_DIM = 64
DROPOUT = 0.5

# ----------------- Bộ Tokenizer cấp ký tự -----------------
# Tập ký tự ASCII hiển thị được làm từ điển
CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~:/?#[]@!$&'()*+,;= "
CHAR_TO_IDX = {char: idx + 2 for idx, char in enumerate(CHAR_VOCAB)} # 0: padding, 1: unknown
CHAR_TO_IDX['<pad>'] = 0
CHAR_TO_IDX['<unk>'] = 1
VOCAB_SIZE = len(CHAR_TO_IDX)

def tokenize_url(url: str, max_len: int = MAX_LEN) -> np.ndarray:
    """Chuyển đổi URL dạng chuỗi thành mảng các chỉ số ký tự với độ dài cố định."""
    if not isinstance(url, str):
        url = ""
    tokenized = []
    for char in url[:max_len]:
        tokenized.append(CHAR_TO_IDX.get(char, 1)) # 1 đại diện cho <unk>
    # Padding nếu ngắn hơn max_len
    if len(tokenized) < max_len:
        tokenized += [0] * (max_len - len(tokenized)) # 0 đại diện cho <pad>
    return np.array(tokenized, dtype=np.int64)

# ----------------- PyTorch Dataset -----------------
class PhishingURLDataset(Dataset):
    def __init__(self, csv_path: str, max_len: int = MAX_LEN):
        print(f"Đang tải dữ liệu từ {csv_path}...")
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['url', 'result']).copy()
        
        self.urls = df['url'].values
        self.labels = df['result'].values.astype(np.float32)
        self.max_len = max_len
        
        # Tokenize toàn bộ URLs trước để tăng tốc độ huấn luyện
        self.tokenized_urls = [tokenize_url(url, self.max_len) for url in self.urls]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.tokenized_urls[idx]), torch.tensor(self.labels[idx])

# ----------------- Mô hình CNN + BiLSTM -----------------
class PhishingCNNBiLSTM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, 
                 cnn_filters=CNN_FILTERS, kernel_size=CNN_KERNEL_SIZE,
                 lstm_hidden=LSTM_HIDDEN_DIM, dropout=DROPOUT):
        super(PhishingCNNBiLSTM, self).__init__()
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2. 1D Convolutional Layer
        # Conv1d yêu cầu đầu vào dạng (Batch, Channel, Length), channel ở đây là embedding_dim
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=cnn_filters, 
                                kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        
        # 3. Max Pooling Layer
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        
        # 4. Bidirectional LSTM Layer
        # Sau max pool 1d với kernel=2, chiều dài chuỗi giảm đi một nửa
        self.bilstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden, 
                              num_layers=1, bidirectional=True, batch_first=True)
        
        # 5. Fully Connected Layers
        self.dropout = nn.Dropout(dropout)
        # BiLSTM trả về output có chiều ẩn gấp đôi (vì là bidirectional)
        self.fc = nn.Linear(lstm_hidden * 2, 1)
        
    def forward(self, x):
        # x: (Batch, SeqLen)
        embedded = self.embedding(x) # (Batch, SeqLen, EmbeddingDim)
        
        # Chuyển trục để đưa vào Conv1d: (Batch, EmbeddingDim, SeqLen)
        embedded = embedded.transpose(1, 2)
        
        conv_out = self.conv1d(embedded) # (Batch, CnnFilters, SeqLen)
        conv_out = self.relu(conv_out)
        
        pooled = self.max_pool(conv_out) # (Batch, CnnFilters, SeqLen // 2)
        
        # Chuyển trục về dạng cho LSTM: (Batch, SeqLen // 2, CnnFilters)
        lstm_in = pooled.transpose(1, 2)
        
        # lstm_out: (Batch, SeqLen // 2, HiddenDim * 2)
        lstm_out, _ = self.bilstm(lstm_in)
        
        # Lấy hidden state cuối cùng (ở vị trí cuối cùng của chuỗi) làm vector đại diện
        # lstm_out[:, -1, :] có kích thước (Batch, HiddenDim * 2)
        repr_vec = lstm_out[:, -1, :]
        
        out = self.dropout(repr_vec)
        logits = self.fc(out) # (Batch, 1)
        
        return logits.squeeze(1)

# ----------------- Quy trình Huấn luyện -----------------
def train_model(train_loader, val_loader, model, device, epochs=EPOCHS, lr=LEARNING_RATE):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\n⚡ Đang huấn luyện trên thiết bị: {device}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        start_time = time.time()
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct_train += (preds == y_batch).sum().item()
            total_train += y_batch.size(0)
            
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct_val += (preds == y_batch).sum().item()
                total_val += y_batch.size(0)
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{epochs:02d} [{elapsed:.1f}s] | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.4%} | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4%}")
              
        # Early Stopping & Lưu mô hình tốt nhất
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ Đã lưu mô hình tốt nhất tại: {MODEL_SAVE_PATH}")
            
    return history

# ----------------- Đánh giá & Vẽ đồ thị -----------------
def evaluate_model(test_loader, model, history, device):
    print("\n📊 Đang tải trọng số mô hình tốt nhất để đánh giá tập Test...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    all_preds = []
    all_probas = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probas = torch.sigmoid(logits).cpu().numpy()
            preds = (probas >= 0.5).astype(np.float32)
            
            all_preds.extend(preds)
            all_probas.extend(probas)
            all_labels.extend(y_batch.numpy())
            
    all_preds = np.array(all_preds)
    all_probas = np.array(all_probas)
    all_labels = np.array(all_labels)
    
    # Tính toán metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    
    print("\n================ DEEP LEARNING EVALUATION ================")
    print(f"Accuracy:  {acc:.4%}")
    print(f"Recall:    {recall:.4%}")
    print(f"F1-Score:  {f1:.4%}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malicious']))
    print("==========================================================")
    
    # Lưu tóm tắt metrics JSON
    metrics_summary = {
        'accuracy': float(acc),
        'recall': float(recall),
        'f1_score': float(f1),
        'final_train_loss': float(history['train_loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1])
    }
    with open(METRICS_SAVE_PATH, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"✓ Đã lưu file cấu hình metrics tại: {METRICS_SAVE_PATH}")
    
    # ----------------- Vẽ các đồ thị liên quan -----------------
    sns.set_theme(style="whitegrid")
    
    # Đồ thị 1: Biểu đồ Loss (Train vs Val)
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train Loss', color='royalblue', lw=2)
    plt.plot(history['val_loss'], label='Val Loss', color='orange', lw=2, linestyle='--')
    plt.title('Deep Learning - Quá trình tối ưu Loss', fontsize=14, pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dl_training_loss.png'), dpi=150)
    plt.close()
    
    # Đồ thị 2: Biểu đồ Accuracy (Train vs Val)
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_acc'], label='Train Accuracy', color='forestgreen', lw=2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='firebrick', lw=2, linestyle='--')
    plt.title('Deep Learning - Tăng trưởng Accuracy', fontsize=14, pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dl_training_accuracy.png'), dpi=150)
    plt.close()
    
    # Đồ thị 3: Confusion Matrix
    plt.figure(figsize=(7, 5))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Benign', 'Malicious'], 
                yticklabels=['Benign', 'Malicious'],
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.title('CNN-BiLSTM - Ma trận nhầm lẫn (Confusion Matrix)', fontsize=14, pad=15)
    plt.xlabel('Nhãn Dự Đoán', fontsize=12)
    plt.ylabel('Nhãn Thực Tế', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dl_confusion_matrix.png'), dpi=150)
    plt.close()
    
    # Đồ thị 4: ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probas)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('CNN-BiLSTM - Đường cong ROC', fontsize=14, pad=15)
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dl_roc_curve.png'), dpi=150)
    plt.close()
    
    # Đồ thị 5: Precision-Recall Curve
    precision, recall_vals, _ = precision_recall_curve(all_labels, all_probas)
    ap = average_precision_score(all_labels, all_probas)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision, color='purple', lw=2.5, label=f'Precision-Recall (AP = {ap:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('CNN-BiLSTM - Đường cong Precision-Recall', fontsize=14, pad=15)
    plt.xlabel('Tỷ lệ phát hiện đúng (Recall)', fontsize=12)
    plt.ylabel('Độ chính xác lớp cảnh báo (Precision)', fontsize=12)
    plt.legend(loc="lower left", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dl_precision_recall_curve.png'), dpi=150)
    plt.close()
    
    print(f"✓ Toàn bộ các biểu đồ phân tích đánh giá đã lưu tại: {OUTPUT_DIR}")

# ----------------- Hàm chính -----------------
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  HUẤN LUYỆN MÔ HÌNH DEEP LEARNING (CHARACTER CNN-BILSTM) ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # Tự động chọn GPU nếu có (CUDA cho NVIDIA, MPS cho Apple Silicon Mac)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    # Xác định đường dẫn file dữ liệu
    # Thử cả 2 đường dẫn để tăng độ linh hoạt của thư mục đầu vào
    train_path = os.path.join(DATASET_DIR, "train_processed.csv")
    val_path = os.path.join(DATASET_DIR, "val_processed.csv")
    test_path = os.path.join(DATASET_DIR, "test_processed.csv")
    
    if not os.path.exists(train_path):
        # Đường dẫn thay thế nếu chạy trong thư mục con khác
        alt_dataset_dir = os.path.join(BASE_DIR, 'data_set', 'dl_dataset', 'processed')
        train_path = os.path.join(alt_dataset_dir, "train_processed.csv")
        val_path = os.path.join(alt_dataset_dir, "val_processed.csv")
        test_path = os.path.join(alt_dataset_dir, "test_processed.csv")
        
    if not os.path.exists(train_path):
        print(f"❌ LỖI: Không tìm thấy tập dữ liệu huấn luyện tại: {train_path}")
        print("Vui lòng chạy pipeline tối ưu hóa trước để tạo các file processed.csv!")
        sys.exit(1)
        
    # Khởi tạo datasets & dataloaders
    train_dataset = PhishingURLDataset(train_path)
    val_dataset = PhishingURLDataset(val_path)
    test_dataset = PhishingURLDataset(test_path)
    
    # Tạo DataLoaders để sinh batches khi train
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Khởi tạo mô hình
    model = PhishingCNNBiLSTM().to(device)
    
    # In thông tin mô hình
    print("\n🤖 Kiến trúc mạng Neural (Character CNN-BiLSTM):")
    print(model)
    
    # Huấn luyện mô hình
    history = train_model(train_loader, val_loader, model, device)
    
    # Đánh giá trên tập kiểm thử
    evaluate_model(test_loader, model, history, device)
    
    print("\n🎉 HOÀN TẤT HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH DEEP LEARNING!")

if __name__ == '__main__':
    main()
