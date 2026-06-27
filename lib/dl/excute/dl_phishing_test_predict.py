"""
dl_phishing_test_predict.py
===========================
Chương trình dự đoán Phishing URL sử dụng mô hình Deep Learning (CNN-BiLSTM) đã huấn luyện.
Hỗ trợ 3 chế độ:
  1. Kiểm tra 1 URL đơn lẻ bằng tham số truyền vào: --url "..."
  2. CLI Interactive Mode: Nhập URL liên tiếp để dự đoán, gõ 'quit' để thoát.
  3. Tích hợp cơ chế Whitelist danh tiếng tên miền loại bỏ False Positives.
"""

import os
import sys
import time
import argparse
import numpy as np

# Xác định đường dẫn gốc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, BASE_DIR)

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("\n❌ LỖI: Chưa cài đặt thư viện PyTorch (torch)!")
    print("Vui lòng chạy lệnh sau để cài đặt PyTorch trước:")
    print("  ./venv/bin/pip install torch")
    sys.exit(1)

# Cấu hình đường dẫn mô hình
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'dl')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'url_phishing_dl_model.pth')

# Whitelist tên miền uy tín hàng đầu
REPUTABLE_DOMAINS = {
    'google.com', 'google.com.vn', 'youtube.com', 'facebook.com', 'instagram.com', 
    'twitter.com', 'linkedin.com', 'github.com', 'gitlab.com', 'microsoft.com', 
    'apple.com', 'amazon.com', 'netflix.com', 'wikipedia.org', 'w3schools.com', 
    'stackoverflow.com', 'stackexchange.com', 'medium.com', 'docker.com', 'docker.io', 
    'kubernetes.io', 'python.org', 'npmjs.com', 'cloudflare.com', 'mozilla.org', 
    'apache.org', 'spring.io', 'oracle.com', 'git-scm.com', 'bitbucket.org'
}

# ----------------- Bộ Tokenizer cấp ký tự -----------------
MAX_LEN = 200
CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~:/?#[]@!$&'()*+,;= "
CHAR_TO_IDX = {char: idx + 2 for idx, char in enumerate(CHAR_VOCAB)} # 0: pad, 1: unk
CHAR_TO_IDX['<pad>'] = 0
CHAR_TO_IDX['<unk>'] = 1
VOCAB_SIZE = len(CHAR_TO_IDX)

# ----------------- Khai báo kiến trúc mô hình -----------------
class PhishingCNNBiLSTM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=64, 
                 cnn_filters=128, kernel_size=5, lstm_hidden=64, dropout=0.5):
        super(PhishingCNNBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=cnn_filters, 
                                kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.bilstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden, 
                              num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, 1)
        
    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)
        conv_out = self.relu(self.conv1d(embedded))
        pooled = self.max_pool(conv_out).transpose(1, 2)
        lstm_out, _ = self.bilstm(pooled)
        repr_vec = lstm_out[:, -1, :]
        out = self.dropout(repr_vec)
        logits = self.fc(out)
        return logits.squeeze(1)

def tokenize_url(url: str, max_len: int = MAX_LEN) -> np.ndarray:
    """Chuyển đổi URL dạng chuỗi thành mảng các chỉ số ký tự với độ dài cố định."""
    if not isinstance(url, str):
        url = ""
    tokenized = []
    for char in url[:max_len]:
        tokenized.append(CHAR_TO_IDX.get(char, 1)) # 1: <unk>
    if len(tokenized) < max_len:
        tokenized += [0] * (max_len - len(tokenized)) # 0: <pad>
    return np.array(tokenized, dtype=np.int64)

def get_registered_domain(url: str) -> str:
    """Trích xuất registered domain đơn giản để kiểm tra whitelist."""
    # Loại bỏ scheme
    domain = url.lower()
    if "://" in domain:
        domain = domain.split("://")[1]
    domain = domain.split("/")[0].split("?")[0].split(":")[0]
    
    parts = domain.split('.')
    if len(parts) >= 2:
        # Xử lý các đuôi com.vn, co.uk...
        if len(parts) >= 3 and parts[-2] in ('co', 'com', 'org', 'net', 'edu', 'gov'):
            return '.'.join(parts[-3:])
        return '.'.join(parts[-2:])
    return domain

def load_model(device):
    """Tải trọng số mô hình đã huấn luyện."""
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ LỖI: Không tìm thấy tệp trọng số mô hình tại: {MODEL_PATH}")
        print("Vui lòng chạy file huấn luyện trước: ./venv/bin/python lib/dl/excute/train_dl.py")
        sys.exit(1)
        
    model = PhishingCNNBiLSTM()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_url(url: str, model, device):
    """Dự đoán URL là độc hại hay lành tính bằng mô hình đã nạp."""
    reg_domain = get_registered_domain(url)
    is_whitelisted = reg_domain in REPUTABLE_DOMAINS
    
    start_time = time.time()
    
    if is_whitelisted:
        prediction = 0
        prob_phishing = 0.0
        prob_legitimate = 1.0
        label = "LEGITIMATE ✅ (Danh tiếng uy tín)"
    else:
        # Chuẩn bị Tensor đầu vào
        tokenized = tokenize_url(url)
        input_tensor = torch.tensor(np.array([tokenized])).to(device)
        
        with torch.no_grad():
            logits = model(input_tensor)
            prob_phishing = torch.sigmoid(logits).item()
            prob_legitimate = 1.0 - prob_phishing
            prediction = 1 if prob_phishing >= 0.5 else 0
            label = "PHISHING 🚨" if prediction == 1 else "LEGITIMATE ✅"
            
    inference_time_ms = (time.time() - start_time) * 1000
    
    return {
        'url': url,
        'prediction': prediction,
        'label': label,
        'prob_phishing': prob_phishing,
        'prob_legitimate': prob_legitimate,
        'confidence': max(prob_phishing, prob_legitimate),
        'is_whitelisted': is_whitelisted,
        'inference_time_ms': inference_time_ms
    }

def display_result(result):
    """Hiển thị kết quả ra màn hình dạng hộp văn bản trực quan."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print(f"║  🎯 KẾT QUẢ DỰ ĐOÁN URL (CNN-BILSTM DEEP LEARNING){'':>17}║")
    print("╠" + "═" * 68 + "╣")
    url_trimmed = result['url'][:60] + ('...' if len(result['url']) > 60 else '')
    print(f"║  🔗 URL: {url_trimmed:<58}║")
    print("╠" + "═" * 68 + "╣")

    if result['prediction'] == 1:
        print(f"║  ⚠️  KẾT QUẢ: {result['label']:<47}║")
        print(f"║  📈 Xác suất Phishing:   {result['prob_phishing']:.2%}{'':>37}║")
    else:
        print(f"║  ✅ KẾT QUẢ: {result['label']:<47}║")
        print(f"║  📈 Xác suất An toàn:    {result['prob_legitimate']:.2%}{'':>37}║")

    print(f"║  🔥 Độ tin cậy:          {result['confidence']:.2%}{'':>37}║")
    print(f"║  ⚡ Thời gian xử lý:     {result['inference_time_ms']:.2f} ms{'':>37}║")
    print("╚" + "═" * 68 + "╝")

def interactive_mode(model, device):
    """Giao diện CLI tương tác liên tục."""
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  DEEP LEARNING - CHARACTER CNN-BILSTM DETECTOR           ║")
    print("║  Nhập URL cần kiểm tra (Nhập 'quit' để thoát)           ║")
    print("╚══════════════════════════════════════════════════════════╝")

    while True:
        print("\n" + "─" * 70)
        url = input("🔗 Nhập URL: ").strip()

        if url.lower() in ('quit', 'exit', 'q'):
            print("\n👋 Tạm biệt!")
            break

        if not url:
            print("⚠ Vui lòng nhập chuỗi URL hợp lệ!")
            continue

        try:
            result = predict_url(url, model, device)
            display_result(result)
        except Exception as e:
            print(f"❌ Có lỗi xảy ra trong quá trình dự đoán: {e}")

def main():
    parser = argparse.ArgumentParser(description="CNN-BiLSTM Phishing URL Predictor")
    parser.add_argument('--url', type=str, help='Đường dẫn URL cần kiểm tra')
    args = parser.parse_args()

    # Chọn thiết bị tối ưu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = load_model(device)

    if args.url:
        result = predict_url(args.url, model, device)
        display_result(result)
    else:
        interactive_mode(model, device)

if __name__ == '__main__':
    main()
