import os
import sys
import shutil
import numpy as np

# Xác định đường dẫn gốc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("\n❌ LỖI: Chưa cài đặt thư viện PyTorch (torch)!")
    print("Vui lòng chạy lệnh: ./venv/bin/pip install torch")
    sys.exit(1)

# Cấu hình đường dẫn
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'dl')
MODEL_PTH = os.path.join(OUTPUT_DIR, 'url_phishing_dl_model.pth')
MODEL_ONNX = os.path.join(OUTPUT_DIR, 'url_phishing_dl_model.onnx')
FLUTTER_MODEL_DIR = os.path.join(BASE_DIR, 'phising_detection', 'assets', 'models')
FLUTTER_MODEL_ONNX = os.path.join(FLUTTER_MODEL_DIR, 'url_phishing_dl_model.onnx')

# Các siêu tham số
MAX_LEN = 200
CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~:/?#[]@!$&'()*+,;= "
CHAR_TO_IDX = {char: idx + 2 for idx, char in enumerate(CHAR_VOCAB)} # 0: pad, 1: unk
CHAR_TO_IDX['<pad>'] = 0
CHAR_TO_IDX['<unk>'] = 1
VOCAB_SIZE = len(CHAR_TO_IDX)

# ----------------- Định nghĩa kiến trúc mô hình -----------------
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
        # x: (Batch, SeqLen)
        embedded = self.embedding(x).transpose(1, 2) # (Batch, EmbeddingDim, SeqLen)
        conv_out = self.relu(self.conv1d(embedded)) # (Batch, CnnFilters, SeqLen)
        pooled = self.max_pool(conv_out).transpose(1, 2) # (Batch, SeqLen // 2, CnnFilters)
        lstm_out, _ = self.bilstm(pooled) # (Batch, SeqLen // 2, HiddenDim * 2)
        repr_vec = lstm_out[:, -1, :] # (Batch, HiddenDim * 2)
        out = self.dropout(repr_vec)
        logits = self.fc(out) # (Batch, 1)
        return logits.squeeze(1) # (Batch,)

def tokenize_url(url: str, max_len: int = MAX_LEN) -> np.ndarray:
    if not isinstance(url, str):
        url = ""
    tokenized = []
    for char in url[:max_len]:
        tokenized.append(CHAR_TO_IDX.get(char, 1)) # 1: <unk>
    if len(tokenized) < max_len:
        tokenized += [0] * (max_len - len(tokenized)) # 0: <pad>
    return np.array(tokenized, dtype=np.int64)

def main():
    print("🚀 Khởi tạo quá trình chuyển đổi PyTorch sang ONNX...")
    
    if not os.path.exists(MODEL_PTH):
        print(f"❌ Không tìm thấy tệp trọng số mô hình tại: {MODEL_PTH}")
        sys.exit(1)
        
    # 1. Nạp mô hình PyTorch
    device = torch.device('cpu')
    model = PhishingCNNBiLSTM()
    model.load_state_dict(torch.load(MODEL_PTH, map_location=device))
    model.eval()
    print("✓ Đã nạp thành công mô hình PyTorch.")

    # 2. Xuất sang định dạng ONNX
    dummy_input = torch.zeros((1, MAX_LEN), dtype=torch.long)
    print(f"📦 Đang chuyển đổi sang định dạng ONNX tại: {MODEL_ONNX} ...")
    
    torch.onnx.export(
        model,
        dummy_input,
        MODEL_ONNX,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=12
    )
    print("✓ Đã xuất thành công sang ONNX.")

    # 3. Xác minh tính đúng đắn của mô hình sau chuyển đổi
    try:
        import onnxruntime as ort
    except ImportError:
        print("\n⚠️  Cảnh báo: Chưa cài đặt onnxruntime trong python environment để tự động xác minh.")
        print("Đang tiến hành cài đặt onnxruntime...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime"])
        import onnxruntime as ort

    print("\n🔍 Bắt đầu kịch bản xác minh dự đoán (PyTorch vs ONNX Runtime)...")
    
    # Một số URL kiểm thử đặc trưng
    test_urls = [
        "https://www.google.com",
        "http://phishing-scam-update-account-verification-login.com/secure",
        "https://github.com/login",
        "http://123.45.67.89/paypal/signin.php?email=test@example.com",
        "https://my-bank-secure-login-update.blogspot.com/2026/05/home.html",
        "https://amazon.com",
        "https://youtube.com/watch?v=dQw4w9WgXcQ"
    ]
    
    # Tokenize
    tokenized_inputs = [tokenize_url(url) for url in test_urls]
    input_batch = np.array(tokenized_inputs, dtype=np.int64)
    
    # Chạy suy luận trên PyTorch
    with torch.no_grad():
        pytorch_input = torch.tensor(input_batch)
        pytorch_logits = model(pytorch_input)
        pytorch_probs = torch.sigmoid(pytorch_logits).numpy()
    
    # Chạy suy luận trên ONNX Runtime
    ort_session = ort.InferenceSession(MODEL_ONNX)
    ort_inputs = {ort_session.get_inputs()[0].name: input_batch}
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_logits = ort_outputs[0]
    
    # Hàm sigmoid thủ công cho ONNX output
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    onnx_probs = sigmoid(onnx_logits)
    
    # Kiểm tra sai số tuyệt đối lớn nhất
    max_diff_prob = np.max(np.abs(pytorch_probs - onnx_probs))
    max_diff_logit = np.max(np.abs(pytorch_logits.numpy() - onnx_logits))
    
    print("\n---------------- KẾT QUẢ SO SÁNH DỰ ĐOÁN ----------------")
    for i, url in enumerate(test_urls):
        url_short = url[:50] + ("..." if len(url) > 50 else "")
        print(f"URL: {url_short}")
        print(f"  - PyTorch Prob: {pytorch_probs[i]:.6f}")
        print(f"  - ONNX Prob:    {onnx_probs[i]:.6f}")
        print(f"  - Sai lệch:     {abs(pytorch_probs[i] - onnx_probs[i]):.2e}")
    print("---------------------------------------------------------")
    print(f"🔥 Sai lệch Logit lớn nhất: {max_diff_logit:.2e}")
    print(f"🔥 Sai lệch Xác suất (Prob) lớn nhất: {max_diff_prob:.2e}")
    
    if max_diff_prob < 1e-5:
        print("\n✅ XÁC MINH THÀNH CÔNG: Mô hình ONNX hoạt động khớp hoàn toàn với mô hình PyTorch gốc!")
    else:
        print("\n❌ CẢNH BÁO: Phát hiện sai lệch đáng kể giữa PyTorch và ONNX. Vui lòng kiểm tra lại!")
        sys.exit(1)

    # 4. Sao chép mô hình ONNX sang assets của Flutter
    os.makedirs(FLUTTER_MODEL_DIR, exist_ok=True)
    shutil.copy2(MODEL_ONNX, FLUTTER_MODEL_ONNX)
    print(f"\n📂 Đã sao chép mô hình ONNX sang Flutter assets tại:\n   -> {FLUTTER_MODEL_ONNX}")
    print("\n🎉 HOÀN THÀNH XUẤT MÔ HÌNH DEEP LEARNING!")

if __name__ == '__main__':
    main()
