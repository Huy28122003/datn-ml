"""
export_to_onnx.py
===================
Script chuyển đổi mô hình RF Hybrid và DL sang định dạng ONNX
để nhúng và chạy offline trên ứng dụng Flutter:
  1. Random Forest (Hybrid - 25 đặc trưng)
  2. Deep Learning CNN-BiLSTM (đã được export riêng trong training script)

Yêu cầu cài đặt các thư viện chuyển đổi trước khi chạy:
  pip install skl2onnx onnx
"""

import os
import sys
import json
import pickle

# Xác định đường dẫn gốc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RF_HYBRID_PKL = os.path.join(BASE_DIR, 'output', 'rf_vs_hybrid', 'rf_phishing_model.pkl')
RF_HYBRID_ONNX = os.path.join(BASE_DIR, 'output', 'rf_vs_hybrid', 'rf_phishing_model.onnx')

FLUTTER_CONFIG_PATH = os.path.join(BASE_DIR, 'output', 'flutter_models_config.json')


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    print("🚀 Bắt đầu quá trình chuyển đổi mô hình sang ONNX...")

    # Kiểm tra cài đặt thư viện chuyển đổi
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType as SklFloatTensorType
    except ImportError:
        print("\n❌ LỖI: Thiếu các thư viện hỗ trợ ONNX.")
        print("  → Vui lòng cài đặt bằng lệnh:")
        print("    pip install skl2onnx onnx")
        sys.exit(1)

    features_config = {}

    # 1. Chuyển đổi Random Forest (Hybrid)
    if os.path.exists(RF_HYBRID_PKL):
        print("\n📦 Đang chuyển đổi Random Forest (Hybrid)...")
        rf_hybrid_data = load_pickle(RF_HYBRID_PKL)
        model = rf_hybrid_data['model']
        features = rf_hybrid_data['features']
        features_config['rf_hybrid'] = features
        
        initial_type = [('float_input', SklFloatTensorType([None, len(features)]))]
        onx = convert_sklearn(model, initial_types=initial_type, target_opset=12)
        with open(RF_HYBRID_ONNX, 'wb') as f:
            f.write(onx.SerializeToString())
        print(f"  ✓ Đã xuất: {RF_HYBRID_ONNX}")
        print(f"  ✓ Số đặc trưng: {len(features)}")
    else:
        print(f"⚠️  Không tìm thấy file: {RF_HYBRID_PKL}")

    # 2. Xuất danh sách đặc trưng ra JSON cho Flutter sử dụng để trích xuất đúng thứ tự
    with open(FLUTTER_CONFIG_PATH, 'w') as f:
        json.dump(features_config, f, indent=4)
    print(f"\n📂 Đã lưu danh sách đặc trưng cấu hình Flutter tại: {FLUTTER_CONFIG_PATH}")
    print("\n🎉 HOÀN THÀNH CHUYỂN ĐỔI SANG ONNX!")
    print("\n📋 Lưu ý: Model DL (url_phishing_dl_model.onnx) đã được export")
    print("   trong quá trình training. Không cần chuyển đổi thêm.")


if __name__ == '__main__':
    main()
