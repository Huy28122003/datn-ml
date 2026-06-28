import os
import sys
import json
import pickle
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RF_HYBRID_PKL = os.path.join(BASE_DIR, 'output', 'rf_vs_hybrid', 'rf_phishing_model.pkl')
RF_HYBRID_ONNX = os.path.join(BASE_DIR, 'output', 'rf_vs_hybrid', 'rf_phishing_model.onnx')
FLUTTER_CONFIG_PATH = os.path.join(BASE_DIR, 'output', 'flutter_models_config.json')

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType as SklFloatTensorType
    except ImportError:
        sys.exit(1)
    features_config = {}
    if os.path.exists(RF_HYBRID_PKL):
        rf_hybrid_data = load_pickle(RF_HYBRID_PKL)
        model = rf_hybrid_data['model']
        features = rf_hybrid_data['features']
        features_config['rf_hybrid'] = features
        initial_type = [('float_input', SklFloatTensorType([None, len(features)]))]
        onx = convert_sklearn(model, initial_types=initial_type, target_opset=12)
        with open(RF_HYBRID_ONNX, 'wb') as f:
            f.write(onx.SerializeToString())
    with open(FLUTTER_CONFIG_PATH, 'w') as f:
        json.dump(features_config, f, indent=4)
if __name__ == '__main__':
    main()
