import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from utils.data_loader import load_dataset, split_data
from utils.metrics import calculate_metrics
from models.layers import SelfAttention # 必須匯入這個 Class 才能載入模型

parser = argparse.ArgumentParser(description='Predict and Evaluate Model')
parser.add_argument('--model_path', type=str, required=True, help='Path to the .h5 model file')
parser.add_argument('--data_path', type=str, default='./dataset/Dataset_BUSI_with_GT', help='Path to dataset')
parser.add_argument('--samples', type=int, default=5, help='Number of samples to visualize')

args = parser.parse_args()

def predict():
    # 1. 準備測試資料 (我們重新讀取並分割，確保使用的是同一份測試集)
    # 注意：實務上最好在訓練時就把 X_test 存起來，這裡為了簡化直接重新 split
    print("[Info] Loading test data...")
    X, y = load_dataset(args.data_path)
    _, X_test, _, y_test = split_data(X, y)
    print(f"[Info] Test data loaded. Shape: {X_test.shape}")

    # 2. 載入模型 (處理自定義層)
    print(f"[Info] Loading model from {args.model_path}...")
    try:
        # 使用 custom_object_scope 告訴 Keras 怎麼讀取 SelfAttention
        with tf.keras.utils.custom_object_scope({'SelfAttention': SelfAttention}):
            model = tf.keras.models.load_model(args.model_path)
    except Exception as e:
        print(f"[Error] Failed to load model. Did you train it with attention? {e}")
        return

    # 3. 進行預測
    print("[Info] Predicting...")
    y_pred = model.predict(X_test, verbose=1)
    
    # 4. 計算指標
    metrics = calculate_metrics(y_test, y_pred)
    print("\n" + "="*30)
    print("       Evaluation Metrics       ")
    print("="*30)
    for k, v in metrics.items():
        print(f"{k:15s}: {v:.4f}")
    print("="*30 + "\n")

    # 5. 視覺化結果
    visualize_results(model, X_test, y_test, num_samples=args.samples)

def visualize_results(model, X_test, y_test, num_samples=5):
    """隨機挑選樣本並畫出 Original Image, Ground Truth, Prediction"""
    indices = np.random.randint(0, X_test.shape[0], num_samples)
    
    fig, ax = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))
    
    for i, idx in enumerate(indices):
        # 原始圖片
        ax[i, 0].imshow(X_test[idx].squeeze(), cmap='gray')
        ax[i, 0].set_title('Original Image')
        ax[i, 0].axis('off')
        
        # 真實 Mask
        ax[i, 1].imshow(y_test[idx].squeeze(), cmap='gray')
        ax[i, 1].set_title('Ground Truth')
        ax[i, 1].axis('off')
        
        # 預測結果
        pred_img = model.predict(np.expand_dims(X_test[idx], 0), verbose=0)[0]
        # 二值化處理以便顯示
        pred_binary = (pred_img > 0.5).astype(np.float32)
        
        ax[i, 2].imshow(pred_binary.squeeze(), cmap='gray')
        ax[i, 2].set_title('Prediction')
        ax[i, 2].axis('off')
    
    plt.tight_layout()
    save_path = 'prediction_results.png'
    plt.savefig(save_path)
    print(f"[Info] Visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    predict()