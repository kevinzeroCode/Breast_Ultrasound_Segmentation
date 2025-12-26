import os
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping

# 匯入我們寫好的模組
from utils.data_loader import load_dataset, split_data
from models.architecture import build_unet

# 設定參數解析
parser = argparse.ArgumentParser(description='Train Breast Tumor Segmentation Model')
parser.add_argument('--model', type=str, default='attention', choices=['unet', 'attention'], help='Choose model type: unet or attention')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--data_path', type=str, default='./dataset/Dataset_BUSI_with_GT', help='Path to dataset')

args = parser.parse_args()

def train():
    # 1. 準備資料
    print(f"[Info] Loading data from {args.data_path}...")
    X, y = load_dataset(args.data_path)
    print(f"[Info] Data loaded. X shape: {X.shape}, y shape: {y.shape}")
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"[Info] Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # 2. 建構模型
    use_attention = (args.model == 'attention')
    print(f"[Info] Building model: {'Attention U-Net' if use_attention else 'Standard U-Net'}...")
    
    model = build_unet(input_shape=(128, 128, 1), use_attention=use_attention)
    model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
    # model.summary() # 如果想看架構可以打開

    # 3. 設定 Callbacks
    model_filename = f"best_{args.model}_model.h5"
    callbacks = [
        ModelCheckpoint(model_filename, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        EarlyStopping(patience=15, monitor='val_loss', verbose=1) # 增加早停機制避免過擬合
    ]

    # 4. 開始訓練
    print("[Info] Start training...")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )

    # 5. 繪製並儲存結果圖
    plot_history(history, args.model)
    print(f"[Info] Training finished. Best model saved to {model_filename}")

def plot_history(history, model_name):
    """繪製 Loss 和 Accuracy 曲線"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss
    ax[0].plot(history.history["loss"], label="Train loss")
    ax[0].plot(history.history["val_loss"], label="Validation loss")
    ax[0].set_title(f"{model_name} Loss")
    ax[0].legend()
    
    # Accuracy
    ax[1].plot(history.history["accuracy"], label="Train accuracy")
    ax[1].plot(history.history["val_accuracy"], label="Validation accuracy")
    ax[1].set_title(f"{model_name} Accuracy")
    ax[1].legend()
    
    save_path = f"{model_name}_training_history.png"
    plt.savefig(save_path)
    print(f"[Info] History plot saved to {save_path}")

if __name__ == "__main__":
    # 確保 GPU 可用 (Optional)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    train()