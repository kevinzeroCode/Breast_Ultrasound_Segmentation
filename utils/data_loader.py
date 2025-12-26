import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

def load_image(path, size):
    """讀取並預處理單張圖片"""
    image = cv2.imread(path)
    image = cv2.resize(image, (size, size))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image / 255.0  # Normalize
    return image

def load_dataset(root_path, size=128):
    """
    從指定路徑讀取 Dataset，處理多個 Mask 的合併邏輯
    回傳: X (images), y (masks)
    """
    images = []
    masks = []
    
    # 搜尋所有子資料夾內的檔案
    paths = sorted(glob(os.path.join(root_path, '*/*')))
    
    x = 0  # flag to identify images with multiple masks
    
    for path in paths:
        img = load_image(path, size)
        
        if 'mask' in path:
            if x:  # 如果這張圖對應多個 mask
                masks[-1] += img
                # 確保數值在 0-1 之間 (Binary)
                masks[-1] = np.array(masks[-1] > 0.5, dtype='float64')
            else:
                masks.append(img)
                x = 1  # 標記目前正在處理 mask
        else:
            images.append(img)
            x = 0  # 遇到新圖片，重置 flag

    X = np.array(images)
    y = np.array(masks)
    
    # 擴增維度以符合 Keras 輸入 (H, W, 1)
    X = np.expand_dims(X, -1)
    y = np.expand_dims(y, -1)
    
    return X, y

def split_data(X, y, test_size=0.1, random_state=42):
    """分割訓練集與測試集"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)