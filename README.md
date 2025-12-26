# Integration of Self-Attention with U-Net for Breast Tumor Segmentation

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository implements the research paper **"Integration Self-attention with UNet for Tumor Segmentation in Breast Ultrasound"**.

We propose a hybrid deep learning model that incorporates **Self-Attention mechanisms** into the standard **U-Net** architecture. [cite_start]This enhancement allows the model to capture global contextual dependencies between features, significantly improving segmentation accuracy and robustness compared to traditional CNN-based methods[cite: 10, 22].


*(Place your architecture diagram here, e.g., docs/images/architecture.png)*

---

## 📖 Background & Motivation

### The Problem
Traditional U-Net architectures rely primarily on convolution operations. [cite_start]While effective at extracting local features, they often struggle to capture **global information** and long-range dependencies, which limits performance in complex medical image segmentation tasks[cite: 17].

### Our Solution
[cite_start]Inspired by the success of Transformers in NLP and Vision (ViT, Swin Transformer)[cite: 26, 30], we integrated a **Self-Attention Mechanism** into the U-Net decoder and skip connections. This allows the model to:
1.  [cite_start]**Global Context:** Enable each pixel to "attend" to all other pixels, understanding the global structure[cite: 43].
2.  [cite_start]**Feature Integration:** Better fuse multi-scale feature maps from the encoder and decoder[cite: 22].

---

## 📂 Project Structure

The project is refactored into a modular structure for maintainability and scalability:

```text
Breast_Ultrasound_Segmentation/
├── dataset/                # Dataset directory (BUSI)
├── models/                 # Model definitions
│   ├── layers.py           # Custom Self-Attention Layer
│   └── architecture.py     # U-Net & Attention U-Net construction
├── utils/                  # Helper functions
│   ├── data_loader.py      # Image processing & mask merging
│   └── metrics.py          # IoU, F1-Score calculations
├── train.py                # Main training script
├── predict.py              # Inference & Visualization script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🛠️ InstallationClone the repository
```
bash
git clone [https://github.com/your-username/Breast_Ultrasound_Segmentation.git](https://github.com/your-username/Breast_Ultrasound_Segmentation.git)
cd Breast_Ultrasound_Segmentation
```
Install dependencies
```
bash
pip install -r requirements.txt
```
### Dataset Preparation

[cite_start]Download the **Breast Ultrasound Images Dataset (BUSI)** [cite: 87, 88] and place it in the `dataset/` folder.
The directory structure should look like this:

```text
dataset/Dataset_BUSI_with_GT/
├── benign/
├── malignant/
└── normal/

## 🚀 Usage
1. Train the ModelYou can train either the standard U-Net (baseline) or the proposed Attention U-Net.

Train Attention U-Net (Proposed Method):
```
bash
python train.py --model attention --epochs 100 --batch_size 16
```
Train Standard U-Net (Baseline):
```
bashpython train.py --model unet --epochs 100 --batch_size 16
```
Training logs and the best model weights (best_attention_model.h5) will be saved automatically.

2. Evaluation & Prediction
Run the prediction script to calculate metrics (IoU, F1 Score) and visualize segmentation results.
```
bash
python predict.py --model_path best_attention_model.h5 --samples 5
```

## 📊 Experimental ResultsWe evaluated the model on the BUSI dataset. The integration of Self-Attention, particularly in the Decoder and Skip Connections, showed superior performance.
Quantitative Comparison
Model Architecture,Mean IoU,Precision,Recall,F1 Score
Standard U-Net,0.764,0.743,0.718,0.730
Attention U-Net (Ours),0.790,0.756,0.789,0.772
(Metrics based on our experimental logs)

Qualitative Visualization
Loss & Accuracy Curves: (Self-Attention U-Net shows faster convergence and higher validation accuracy)

Segmentation Output: (Left: Original, Middle: Ground Truth, Right: Model Prediction)


## 🧬 Methodology Details
Self-Attention Layer
We implemented a custom Keras layer that computes the attention map :

$$ Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V $$

This mechanism is inserted into the decoder blocks to filter features before concatenation.

Implementation Details

Input Size: 128x128 pixels 


Optimizer: Adam (Learning Rate = 0.001) 

Loss Function: Binary Cross-Entropy


Hardware: Trained on NVIDIA GeForce RTX 4090

## 📝 Citation

If you find this project useful, please reference the original paper:

Integration Self-attention with UNet for Tumor Segmentation in Breast Ultrasound Chii-Jen Chen, Yu-Jie Chiou, Shao-Hua Hsu, Yu-Cheng Chang Department of Computer Science and Information Engineering, Tamkang University.