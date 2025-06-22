# 🛰️ SAR Target Classification Using Deep Learning  
### Joint Despeckling-Recognition CNN with Task-Driven Pruning (TDP-SAR)

This repository implements an advanced pipeline for classifying ground military targets using Synthetic Aperture Radar (SAR) imagery. It features a custom dual-branch CNN architecture (J-CNN) combined with **Task-Driven Pruning (TDP-SAR)** for optimized inference speed, accuracy, and noise resilience.

---

## 🌟 Core Innovations

✅ A frequency-domain optimized deep learning system that:

1. 🌀 **Despeckles** SAR images using physics-aware constraints  
2. 🎯 **Classifies** targets while preserving geometric integrity  
3. ✂️ **Prunes** ~17.7% of filters without accuracy degradation

---

## 🧠 Theoretical Foundation

### 🎯 Joint Despeckling + Classification Loss (J-CNN)

The model uses a coupled loss function for joint learning:

```math
\mathcal{L} = \|f[\varphi(X)] - y\|_2^2 + \lambda \|\varphi(X) - Y_{clean}\|_2^2 + \eta \|w\|_1
```
Where:

φ(X) is the speckle-suppressed output

f[·] is the classification subnetwork

λ = 0.3 balances the dual tasks .

![Image](https://github.com/user-attachments/assets/5369ac57-dd34-4b82-b943-eaeaf60955c5) 

### 2. TDP-SAR Pruning
Task	Analysis Method	Math Criterion	Threshold
Despeckling	Amplitude Spectrum	$r=\frac{P_h}{P_l}>th_r$	0.7
Recognition	Phase Correlation	$Corr[f_k,f_r]<th_c$	0.5

## 📊 Performance Highlights
Metric	Baseline J-CNN	TDP-SAR
Accuracy (L=1)	89.2%	91.1%
Parameters	1.2M	0.98M
Inference Time (GPU)	22ms	17ms
Robustness (L=0.2)	72.3%	83.4%

## System Architecture
![Image](https://github.com/user-attachments/assets/35a3d1ef-7f31-42a4-9a9c-8eb087906c9a)

## ✂️ Task-Driven Pruning (TDP-SAR)

| **Task**       | **Analysis Method**     | **Pruning Rule**                          | **Threshold** |
|----------------|-------------------------|-------------------------------------------|---------------|
| Despeckling    | Amplitude Spectrum      | \( \frac{P_h}{P_l} > t_r \)               | 0.7           |
| Recognition    | Phase Correlation       | \( \text{Corr}[f_k, f_r] < t_c \)         | 0.5           |

---

## 📚 Dataset

The project uses the **MSTAR (Moving and Stationary Target Acquisition and Recognition)** dataset for training and evaluation. Images are captured at 15° and 17° depression angles using X-band radar.

### Classes:
- **2S1**: Self-propelled Artillery  
- **BRDM_2**: Armored Reconnaissance Vehicle  
- **BTR_60**: Armored Personnel Carrier  
- **D7**: Armored Bulldozer  
- **SLICY**: Calibration Target  
- **T62**: Main Battle Tank  
- **ZIL131**: Military Truck  
- **ZSU_23_4**: Self-propelled Anti-Aircraft Gun  

---

## ⚙️ Methodology

### 📦 Preprocessing

- SAR images normalized to `[0, 1]`
- Added synthetic Gamma-distributed speckle noise with \( L \in \{0.2, 1, 5\} \)
- Dataset split: **80% train / 20% validation**

### 🧠 Model Architecture

| **Layer**             | **Details**                             | **Output Shape**     |
|-----------------------|------------------------------------------|----------------------|
| Input                 | Grayscale SAR (128×128)                  | (128, 128, 1)        |
| Despeckling Branch    | Conv2D (16→256→16, 3×3)                  | (128, 128, 16)       |
| Recognition Branch    | Conv2D (6→36, 9×9)                       | (128, 128, 36)       |
| Global Avg Pooling    | —                                        | (36,)                |
| Dense + Softmax       | Fully connected → 10 target classes      | (10,)                |

---

## 📊 Results

### 📌 Performance Metrics

| **Metric**                 | **Baseline J-CNN** | **TDP-SAR Pruned** |
|---------------------------|--------------------|---------------------|
| Accuracy (L = 1)          | 94.0%              | 93.7%               |
| Parameters                | 1.2M               | 0.98M               |
| Inference Time (GPU)      | 22ms               | 17ms                |
| Robustness (L = 0.2 noise)| 72.3%              | 83.4%               |
| PSNR Improvement          | —                  | +4.2 dB             |


---

## 🔍 Challenges

- SAR imagery is prone to:
  - **Speckle noise**
  - **Viewpoint distortion**
  - **Low availability of annotated data**

- Class imbalance affects minority categories  
  → Future solution: weighted loss, oversampling, and synthetic generation

---

## 🔮 Future Work

- ✅ Explore attention mechanisms or ViT (Vision Transformers)
- ✅ Replace static thresholds with learnable parameters
- ✅ Convert model to ONNX / deploy via TensorRT or FPGA for real-time usage

---

## ▶️ Getting Started

### 🔧 Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

### 💻 Run the Notebook

```bash
git clone https://github.com/Deepayanbasu07/SAR_Target_Classification.git
cd SAR_Target_Classification
pip install -r requirements.txt
jupyter notebook SAR_Target_Classification_Using_Deep_Learning.ipynb
```

## 📎 References

- **[1]** Zheng, T., Wu, Q., & Yu, C. (2025).  
  *TDP-SAR: Task-Driven Pruning Method for SAR Target Recognition CNN Model*.  
  [Sensors, 25(10), 3117](https://www.mdpi.com/1424-8220/25/10/3117)

- **[2]** Zhang, Y., & Hao, Y. (2022).  
  *A Survey of SAR Image Target Detection Based on CNNs*.  
  [Remote Sensing, 14(24), 6240](https://www.mdpi.com/2072-4292/14/24/6240)

- **[3]** MSTAR Dataset (1996).  
  *Moving and Stationary Target Acquisition and Recognition Dataset*,  
  Air Force Research Laboratory (AFRL)
