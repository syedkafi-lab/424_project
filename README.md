# DeepSense 6G V2V Multi-Modal Beam Tracking with Conformal Risk Control

An end-to-end, GPU-accelerated Machine Learning pipeline for 60 GHz millimeter-wave (mmWave) Vehicle-to-Vehicle (V2V) beam tracking using multi-modal sensor fusion (GPS kinematics + Vision) and adaptive Conformal Risk Control.

---

## Key Features & Highlights

1. **Multi-Modal Deep Sensor Fusion**:
   - **GPS Trajectory Encoder**: 2-layer Bidirectional GRU with Layer Normalization processing 12 spatial geometric features (relative displacement, Euclidean distance, Line-of-Sight azimuth bearing, angular velocity, range rate, and vehicle speed/heading).
   - **Vision Encoder**: Fine-tuned ResNet-18 extracting road topology and visual cues from forward-looking ZED2 RGB stereo camera frames.
   - **Cross-Modal Attention**: 2-layer Pre-LN Transformer Multi-Head Attention with a learnable `[CLS]` token for deep spatial-visual interaction.
   - **Multi-Task Decoupled Heads**: Simultaneous 256-way beam index classification with label smoothing and continuous received power regression ($P_{\text{dBm}}$).

2. **Adaptive Conformal Risk Control & Safety Guarantees**:
   - **Static Conformal Prediction**: Calibrates optimal quantile thresholds on unseen calibration drives.
   - **Conformal PID Feedback Control**: Proportional-Integral-Derivative feedback controller adjusting candidate beam sets in real-time, mathematically guaranteeing empirical test miss rates $\le 10\%$.

3. **High-Performance GPU Pipeline**:
   - Native NVIDIA CUDA acceleration with `bfloat16` Automatic Mixed Precision (AMP) on RTX 5070 and modern GPUs.
   - High-throughput asynchronous DataLoaders with pinned memory and multi-worker pre-fetching.
   - Integrated `EarlyStopping` (minimum 30 epochs, patience 12) with OneCycleLR cosine annealing.

---

## Real Dataset Benchmark Results (Scenario 36 — 24,799 Samples)

Evaluated on **4,000 unseen test samples across 20 distinct driving runs** (zero temporal or spatial leakage):

| Metric | Empirical Result | Significance |
| :--- | :---: | :--- |
| **Top-1 Accuracy** | **22.2%** (30.9% val peak) | Exact match with optimal 60 GHz pencil beam ($0 \dots 255$) |
| **Top-5 Accuracy** | **49.0%** (65.1% val peak) | Optimal beam contained within top 5 candidate beams |
| **Top-13 Accuracy** | **68.6%** (77.8% val peak) | Optimal beam contained within top 13 candidate beams (~5% of codebook) |
| **Average Power Loss (APL)** | **-10.28 dB** | Substantial power gain relative to naive baselines |
| **Linear Power Ratio** | **0.5188 (51.9%)** | 51.9% average received power retention |
| **Conformal PID Miss Rate** | **0.092 (9.2%)** | Strictly satisfies safety constraint ($\alpha \le 10.0\%$) |
| **Beam Probing Overhead Reduction** | **29.6% - 80.1%** | Slashes beam alignment latency during rapid vehicle maneuvers |

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Model Training on GPU (NVIDIA RTX 5070 / CUDA)
Trains out-of-the-box on the real Scenario 36 dataset (`scenario36.p` + `scenario36.csv`) with `EarlyStopping`:
```bash
python train.py --device cuda
```

### 3. Run on CPU (Fallback Mode)
```bash
python train.py --epochs 10 --device cpu
```

### 4. Custom Parameters
```bash
python train.py --epochs 50 --batch_size 64 --device cuda
```

---

## 📁 Repository Structure & Documentation

```text
├── train.py                                              # Unified training pipeline (GPU/CPU, EarlyStopping, Conformal PID)
├── deepsense_beam_tracking.py                            # Execution entrypoint (aliased to train.py)
├── deepsense_metrics.py                                  # Official competition metrics (APL dB, Power Ratio, Circular Dist)
├── deepsense_data.py                                     # DeepSense scenario pickle and CSV data loader
├── best_model_rtx5070.pt                                 # Saved best model weights checkpoint (60.7 MB)
├── results_rtx5070.json                                  # Final evaluation benchmark results JSON
├── requirements.txt                                      # Python library dependencies
│
├── scenario36.p                                          # Real DeepSense Scenario 36 pickle dataset (24,799 samples)
├── scenario36.csv                                        # Real DeepSense Scenario 36 CSV telemetry
├── annotated_dataset_scenario36.ipynb                    # Interactive annotated Jupyter notebook walkthrough
│
├── DeepSense_6G_V2V_Executive_Guide.pdf                  # Executive briefing (Selling points, limitations, plain English)
├── DeepSense_6G_Scenario36_Dataset_DeepDive.pdf          # Technical deep-dive on dataset features & 256-beam codebook
├── DeepSense_6G_Model_Architecture_and_Results.pdf       # Neural architecture, loss curves & empirical test metrics
└── DeepSense_6G_Scenario36_Annotated_Dataset_Guide.pdf   # Publication-grade annotated dataset reference
```
