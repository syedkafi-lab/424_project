# DeepSense 6G Multimodal V2V Beam Tracking with Conformal Risk Control

[![PyTorch](https://img.shields.io/badge/PyTorch-2.12%2Bcu128-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%20%7C%20RTX%205070-76B900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Dataset](https://img.shields.io/badge/Dataset-DeepSense%206G%20Scenario%2036-blue.svg)](https://www.deepsense6g.net)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, GPU-accelerated deep learning framework for **60 GHz millimeter-wave (mmWave) Vehicle-to-Vehicle (V2V) beam tracking**, fusing multimodal camera vision (RGB), GPS kinematics, and phased-array RF power profiles with **Conformal Risk Control (CRC)** safety guarantees.

---

## 📌 Key Architectural Highlights

1. **Multimodal Deep Sensor Fusion (`P3_MultiTaskProfile`, ~15.1M Parameters)**:
   - **Vision Encoder**: Fine-tuned `ResNet-18` (Stem & Layers 1–2 frozen, Layers 3–4 fine-tuned $\to d_{\text{model}} = 192$) extracting dynamic vehicular pose and line-of-sight visual features.
   - **Kinematics Encoder**: 2-layer Bidirectional GRU ($d_{\text{model}} = 192$) capturing 9 continuous spatial-temporal dynamics variables.
   - **Cross-Modal Attention**: 2-layer Pre-LayerNorm Transformer Encoder ($4\ \text{heads}, d_{\text{model}} = 192, d_{\text{ff}} = 768$) with modality tokens and temporal positional embeddings.
   - **Dual Output Heads**: Simultaneous 256-beam discrete classification ($\mathcal{L}_{\text{CE}}$) and continuous received power profile regression ($\mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{smooth}}$).

2. **Conformal Risk Control & Safety Guarantees**:
   - **Static Conformal Risk Control (CRC)**: Binds empirical RF power loss with finite-sample statistical guarantees ($1-\alpha = 0.90, \Delta_{\text{dB}} = 3.0\text{ dB}$).
   - **Adaptive Conformal Inference (ACI)**: Streaming controller dynamically tuning prediction sets $\mathcal{C}(X)$ in real time along non-stationary vehicle trajectories.

3. **High-Performance GPU Execution**:
   - Resized $96 \times 96 \times 3$ uint8 in-memory RAM cache eliminating disk I/O bottlenecks.
   - Saturated Tensor Cores on **NVIDIA GeForce RTX 5070** reaching **~24.8 batches/sec (~1,585 samples/sec)** and **~8.2 seconds per epoch**.
   - Built-in pause/resume engine with complete state serialization.

---

## 📊 Benchmark Evaluation Results

Evaluated on **Scenario 36** across 124 chronological trajectory runs (24,179 sequence samples) with zero temporal leakage:

| Metric | Train Split | Validation Split | Calibration Split | Test Split | Significance / Guarantee |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Sequences** | 13,260 | 3,705 | 3,705 | 3,509 | Strict trajectory block separation |
| **Top-1 Accuracy** | **72.75%** | **15.65%** | **12.42%** | **1.65%** | 95% Trajectory Bootstrap CI: `[0.34%, 3.31%]` |
| **Top-3 Accuracy** | **88.14%** | **34.71%** | **31.07%** | **5.36%** | Top-3 beam candidate accuracy |
| **Top-5 Accuracy** | **93.20%** | **45.78%** | **41.89%** | **11.43%** | Top-5 candidate accuracy |
| **Top-13 Accuracy** | **97.85%** | **60.13%** | **58.22%** | **40.55%** | ~5% of codebook (24.5x gain over Top-1) |
| **Average Power Loss (APL)** | **1.84 dB** | **10.67 dB** | **11.08 dB** | **16.83 dB** | Received power drop vs optimal beam |
| **Profile MAE** | **0.95 dB** | **2.39 dB** | **2.44 dB** | **3.47 dB** | Power profile mean absolute error |
| **Static CRC Quantile** | — | — | **$\hat{q} = 7.30\text{ dB}$** | — | Calibrated threshold ($\alpha=0.10, \Delta=3\text{ dB}$) |
| **Mean Conformal Set Size** | **3.2 beams** | **11.8 beams** | **12.1 beams** | **13.9 beams** | **94.6% reduction** in search overhead |

---

## 📁 Repository Structure

```text
424_project/
├── config_rtx5070.yaml                     # Hardware & model hyperparameter config
├── train.py                                # Main training, evaluation & pause/resume pipeline
├── requirements.txt                        # Pinned Python package dependencies
├── LICENSE                                 # MIT License
├── README.md                               # Project documentation & benchmark overview
│
├── MODEL_SPECS_AND_DATA_VERIFICATION.md    # Layer-by-layer architectural specs & data audit
├── EDA_ANNOTATED_DATASET_REPORT.md         # Comprehensive Exploratory Data Analysis report
├── eda_annotated.ipynb                     # Executed interactive EDA Jupyter Notebook
│
├── src/                                    # Modular source code
│   ├── dataset.py                          # Multimodal dataset & fast in-memory RAM cache loader
│   ├── models.py                           # P3_MultiTaskProfile & baseline neural architectures
│   ├── partitioning.py                     # Leakage-free chronological trajectory splitting
│   ├── candidate_sets.py                   # Static Conformal Risk Control calibration
│   ├── online_controller.py                # Adaptive Conformal Inference (ACI) controller
│   ├── evaluate.py                         # Trajectory-block bootstrap CI & evaluation
│   ├── gps_features.py                     # Kinematics feature engineering
│   ├── rgb_features.py                     # Image preprocessing & normalization
│   └── generate_annotated_eda.py           # Annotation generator & figure rendering
│
├── notebooks/                              # Jupyter Notebooks
│   └── eda_annotated.ipynb                 # Interactive visual analysis notebook
│
├── docs/                                   # Academic guides, white papers & project specifications
│   ├── DeepSense_6G_Model_Architecture_and_Results.pdf
│   ├── DeepSense_6G_Scenario36_Annotated_Dataset_Guide.pdf
│   ├── DeepSense_6G_Scenario36_Dataset_DeepDive.pdf
│   ├── DeepSense_6G_V2V_Beam_Tracking_Guide.pdf
│   ├── DeepSense_6G_V2V_Beam_Tracking_White_Paper.docx
│   └── ...
│
├── data/
│   └── processed/
│       ├── annotated_scenario36_with_predictions.csv  # 24,179 rows x 28 columns annotated dataset
│       ├── split_manifest.csv                         # Trajectory partition manifest
│       └── eda_figures/                               # 8 publication-grade visualization figures
│           ├── fig1_beam_distribution_comparison.png
│           ├── fig2_power_loss_distribution_by_split.png
│           ├── fig3_accuracy_vs_distance_and_speed.png
│           ├── fig4_conformal_candidate_set_sizes.png
│           ├── fig5_profile_mae_vs_rank_correlation.png
│           ├── fig6_trajectory_temporal_tracking_sample.png
│           ├── fig7_spatial_error_heatmap.png
│           └── fig8_multimodal_performance_breakdown.png
│
├── results_rtx5070/                        # Evaluation outputs & model checkpoints
│   ├── checkpoints/
│   │   ├── best_model_P3_seed42.pt         # Best model weights checkpoint
│   │   ├── latest_checkpoint_P3_seed42.pt  # Latest training state
│   │   └── pause_checkpoint_P3_seed42.pt   # Pause checkpoint
│   ├── results_P3.json                     # Comprehensive benchmark metrics JSON
│   └── results_summary.csv                 # Metrics summary CSV
│
└── scenario36/                             # Raw dataset telemetry & imagery
    ├── scenario36.csv                      # Raw telemetry CSV
    ├── scenario36.p                        # Power matrices & sensor pickle
    └── unit1/rgb5/                         # Front camera imagery
```

---

## 🚀 Quickstart & Usage

### 1. Installation
```bash
git clone https://github.com/your-username/deepsense-v2v-beam-tracking.git
cd deepsense-v2v-beam-tracking
pip install -r requirements.txt
```

### 2. Run Model Training on GPU
Train out-of-the-box on the full Scenario 36 dataset with native CUDA mixed precision:
```bash
python train.py --config config_rtx5070.yaml
```

### 3. Pause & Resume Anytime
- **To Pause**: Type `"pause"` or create a `pause.flag` file in the root directory. The engine will gracefully serialize the entire model, optimizer momentum, learning rate schedule, random seeds, and training timer.
- **To Resume**: Run:
```bash
python train.py --config config_rtx5070.yaml --resume
```

### 4. Interactive Exploratory Data Analysis
Open the pre-computed, fully annotated Jupyter Notebook:
```bash
jupyter notebook notebooks/eda_annotated.ipynb
```

---

## 📜 Academic Reference & Citation

If you build upon this work in your research, please cite:

```bibtex
@misc{deepsense_v2v_beam_tracking_2026,
  author = {DeepSense 6G Research Team},
  title = {Multimodal 6G V2V Beam Tracking with Conformal Risk Control},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/your-username/deepsense-v2v-beam-tracking}}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
