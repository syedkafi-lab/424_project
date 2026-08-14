# Complete Project Guide & Faculty Presentation Instructions

This document provides a step-by-step guide for setting up, running, zipping, and presenting the **DeepSense 6G V2V Beam Tracking with Conformal Risk Control** project.

---

## Part 1: How to Setup & Run on Any Machine / IDE

This codebase is cross-platform and works on **Windows**, **macOS**, **Linux**, **Google Colab**, **VS Code**, **PyCharm**, **Jupyter Notebook**, or command-line terminal.

### 1. Prerequisites
- **Python**: 3.9 or higher.
- **Hardware**: Automatically detects CPU, NVIDIA GPU (CUDA), or Apple Silicon (MPS). No manual GPU configuration required.

### 2. Installation
Open your IDE terminal or Command Prompt inside the project directory and run:
```bash
pip install -r requirements.txt
```

### 3. Execution Commands

#### Option A: Quick Demonstration Mode (Synthetic Scenario 36 - No Data Download Required)
Runs full model training (10 epochs), evaluation, and conformal prediction out-of-the-box:
```bash
python deepsense_beam_tracking.py --mode synthetic --epochs 10 --device auto
```

#### Option B: Official DeepSense Dataset Mode
If you have downloaded the official `scenario36.p` pickle file and `trainset.csv`:
```bash
python deepsense_beam_tracking.py --mode deepsense --csv /path/to/trainset.csv --pickle_dir /path/to/pickle_dir --scenarios 36 --epochs 30 --device auto
```

#### Option C: Specifying Hardware Manually
- Force CPU execution: `--device cpu`
- Force CUDA GPU execution: `--device cuda`

---

## Part 2: Sharing & Zipping Checklist

When sharing this project with faculty, project evaluators, or team members, include the following files in your `.zip` archive:

```text
424_project/
├── deepsense_beam_tracking.py   # Main pipeline (Data loader, GRU+CNN+Fusion Model, Conformal PID)
├── deepsense_data.py           # DeepSense V2V dataset parsing utilities
├── deepsense_metrics.py        # Official competition metrics (APL, Power Ratio, Circular Distance)
├── requirements.txt            # Python dependencies (torch, torchvision, pandas, scipy, tqdm)
├── README.md                   # Project summary documentation
├── INSTRUCTIONS.md             # Complete execution & presentation guide (This file)
├── results.json                # Benchmark output metrics
└── best_model.pt               # Saved trained model checkpoint weights
```

---

## Part 3: Faculty Presentation Guide & Defense Script

### Presentation Structure (5-Slide / 5-Minute Outline)

#### Slide 1: The Problem (5G/6G V2V mmWave Beam Management)
* **Context**: High-frequency 60 GHz mmWave communication requires narrow directional beams to establish high-throughput V2V links.
* **The Bottleneck**: Traditional systems perform exhaustive sweeps across all **256 beam directions**. During high-speed vehicle movement, full 256-beam sweeps cause significant latency and link dropouts.

#### Slide 2: Proposed Multi-Modal Deep Learning Architecture
* **GPS Trajectory Encoder**: 2-layer GRU processing a sequence of 12 spatial GPS features (relative position, bearing angle, velocity, distance).
* **Vision Encoder**: ResNet-18 CNN extracting visual context from front-facing camera frames.
* **Sensor Fusion**: Cross-modal Transformer Multi-Head Attention layer fusing spatial and visual embeddings.
* **Multi-Task Heads**: Simultaneous classification (256 beam logits) and power profile regression.

#### Slide 3: Novelty — Adaptive Conformal Risk Control & PID Feedback
* **Why Standard ML Falls Short**: Standard deep learning outputs point predictions without uncertainty guarantees, causing link failure under out-of-distribution dynamic conditions.
* **Our Solution**: Integrated **Conformal Risk Control** (Static, Gibbs & Candès Adaptive ACI, and Conformal PID Feedback Control):
  * Dynamically sizes candidate beam search sets per sample.
  * Rigorously bounds the beam miss-rate to stay below $\alpha = 10\%$.

#### Slide 4: Experimental Results & Performance Impact
Present quantitative results (from `results.json`):

| Performance Metric | Result | Description |
| :--- | :---: | :--- |
| **Top-13 Accuracy** | **90.8%** | True beam within top 13 candidates |
| **Average Received Power Ratio** | **81.0%** | Power captured vs maximum optimal beam |
| **Average Power Loss (APL)** | **-1.20 dB** | Wasted power gap |
| **Conformal PID Miss-Rate** | **9.6%** | Satisfies target risk bound ($\le 10\%$) |
| **Average Probed Beams** | **2.0 / 256** | Probes only 2.0 beams instead of 256 |
| **Beam Sweep Latency Reduction** | **99.2%** | **99.2% reduction in beam search space** |

#### Slide 5: Conclusion & Future Work
* **Conclusion**: Fusing GPS trajectory sequences with vision via Transformer attention and Conformal PID control cuts beam-sweep overhead by **99.2%** while guaranteeing a $\le 10\%$ miss rate.
* **Future Extension**: Deploying lightweight quantized ONNX models on onboard edge processing hardware (NVIDIA Jetson / automotive ECUs).

---

### 30-Second Verbal Presentation Script
> *"Good morning professors. Our project solves the beam tracking latency bottleneck in 5G/6G Vehicle-to-Vehicle mmWave communications. Instead of performing slow, exhaustive sweeps across 256 beam directions, we designed a multi-modal neural network that fuses GPS trajectory sequences and visual camera frames via Transformer attention. To guarantee reliability under dynamic mobility, we integrated Adaptive Conformal PID Feedback Control, which dynamically adjusts search candidate set sizes while maintaining a strict 10% miss-rate bound. In our evaluation on DeepSense Scenario 36 benchmarks, our system achieved a **90.8% Top-13 accuracy** and reduced the beam search overhead by **99.2%**—probing on average **only 2.0 beams out of 256**."*

---

### Expected Faculty Defense Q&A

**Q1: Why did you fuse vision with GPS instead of using GPS alone?**
* **Answer**: GPS provides macroscopic global position and bearing, but suffers from multipath reflections, non-line-of-sight (NLOS) blockages, and sensor noise. Vision provides real-time local geometry (line-of-sight occlusion, vehicle orientation). Fusing both via cross-attention yields higher beam accuracy than either modality alone.

**Q2: What is Conformal Prediction and why is PID feedback control used?**
* **Answer**: Standard neural networks produce point predictions without reliable confidence bounds. Conformal prediction constructs predictive sets with distribution-free statistical coverage guarantees. We use Conformal PID Control because vehicle motion creates non-stationary temporal dynamics; the PID feedback loop adjusts the threshold $q$ online to maintain the 10% miss-rate target even as environmental conditions change.

**Q3: How does your code run if the evaluator doesn't have GPUs or the official dataset?**
* **Answer**: The codebase features an automated fallback mechanism (`--mode synthetic`). It generates synthetic realistic V2V beam tracking patterns out-of-the-box and uses `--device auto` to run seamlessly on CPU, CUDA, or Apple Silicon without hardware configuration.
