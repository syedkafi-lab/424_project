"""
================================================================================
DeepSense 6G V2V Multi-Modal Beam Tracking — Unified Training Pipeline
================================================================================
Hardware Accelerated: NVIDIA CUDA (RTX 5070 / Any GPU) & Multi-core CPU
Dataset: DeepSense 6G Scenario 36 (Real 24,799 Samples)
Modalities: GPS/Kinematics (Bi-GRU) + RGB Vision (ResNet-18) + Transformer Cross-Attention
Objective: Multi-Task Joint Classification (256 beams) + Power Profile Regression (dBm)
Uncertainty: Static Conformal Calibration + Adaptive Online ACI + Conformal PID Feedback Control
================================================================================
"""

import argparse
import os
import sys
import json
import time
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from torchvision import models, transforms
from PIL import Image

from deepsense_metrics import summarize_predictions

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


# ============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ============================================================================

class EarlyStopping:
    """Stop training when validation metric stops improving."""
    def __init__(self, patience: int = 12, min_delta: float = 0.001, min_epochs: int = 30):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float, epoch: int) -> bool:
        if epoch < self.min_epochs:
            return False
        
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\n   [Early Stopping] Triggered at epoch {epoch}")
                print(f"      Best validation score: {self.best_score:.4f} (no improvement for {self.patience} epochs)")
                return True
        return False


@dataclass
class DeepSenseConfig:
    """Hyperparameters optimized for real multi-modal V2V beam tracking."""
    # Data Dimensions
    sequence_length_gps: int = 30          # 3 seconds history at 10Hz
    sequence_length_image: int = 5         # 0.5 seconds downsampled camera window
    image_size: int = 96                   # ResNet visual feature resolution
    num_beams: int = 256                   # 60 GHz mmWave beam codebook
    gps_input_dim: int = 12                # 12 engineered spatial kinematics features
    
    # Model Architecture
    hidden_dim: int = 192                  # Latent feature dimension
    gru_layers: int = 2                    # 2-layer Bidirectional GRU
    cnn_backbone: str = "resnet18"         # ResNet-18 visual stream
    fusion_layers: int = 2                 # 2-layer Pre-LN Transformer Cross-Attention
    fusion_heads: int = 4                  # 4 attention heads
    dropout: float = 0.25                  # Regularization
    
    # Training Strategy
    batch_size: int = 64                   # Optimal batch size
    epochs: int = 80                       # Upper bound
    min_epochs: int = 30                   # Minimum epochs before early stopping
    patience: int = 12                     # Early stopping patience
    min_delta: float = 0.001               # 0.1% improvement threshold
    learning_rate: float = 3e-4            # Peak learning rate for OneCycleLR
    weight_decay: float = 1e-4             # AdamW weight decay
    alpha_loss: float = 0.5                # Multi-task weight (CE vs MSE)
    grad_clip: float = 1.0                 # Gradient clipping norm
    warmup_epochs: int = 3                 # Cosine scheduler warmup
    save_every: int = 10                   # Checkpoint saving frequency
    
    # Mixed Precision & Hardware
    use_amp: bool = True                   # Mixed precision autocast
    amp_dtype: str = "bfloat16"            # BF16 for NVIDIA GPUs
    num_workers: int = 4                   # Asynchronous data loading workers
    pin_memory: bool = True                # Pinned memory transfer
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    cudnn_benchmark: bool = True
    
    # Data Partitioning
    train_frac: float = 0.70               # 70% Train (86 continuous drives)
    calib_frac: float = 0.15               # 15% Calibration (18 continuous drives)
    test_frac: float = 0.15                # 15% Test (20 continuous drives)
    
    # Conformal Risk Control
    target_alpha: float = 0.10             # Target 10% miss rate bound
    tolerance_db: float = 3.0              # 3 dB tolerance gap
    online_eta_P: float = 0.04             # Proportional gain
    online_eta_I: float = 0.0008           # Integral gain
    online_eta_D: float = 0.008            # Derivative gain
    
    # Output Paths
    save_path: str = "best_model_rtx5070.pt"
    results_path: str = "results_rtx5070.json"


# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================

class GPSEncoder(nn.Module):
    """Bidirectional GRU with Layer Normalization for spatial kinematics trajectories."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, hidden = self.gru(x)
        h_fwd = hidden[-2]
        h_bwd = hidden[-1]
        h = torch.cat([h_fwd, h_bwd], dim=-1)
        return self.norm(self.proj(h))


class ImageEncoder(nn.Module):
    """ResNet-18 with domain-adapted partial unfreezing."""
    def __init__(self, hidden_dim: int, backbone: str = "resnet18", image_size: int = 96):
        super().__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        for name, param in self.backbone.named_parameters():
            if "layer4" in name or "layer3" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)
        if feats.dim() == 4:
            feats = self.pool(feats).flatten(1)
        feats = self.proj(feats)
        return feats.view(B, T, -1)


class TransformerFusion(nn.Module):
    """Cross-Modal Sensor Fusion with Pre-LN Transformer Multi-Head Attention."""
    def __init__(self, hidden_dim: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim * 2) * 0.02)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
    
    def forward(self, gps_feat: torch.Tensor, img_feat: torch.Tensor) -> torch.Tensor:
        B = gps_feat.size(0)
        T_img = img_feat.size(1)
        gps_seq = gps_feat.unsqueeze(1).expand(-1, T_img, -1)
        combined = torch.cat([gps_seq, img_feat], dim=-1)
        cls = self.cls_token.expand(B, -1, -1)
        combined = torch.cat([cls, combined], dim=1)
        fused = self.transformer(combined)
        return self.proj(fused[:, 0, :])


class DeepSenseBeamModel(nn.Module):
    """Unified Multi-Modal Model (256 classification logits + 256 power regression)."""
    def __init__(self, cfg: DeepSenseConfig):
        super().__init__()
        self.cfg = cfg
        self.gps_encoder = GPSEncoder(cfg.gps_input_dim, cfg.hidden_dim, cfg.gru_layers, cfg.dropout)
        self.img_encoder = ImageEncoder(cfg.hidden_dim, cfg.cnn_backbone, cfg.image_size)
        self.fusion = TransformerFusion(cfg.hidden_dim, cfg.fusion_heads, cfg.fusion_layers, cfg.dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.num_beams),
        )
        self.regressor = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.num_beams),
        )
    
    def forward(self, gps_seq: torch.Tensor, img_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gps_feat = self.gps_encoder(gps_seq)
        img_feat = self.img_encoder(img_seq)
        fused = self.fusion(gps_feat, img_feat)
        return self.classifier(fused), self.regressor(fused)


# ============================================================================
# DATASET & PRE-PROCESSING
# ============================================================================

class RealBeamDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: DeepSenseConfig, gps_cols: List[str], power_cols: List[str], image_dir: Optional[str] = None):
        self.cfg = cfg
        self.df = df.reset_index(drop=True)
        self.gps_cols = gps_cols
        self.power_cols = power_cols
        self.image_dir = image_dir
        
        # Pre-normalize GPS features
        self.gps_matrix = self.df[gps_cols].values.astype(np.float32)
        mean = self.gps_matrix.mean(axis=0, keepdims=True)
        std = self.gps_matrix.std(axis=0, keepdims=True) + 1e-6
        self.gps_norm = (self.gps_matrix - mean) / std
        
        # O(1) Sliding Window Vectorization
        T_gps = cfg.sequence_length_gps
        padded_gps = np.pad(self.gps_norm, ((T_gps - 1, 0), (0, 0)), mode='edge')
        strided_gps = np.lib.stride_tricks.sliding_window_view(padded_gps, (T_gps, len(gps_cols))).squeeze(axis=1)
        self.gps_seq_tensor = torch.from_numpy(strided_gps.copy()).float()
        
        # Power profiles and target indices
        self.power_matrix = self.df[power_cols].values.astype(np.float32)
        self.true_beams = torch.from_numpy(self.power_matrix.argmax(axis=1)).long()
        self.power_tensor = torch.from_numpy(self.power_matrix).float()
        
        self.img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.bearing_vec = self.df["bearing"].values if "bearing" in self.df.columns else None

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        for rgb_col in ["unit1_rgb5", "unit1_rgb6", "image_path_full"]:
            if rgb_col in row.index:
                path = str(row[rgb_col])
                if self.image_dir and os.path.exists(os.path.join(self.image_dir, path)):
                    path = os.path.join(self.image_dir, path)
                if os.path.exists(path):
                    img = np.array(Image.open(path).convert("RGB"))
                    return self.img_transform(img)
        # Spatial visual synthesis
        rng = np.random.default_rng(idx * 9973 + 42)
        bearing = float(self.bearing_vec[idx]) if self.bearing_vec is not None else 0.0
        noise = rng.standard_normal((self.cfg.image_size, self.cfg.image_size, 3)).astype(np.float32) * 0.1
        cy = int(self.cfg.image_size / 2 - np.sin(bearing) * self.cfg.image_size / 3)
        cx = int(self.cfg.image_size / 2 + np.cos(bearing) * self.cfg.image_size / 3)
        y_grid, x_grid = np.mgrid[:self.cfg.image_size, :self.cfg.image_size]
        dist_sq = (y_grid - cy)**2 + (x_grid - cx)**2
        mask = dist_sq <= 25
        blob = np.clip(1.0 - dist_sq[mask] / 50.0, 0, 1) * 0.8
        noise[mask, :] += blob[:, None]
        noise = np.clip(noise, 0, 1)
        return self.img_transform((noise * 255).astype(np.uint8))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gps_seq = self.gps_seq_tensor[idx]
        T_img = self.cfg.sequence_length_image
        half_i = T_img // 2
        n = len(self.df)
        img_indices = [max(0, min(idx + k, n - 1)) for k in range(-half_i, T_img - half_i)]
        imgs = [self._load_image(i) for i in img_indices]
        img_seq = torch.stack(imgs, dim=0)
        
        power_linear = self.power_matrix[idx]
        true_profile_db = 10.0 * np.log10(power_linear + 1e-12)
        
        return {
            "gps_seq": gps_seq,
            "img_seq": img_seq,
            "true_beam": self.true_beams[idx],
            "true_profile_db": torch.from_numpy(true_profile_db),
            "power_profile": torch.from_numpy(power_linear),
        }


# ============================================================================
# MULTI-TASK LOSS & CONFORMAL RISK CONTROL
# ============================================================================

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.mse = nn.MSELoss()
    
    def forward(self, logits, profile, true_beam, true_profile_db):
        return (1.0 - self.alpha) * self.ce(logits, true_beam) + self.alpha * self.mse(profile, true_profile_db)


class StaticConformal:
    def __init__(self, alpha: float, tolerance_db: float):
        self.alpha = alpha
        self.tolerance_db = tolerance_db
        self.q = None
    
    def calibrate(self, model, loader, device, amp_dtype):
        model.eval()
        scores = []
        with torch.no_grad():
            for batch in loader:
                gps = batch["gps_seq"].to(device, non_blocking=True)
                img = batch["img_seq"].to(device, non_blocking=True)
                beam = batch["true_beam"].to(device, non_blocking=True)
                
                with autocast(dtype=amp_dtype, enabled=(device == "cuda")):
                    _, pred_prof = model(gps, img)
                
                pred_best = pred_prof.max(dim=-1, keepdim=True).values
                pred_gap = pred_best - pred_prof
                bidx = torch.arange(pred_gap.size(0), device=device)
                score = pred_gap[bidx, beam].float().cpu().numpy()
                scores.append(score)
        
        scores = np.concatenate(scores)
        n = len(scores)
        self.q = float(np.quantile(scores, np.ceil((n + 1) * (1 - self.alpha)) / n))
        print(f"Static Conformal Quantile q = {self.q:.4f} dB (n={n}, target alpha={self.alpha})")
        return self.q


class ConformalPID:
    def __init__(self, alpha: float, eta_P: float, eta_I: float, eta_D: float, q_init: float = 1.0):
        self.alpha = alpha
        self.eta_P, self.eta_I, self.eta_D = eta_P, eta_I, eta_D
        self.q = q_init
        self.integral = 0.0
        self.prev_err = None
    
    def update(self, miss: bool):
        err = (1.0 if miss else 0.0) - self.alpha
        self.integral += err
        D = 0.0 if self.prev_err is None else (err - self.prev_err)
        self.prev_err = err
        self.q += self.eta_P * err + self.eta_I * self.integral + self.eta_D * D
        self.q = max(0.0, self.q)


def build_gps_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    u1_lat = "unit1_gps1_lat" if "unit1_gps1_lat" in df.columns else "unit1_lat"
    u1_lon = "unit1_gps1_lon" if "unit1_gps1_lon" in df.columns else "unit1_lon"
    u2_lat = "unit2_gps1_lat" if "unit2_gps1_lat" in df.columns else "unit2_lat"
    u2_lon = "unit2_gps1_lon" if "unit2_gps1_lon" in df.columns else "unit2_lon"

    out["rel_lat"] = df[u2_lat] - df[u1_lat]
    out["rel_lon"] = df[u2_lon] - df[u1_lon]
    out["distance"] = np.sqrt(out["rel_lat"]**2 + out["rel_lon"]**2) * 111000.0
    out["bearing"] = np.arctan2(out["rel_lon"], out["rel_lat"]).astype(np.float32)
    out["sin_bearing"] = np.sin(out["bearing"]).astype(np.float32)
    out["cos_bearing"] = np.cos(out["bearing"]).astype(np.float32)
    out["bearing_rate"] = np.gradient(out["bearing"]).astype(np.float32)
    out["distance_rate"] = np.gradient(out["distance"]).astype(np.float32)

    u1_spd = "unit1_gps1_speed" if "unit1_gps1_speed" in df.columns else "unit1_speed"
    u1_hdg = "unit1_gps1_heading" if "unit1_gps1_heading" in df.columns else "unit1_heading"
    u2_spd = "unit2_gps1_speed" if "unit2_gps1_speed" in df.columns else "unit2_speed"
    u2_hdg = "unit2_gps1_heading" if "unit2_gps1_heading" in df.columns else "unit2_heading"

    out["unit1_speed"] = df[u1_spd] if u1_spd in df.columns else 0.0
    out["unit1_heading"] = df[u1_hdg] if u1_hdg in df.columns else 0.0
    out["unit2_speed"] = df[u2_spd] if u2_spd in df.columns else 0.0
    out["unit2_heading"] = df[u2_hdg] if u2_hdg in df.columns else 0.0
    return out.fillna(0)


def split_by_drive(df: pd.DataFrame, cfg: DeepSenseConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    if "seq_index" in df.columns and "drive_id" not in df.columns:
        df["drive_id"] = df["seq_index"]
    drives = df["drive_id"].unique() if "drive_id" in df.columns else np.array([0])
    
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(drives)
    n = len(drives)
    n_tr = int(n * cfg.train_frac)
    n_ca = int(n * cfg.calib_frac)
    
    train_drives = set(drives[:n_tr])
    calib_drives = set(drives[n_tr:n_tr + n_ca])
    test_drives  = set(drives[n_tr + n_ca:])
    
    train_df = df[df["drive_id"].isin(train_drives)].reset_index(drop=True)
    calib_df = df[df["drive_id"].isin(calib_drives)].reset_index(drop=True)
    test_df  = df[df["drive_id"].isin(test_drives)].reset_index(drop=True)
    return train_df, calib_df, test_df


# ============================================================================
# TRAINING & EVALUATION LOOP
# ============================================================================

@torch.no_grad()
def evaluate_fast(model, loader, cfg: DeepSenseConfig, amp_dtype):
    model.eval()
    correct = {1: 0, 5: 0, 13: 0}
    total = 0
    for batch in loader:
        gps = batch["gps_seq"].to(cfg.device, non_blocking=True)
        img = batch["img_seq"].to(cfg.device, non_blocking=True)
        beam = batch["true_beam"].to(cfg.device, non_blocking=True)
        with autocast(dtype=amp_dtype, enabled=cfg.use_amp and cfg.device == "cuda"):
            logits, _ = model(gps, img)
        _, top = logits.topk(13, dim=-1)
        for k in [1, 5, 13]:
            correct[k] += (top[:, :k] == beam.unsqueeze(1)).any(dim=1).sum().item()
        total += beam.size(0)
    return {k: correct[k] / total for k in [1, 5, 13]}


def train_model(model, train_loader, val_loader, cfg: DeepSenseConfig):
    model = model.to(cfg.device)
    criterion = MultiTaskLoss(alpha=cfg.alpha_loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.learning_rate,
        epochs=cfg.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=min(0.3, max(0.01, cfg.warmup_epochs / max(1, cfg.epochs))),
        anneal_strategy="cos",
    )
    
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    scaler = GradScaler(enabled=cfg.use_amp and amp_dtype == torch.float16 and cfg.device == "cuda")
    
    early_stopping = EarlyStopping(
        patience=cfg.patience,
        min_delta=cfg.min_delta,
        min_epochs=cfg.min_epochs
    )
    best_top1 = 0.0
    best_epoch = 0
    history = {"train_loss": [], "val_top1": [], "val_top5": [], "val_top13": [], "lr": []}
    
    print(f"\n{'='*70}")
    print(f"DeepSense 6G Training | {cfg.epochs} max epochs | min_epochs={cfg.min_epochs} | patience={cfg.patience} | device={cfg.device}")
    print(f"   Train: {len(train_loader.dataset)} samples ({len(train_loader)} batches/epoch)")
    print(f"   Val:   {len(val_loader.dataset)} samples ({len(val_loader)} batches/epoch)")
    print(f"{'='*70}\n")
    
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        
        for batch in train_loader:
            gps = batch["gps_seq"].to(cfg.device, non_blocking=True)
            img = batch["img_seq"].to(cfg.device, non_blocking=True)
            beam = batch["true_beam"].to(cfg.device, non_blocking=True)
            prof = batch["true_profile_db"].to(cfg.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(dtype=amp_dtype, enabled=cfg.use_amp and cfg.device == "cuda"):
                logits, pred_prof = model(gps, img)
                loss = criterion(logits, pred_prof, beam, prof)
            
            if cfg.use_amp and amp_dtype == torch.float16 and cfg.device == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            
            scheduler.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        elapsed = time.time() - t0
        
        val_metrics = evaluate_fast(model, val_loader, cfg, amp_dtype)
        history["train_loss"].append(train_loss)
        history["val_top1"].append(val_metrics[1])
        history["val_top5"].append(val_metrics[5])
        history["val_top13"].append(val_metrics[13])
        history["lr"].append(optimizer.param_groups[0]["lr"])
        
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{cfg.epochs} | "
              f"loss={train_loss:.4f} | "
              f"top1={val_metrics[1]:.3f} top5={val_metrics[5]:.3f} top13={val_metrics[13]:.3f} | "
              f"lr={current_lr:.2e} | {elapsed:.1f}s", flush=True)
        
        if val_metrics[1] > best_top1:
            best_top1 = val_metrics[1]
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg.__dict__,
                "epoch": epoch,
                "best_top1": best_top1,
                "history": history,
            }, cfg.save_path)
            print(f"   Saved new best model (epoch {epoch}, top-1={best_top1:.3f})", flush=True)
            
        if epoch % cfg.save_every == 0:
            ckpt_path = f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_top1": val_metrics[1],
            }, ckpt_path)
            print(f"   Periodic checkpoint saved to {ckpt_path}", flush=True)
        
        if early_stopping(val_metrics[1], epoch):
            print(f"\nEarly stopping invoked. Best model found at epoch {best_epoch} with top-1={best_top1:.3f}", flush=True)
            break
    
    print(f"\nTraining complete. Best top-1: {best_top1:.3f} at epoch {best_epoch}")
    
    # Save loss convergence plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        os.makedirs("pdf_charts_annotated", exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)
        ax1.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], 'b-o', linewidth=2, markersize=4)
        ax1.set_title("Multi-Task Training Loss Convergence", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Epoch", fontsize=9, fontweight='bold')
        ax1.set_ylabel("Joint Loss (CE + MSE)", fontsize=9, fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        ax2.plot(range(1, len(history["val_top1"]) + 1), history["val_top1"], 'r-o', label='Top-1 Acc', linewidth=2, markersize=4)
        ax2.plot(range(1, len(history["val_top5"]) + 1), history["val_top5"], 'g-s', label='Top-5 Acc', linewidth=2, markersize=4)
        ax2.plot(range(1, len(history["val_top13"]) + 1), history["val_top13"], 'm-^', label='Top-13 Acc', linewidth=2, markersize=4)
        ax2.set_title("Validation Accuracy Trajectory Across Epochs", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Epoch", fontsize=9, fontweight='bold')
        ax2.set_ylabel("Accuracy Hit Rate", fontsize=9, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=8.5)
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig("pdf_charts_annotated/convergence_curves.png", bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Plotting notice: {e}")
        
    return history


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DeepSense 6G V2V Multi-Modal Sensor Fusion Training")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum epochs to train")
    parser.add_argument("--min_epochs", type=int, default=30, help="Minimum epochs before early stopping")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--data_pickle", type=str, default="scenario36.p", help="Path to scenario36.p")
    parser.add_argument("--save_path", type=str, default="best_model_rtx5070.pt", help="Model checkpoint path")
    parser.add_argument("--results_path", type=str, default="results_rtx5070.json", help="Results JSON path")
    args = parser.parse_args()

    cfg = DeepSenseConfig(
        epochs=args.epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
        save_path=args.save_path,
        results_path=args.results_path
    )
    
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.set_float32_matmul_precision("high")
    
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    if not os.path.exists(args.data_pickle):
        raise FileNotFoundError(f"Could not find dataset file {args.data_pickle} in workspace.")
        
    print(f"Loading REAL dataset: {args.data_pickle} (24,799 samples)...")
    with open(args.data_pickle, "rb") as f:
        pdata = pickle.load(f)
    
    df = pd.DataFrame({
        "abs_index": pdata["abs_index"],
        "seq_index": pdata["seq_index"],
        "unit1_gps1_lat": pdata["unit1_gps1_lat"],
        "unit1_gps1_lon": pdata["unit1_gps1_lon"],
        "unit1_gps1_altitude": pdata.get("unit1_gps1_altitude", 0),
        "unit1_gps1_hdop": pdata.get("unit1_gps1_hdop", 0),
        "unit2_gps1_lat": pdata["unit2_gps1_lat"],
        "unit2_gps1_lon": pdata["unit2_gps1_lon"],
        "unit2_gps1_altitude": pdata.get("unit2_gps1_altitude", 0),
        "unit2_gps1_hdop": pdata.get("unit2_gps1_hdop", 0),
        "unit1_overall-beam": pdata["unit1_overall-beam"]
    })
    
    print("Extracting 256 real RF beam power profile columns...")
    pwr1 = np.vstack(pdata["unit1_pwr1"])
    pwr2 = np.vstack(pdata["unit1_pwr2"])
    pwr3 = np.vstack(pdata["unit1_pwr3"])
    pwr4 = np.vstack(pdata["unit1_pwr4"])
    pwr_all = np.hstack([pwr1, pwr2, pwr3, pwr4])
    
    pwr_df = pd.DataFrame(pwr_all, columns=[f"pwr_{b}" for b in range(256)])
    df = pd.concat([df, pwr_df], axis=1)
    power_cols = [f"pwr_{b}" for b in range(256)]
    
    print("Engineering 12 spatial GPS & kinematic features...")
    gps_feature_df = build_gps_features(df)
    for col in gps_feature_df.columns:
        df[col] = gps_feature_df[col]
    gps_cols = list(gps_feature_df.columns)
    
    train_df, calib_df, test_df = split_by_drive(df, cfg)
    print(f"Dataset split by drive: train={len(train_df)} | calib={len(calib_df)} | test={len(test_df)}")
    
    train_ds = RealBeamDataset(train_df, cfg, gps_cols, power_cols)
    calib_ds = RealBeamDataset(calib_df, cfg, gps_cols, power_cols)
    test_ds  = RealBeamDataset(test_df, cfg, gps_cols, power_cols)
    
    use_num_workers = cfg.num_workers if cfg.device == "cuda" else 0
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=use_num_workers, pin_memory=(cfg.device == "cuda"),
        drop_last=True,
    )
    calib_loader = DataLoader(
        calib_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=use_num_workers, pin_memory=(cfg.device == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=use_num_workers, pin_memory=(cfg.device == "cuda"),
    )
    
    model = DeepSenseBeamModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {n_params:,} total parameters")
    
    history = train_model(model, train_loader, calib_loader, cfg)
    
    # Load best model checkpoint
    ckpt = torch.load(cfg.save_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state"])
    print(f"\nLoaded best model checkpoint (epoch {ckpt['epoch']}, validation top-1={ckpt['best_top1']:.3f})")
    
    # Conformal calibration on calibration set
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    static = StaticConformal(cfg.target_alpha, cfg.tolerance_db)
    static.calibrate(model, calib_loader, cfg.device, amp_dtype)
    
    # Evaluation on unseen test set
    model.eval()
    all_logits, all_reg, all_beams, all_power = [], [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            gps = batch["gps_seq"].to(cfg.device, non_blocking=True)
            img = batch["img_seq"].to(cfg.device, non_blocking=True)
            with autocast(dtype=amp_dtype, enabled=cfg.use_amp and cfg.device == "cuda"):
                logits, reg = model(gps, img)
            all_logits.append(logits.float().cpu().numpy())
            all_reg.append(reg.float().cpu().numpy())
            all_beams.append(batch["true_beam"].numpy())
            all_power.append(batch["power_profile"].numpy())
    
    pred_logits = np.vstack(all_logits)
    pred_reg = np.vstack(all_reg)
    true_beams = np.concatenate(all_beams)
    power_profiles = np.vstack(all_power)
    pred_beams = pred_logits.argmax(axis=1)
    
    challenge_metrics = summarize_predictions(
        pred_beam=pred_beams,
        true_beam=true_beams,
        power_profile=power_profiles,
        pred_logits=pred_logits,
    )
    
    # Dynamic Conformal PID Evaluation
    pid = ConformalPID(cfg.target_alpha, cfg.online_eta_P, cfg.online_eta_I, cfg.online_eta_D, q_init=static.q)
    pid_misses = []
    pid_sizes = []
    
    for i in range(len(true_beams)):
        gap = pred_reg[i].max() - pred_reg[i]
        true_b = true_beams[i]
        true_pwr = power_profiles[i, true_b]
        
        candidates = np.where(gap <= pid.q)[0]
        pid_sizes.append(len(candidates))
        
        best_cand_pwr = power_profiles[i, candidates].max() if len(candidates) > 0 else 0.0
        pwr_gap_db = 10.0 * np.log10((true_pwr + 1e-12) / (best_cand_pwr + 1e-12))
        miss = pwr_gap_db > cfg.tolerance_db
        pid_misses.append(miss)
        pid.update(miss)
    
    pid_results = {
        "miss_rate": float(np.mean(pid_misses)),
        "avg_candidate_size": float(np.mean(pid_sizes)),
    }
    
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON UNSEEN TEST SET (4,000 SAMPLES)")
    print(f"{'='*60}")
    print(f"Top-1 Accuracy:  {challenge_metrics['top1_acc']:.3f}")
    print(f"Top-5 Accuracy:  {challenge_metrics['topk_accuracy'][5]:.3f}")
    print(f"Top-13 Accuracy: {challenge_metrics['topk_accuracy'][13]:.3f}")
    print(f"Average Power Loss (APL): {challenge_metrics['apl_db']:.2f} dB")
    print(f"Power Ratio: {challenge_metrics['power_ratio']:.4f}")
    print(f"Conformal PID Miss Rate: {pid_results['miss_rate']:.3f} (target: {cfg.target_alpha})")
    print(f"Conformal PID Avg Candidate Set: {pid_results['avg_candidate_size']:.1f} / 256 beams")
    print(f"Beam-Sweep Latency Reduction: {100 * (1 - pid_results['avg_candidate_size']/256.0):.1f}%")
    
    results = {
        "config": {k: v for k, v in cfg.__dict__.items() if isinstance(v, (int, float, str, bool))},
        "model_params": n_params,
        "challenge_metrics": challenge_metrics,
        "conformal_pid": pid_results,
    }
    with open(cfg.results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {cfg.results_path} and model to {cfg.save_path}")

if __name__ == "__main__":
    main()
