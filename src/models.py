import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

N_BEAMS = 256
N_GPS_FEATS = 9

class ResNet18VisionEncoder(nn.Module):
    """
    ResNet-18 Vision Encoder with layers 1-2 frozen and layers 3-4 trainable.
    Projects 5-frame sequence to (B, 5, d_model).
    """
    def __init__(self, d_model=192, freeze_until="layer3"):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT
            base = models.resnet18(weights=weights)
        except Exception:
            base = models.resnet18(weights=None)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.proj = nn.Linear(512, d_model)

        if freeze_until == "layer3":
            for layer in [self.conv1, self.bn1, self.layer1, self.layer2]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        # x: (B, seq_len=5, 3, H, W)
        B, S, C, H, W = x.shape
        x_flat = x.view(B * S, C, H, W)

        x_feat = self.conv1(x_flat)
        x_feat = self.bn1(x_feat)
        x_feat = self.relu(x_feat)
        x_feat = self.maxpool(x_feat)

        x_feat = self.layer1(x_feat)
        x_feat = self.layer2(x_feat)
        x_feat = self.layer3(x_feat)
        x_feat = self.layer4(x_feat)
        x_feat = self.avgpool(x_feat).flatten(1)  # (B*S, 512)

        x_out = self.proj(x_feat)  # (B*S, d_model)
        return x_out.view(B, S, -1)  # (B, S, d_model)


class BiGRUPositionEncoder(nn.Module):
    """
    2-Layer Bidirectional GRU Encoder for GPS Motion dynamics.
    Projects 5-step GPS sequence to (B, 5, d_model).
    """
    def __init__(self, in_dim=N_GPS_FEATS, d_model=192, num_layers=2, dropout=0.15):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        # x: (B, seq_len=5, in_dim=9)
        out, _ = self.gru(x)  # (B, seq_len, d_model * 2)
        return self.proj(out)  # (B, seq_len, d_model)


class PreLNTransformerFusion(nn.Module):
    """
    Pre-LN Transformer Cross-Modal Fusion Network (4 Heads, 2 Layers).
    Combines temporal GPS tokens and visual frame tokens.
    """
    def __init__(self, d_model=192, nhead=4, num_layers=2, dim_feedforward=768, dropout=0.15):
        super().__init__()
        self.modality_gps = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.modality_rgb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 10, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=True,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, rgb_tokens, gps_tokens):
        # rgb_tokens: (B, 5, d_model), gps_tokens: (B, 5, d_model)
        B = rgb_tokens.size(0)
        gps_tok = gps_tokens + self.modality_gps
        rgb_tok = rgb_tokens + self.modality_rgb

        seq = torch.cat([gps_tok, rgb_tok], dim=1)  # (B, 10, d_model)
        seq = seq + self.pos_embed

        fused_seq = self.transformer(seq)  # (B, 10, d_model)
        fused_seq = self.norm(fused_seq)

        # Global temporal mean pooling
        return torch.mean(fused_seq, dim=1)  # (B, d_model)


class P3_MultiTaskProfile(nn.Module):
    """
    Optimized ~15M Parameter Multi-Task Model for RTX 5070 (12GB VRAM).
    ResNet-18 + 2-layer BiGRU + Pre-LN Transformer Fusion + Dual Heads.
    """
    def __init__(
        self,
        d_model=192,
        fusion_heads=4,
        fusion_layers=2,
        freeze_until="layer3",
        dropout=0.15,
        n_beams=N_BEAMS
    ):
        super().__init__()
        self.rgb_encoder = ResNet18VisionEncoder(d_model=d_model, freeze_until=freeze_until)
        self.gps_encoder = BiGRUPositionEncoder(d_model=d_model, num_layers=2, dropout=dropout)
        self.fusion = PreLNTransformerFusion(
            d_model=d_model,
            nhead=fusion_heads,
            num_layers=fusion_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout
        )

        # Dual Prediction Heads
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_beams)
        )
        self.profile_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_beams)
        )

    def forward(self, rgb, gps):
        rgb_tokens = self.rgb_encoder(rgb)   # (B, 5, 192)
        gps_tokens = self.gps_encoder(gps)   # (B, 5, 192)

        fused_rep = self.fusion(rgb_tokens, gps_tokens)  # (B, 192)

        logits = self.cls_head(fused_rep)
        pred_profile = self.profile_head(fused_rep)

        return {
            "logits": logits,
            "pred_profile": pred_profile,
            "fused_rep": fused_rep
        }


# ----------------- BASELINES -----------------

class B1_GPSOnly(nn.Module):
    """GPS-only baseline model."""
    def __init__(self, d_model=192, n_beams=N_BEAMS):
        super().__init__()
        self.gps_enc = BiGRUPositionEncoder(d_model=d_model, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, n_beams)
        )

    def forward(self, rgb, gps):
        feat = self.gps_enc(gps)
        pooled = torch.mean(feat, dim=1)
        logits = self.head(pooled)
        return {"logits": logits}


class B3_Fusion(nn.Module):
    """Multimodal RGB+GPS baseline model (Classification Only)."""
    def __init__(self, d_model=192, n_beams=N_BEAMS):
        super().__init__()
        self.rgb_enc = ResNet18VisionEncoder(d_model=d_model)
        self.gps_enc = BiGRUPositionEncoder(d_model=d_model)
        self.fusion = PreLNTransformerFusion(d_model=d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, n_beams)
        )

    def forward(self, rgb, gps):
        rgb_tokens = self.rgb_enc(rgb)
        gps_tokens = self.gps_enc(gps)
        fused = self.fusion(rgb_tokens, gps_tokens)
        logits = self.head(fused)
        return {"logits": logits}


class P1_ClassificationOnly(nn.Module):
    """P1: Standard Classification-Only Model."""
    def __init__(self, d_model=192, n_beams=N_BEAMS):
        super().__init__()
        self.rgb_enc = ResNet18VisionEncoder(d_model=d_model)
        self.gps_enc = BiGRUPositionEncoder(d_model=d_model)
        self.fusion = PreLNTransformerFusion(d_model=d_model)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, n_beams)
        )

    def forward(self, rgb, gps):
        rgb_tokens = self.rgb_enc(rgb)
        gps_tokens = self.gps_enc(gps)
        fused = self.fusion(rgb_tokens, gps_tokens)
        logits = self.cls_head(fused)
        return {"logits": logits}


class P2_ProfileOnly(nn.Module):
    """P2: Continuous 256-dim Power Profile Regression Model."""
    def __init__(self, d_model=192, n_beams=N_BEAMS):
        super().__init__()
        self.rgb_enc = ResNet18VisionEncoder(d_model=d_model)
        self.gps_enc = BiGRUPositionEncoder(d_model=d_model)
        self.fusion = PreLNTransformerFusion(d_model=d_model)
        self.profile_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, n_beams)
        )

    def forward(self, rgb, gps):
        rgb_tokens = self.rgb_enc(rgb)
        gps_tokens = self.gps_enc(gps)
        fused = self.fusion(rgb_tokens, gps_tokens)
        pred_profile = self.profile_head(fused)
        return {"pred_profile": pred_profile, "logits": pred_profile}


# ----------------- LOSS FUNCTIONS -----------------

def compute_profile_smoothness_loss(pred_profile):
    """
    Smoothness penalty exploiting physical spatial angle continuity:
    sum_i (P_{i} - P_{i-1})^2 across adjacent beams.
    """
    diffs = pred_profile[:, 1:] - pred_profile[:, :-1]
    return torch.mean(diffs ** 2)

class MultiTaskLoss(nn.Module):
    """
    Loss = CE(logits, true_beam) + lambda_prof * MSE(pred_prof, true_prof) + lambda_smooth * Smoothness(pred_prof)
    """
    def __init__(self, lambda_prof=0.1, lambda_smooth=0.01):
        super().__init__()
        self.lambda_prof = lambda_prof
        self.lambda_smooth = lambda_smooth
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, target_beam, target_profile_db):
        loss_ce = self.ce_loss(outputs["logits"], target_beam)
        
        if "pred_profile" in outputs:
            loss_prof = self.mse_loss(outputs["pred_profile"], target_profile_db)
            loss_smooth = compute_profile_smoothness_loss(outputs["pred_profile"])
            total_loss = loss_ce + self.lambda_prof * loss_prof + self.lambda_smooth * loss_smooth
            return total_loss, {
                "loss_total": total_loss.item(),
                "loss_ce": loss_ce.item(),
                "loss_prof": loss_prof.item(),
                "loss_smooth": loss_smooth.item()
            }
        else:
            return loss_ce, {"loss_total": loss_ce.item(), "loss_ce": loss_ce.item()}

def create_model(model_name="P3", **kwargs):
    """Factory function for model instantiation."""
    models_dict = {
        "B1": B1_GPSOnly,
        "B3": B3_Fusion,
        "P1": P1_ClassificationOnly,
        "P2": P2_ProfileOnly,
        "P3": P3_MultiTaskProfile
    }
    if model_name not in models_dict:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from {list(models_dict.keys())}")
    return models_dict[model_name](**kwargs)
