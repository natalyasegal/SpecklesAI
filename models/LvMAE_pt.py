

# Requirements: torch, numpy, scikit-learn, xgboost
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install numpy scikit-learn xgboost

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

import numpy as np
from pathlib import Path

'''
Helpers
'''

# === Save / Load helpers (minimal) ===
def _get_cfg_from_model(model):
    """Extract just enough config to rebuild VideoMAE."""
    return {
        "in_chans": 1,
        "t_patch": getattr(model, "t_patch", 4),
        "p": getattr(model, "p", 4),
        "img_size": (40, 32, 32),
        "embed_dim": model.patch_embed.embed_dim,
        "depth": len(model.encoder_blocks),
        "num_heads": model.encoder_blocks[0].attn.num_heads if len(model.encoder_blocks) else 4,
        "decoder_embed_dim": model.decoder_embed.out_features if hasattr(model, "decoder_embed") else 64,
        "decoder_depth": len(getattr(model, "decoder_blocks", [])),
        "decoder_num_heads": (model.decoder_blocks[0].attn.num_heads
                              if len(getattr(model, "decoder_blocks", [])) else 4),
        "mask_ratio": getattr(model, "mask_ratio", 0.9),
        "use_cls_token": getattr(model, "use_cls_token", True),
        "proj_dim": getattr(model, "proj_dim", model.patch_embed.embed_dim),
    }

def save_lvmae_checkpoint(model, optimizer=None, scaler=None, epoch=0,
                          path="artifacts_lvmae/checkpoint.pt"):
    """One-file checkpoint with model weights, opt/scaler states (if provided), epoch, and cfg."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "cfg": _get_cfg_from_model(model),
    }
    torch.save(ckpt, path)
    print(f"[save_lvmae_checkpoint] Saved to {Path(path).resolve()}")

def load_lvmae_checkpoint(VideoMAE_cls, path="artifacts_lvmae/checkpoint.pt",
                          device=None, lr=None, weight_decay=None, amp=True):
    """
    Returns: model, optimizer, scaler, start_epoch
    - If lr/weight_decay are given, uses fresh optimizer at those values.
    - Otherwise tries to restore optimizer/scaler states from the checkpoint.
    - model is moved to device and set to train() (call .eval() for inference).
    """
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["cfg"]
    model = VideoMAE_cls(**cfg)
    model.load_state_dict(ckpt["model_state"], strict=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=(1e-3 if lr is None else lr),
        weight_decay=(0.05 if weight_decay is None else weight_decay),
    )

    try:
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    if ckpt.get("optimizer_state") is not None and lr is None and weight_decay is None:
        try: optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception as e: print("[load_lvmae_checkpoint] Optimizer state load failed; using fresh:", e)

    if ckpt.get("scaler_state") is not None:
        try: scaler.load_state_dict(ckpt["scaler_state"])
        except Exception as e: print("[load_lvmae_checkpoint] Scaler state load failed; using fresh:", e)

    start_epoch = int(ckpt.get("epoch", 0))
    model.train()
    return model, optimizer, scaler, start_epoch

'''
Helpers - load and resume training
'''

# ==== Saver / Loader that work for resume + inference ====

def save_for_resume_and_infer(model, optimizer=None, scaler=None, epoch=0,
                              path="artifacts_lvmae_1/checkpoint.pt"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "in_chans": 1,
        "t_patch": getattr(model, "t_patch", 4),
        "p": getattr(model, "p", 4),
        "img_size": (40, 32, 32), 
        "embed_dim": model.patch_embed.embed_dim,
        "depth": len(model.encoder_blocks),
        "num_heads": 4,  # set to what you used
        "mlp_ratio": 4.0,
        "decoder_embed_dim": model.decoder_embed.out_features,
        "decoder_depth": len(model.decoder_blocks),
        "decoder_num_heads": 4,  # set to what you used
        "mask_ratio": getattr(model, "mask_ratio", 0.9),
        "use_cls_token": getattr(model, "use_cls_token", True),
        "proj_dim": getattr(model, "proj_dim", model.patch_embed.embed_dim),
    }
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": (optimizer.state_dict() if optimizer is not None else None),
        "scaler_state": (scaler.state_dict() if scaler is not None else None),
        "epoch": int(epoch),
        "cfg": cfg,
    }, path)
    print(f"[save] wrote {Path(path).resolve()}")

def load_for_resume_and_infer(VideoMAE_cls, path="artifacts_lvmae_1/checkpoint.pt",
                              device=None, lr=None, weight_decay=None, amp=True):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("cfg")
    if cfg is None:
        raise KeyError("Checkpoint missing 'cfg'; rebuild cfg once and re-save with save_for_resume_and_infer.")
    model = VideoMAE_cls(**cfg)
    model.load_state_dict(ckpt["model_state"], strict=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=(1e-3 if lr is None else lr),
        weight_decay=(0.05 if weight_decay is None else weight_decay),
    )
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    if ckpt.get("optimizer_state") is not None and lr is None and weight_decay is None:
        try: optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception as e: print("[load] fresh optimizer (state load failed):", e)
    if ckpt.get("scaler_state") is not None:
        try: scaler.load_state_dict(ckpt["scaler_state"])
        except Exception as e: print("[load] fresh scaler (state load failed):", e)

    start_epoch = int(ckpt.get("epoch", 0))
    return model, optimizer, scaler, start_epoch

def run_resume(
    model,                              # <-- preloaded VideoMAE instance
    X: np.ndarray, y: np.ndarray,
    normalize: bool = False,
    mae_epochs: int = 30,
    mae_batch: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    amp: bool = True,
    pool: str = "cls",
    seed: int = 9,
    optimizer: torch.optim.Optimizer = None,   # <-- optional: pass loaded optimizer
    scaler: "torch.cuda.amp.GradScaler" = None,# <-- optional: pass loaded scaler
    start_epoch: int = 0,                      # <-- optional: resume epoch index
    checkpoint_path: str = "artifacts_lvmae_1/checkpoint.pt"
):
    """
    Continues self-supervised training of a *loaded* LV-MAE model.

    X: (N, 40, 32, 32[, 1])
    y: (N,) or (N,1)    (labels unused for MAE, but kept for consistency with your dataset)
    """
    set_seed(seed)
    assert X.ndim in (4, 5), f"Bad X.ndim={X.ndim}"
    y = y.squeeze()
    assert y.ndim == 1 and len(y) == X.shape[0], f"Bad y shape {y.shape}"

    # Device & model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # Data
    dtrain_mae = NumpyVideoDataset(X, y=None, normalize=normalize)
    loader_mae = DataLoader(
        dtrain_mae, batch_size=mae_batch, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )

    # Optimizer / scaler (create if not provided)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scaler is None:
        try:
            scaler = torch.amp.GradScaler('cuda', enabled=amp)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Train loop (reuse your train_mae style but keep resume epoch)
    print("==> Training LVMAE (self-supervised) on TRAIN split ...")
    total_epochs = mae_epochs
    for ep in range(start_epoch, start_epoch + total_epochs):
        epoch_loss, nsteps = 0.0, 0
        for batch in loader_mae:
            vids = batch[0] if isinstance(batch, (tuple, list)) else batch
            vids = vids.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                loss, _, _ = model(vids)

            scaler.scale(loss).backward()
            # (optional but recommended) stabilize with unscale+clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

            epoch_loss += float(loss.item()); nsteps += 1
        print(f"[MAE] Epoch {ep+1} | loss={epoch_loss/max(1,nsteps):.4f}")

    # Save checkpoint that supports both resume & inference
    save_for_resume_and_infer(
        model, optimizer=optimizer, scaler=scaler,
        epoch=start_epoch + total_epochs, path=checkpoint_path
    )

    return {
        "model": model,
        "optimizer": optimizer,
        "scaler": scaler,
        "epoch": start_epoch + total_epochs,
        "checkpoint": checkpoint_path,
    }


# ---- Helper: concatenate K consecutive embeddings (temporal aggregation) ----

def concat_temporal_embeddings(Z: np.ndarray, y: np.ndarray, K: int = 1):
    """
    Z: (N, D) embeddings in temporal order
    y: (N,) labels aligned with Z
    K: window length. If K==1, returns inputs unchanged.

    Returns:
      Zk: (N - K + 1, K*D)
      yk: (N - K + 1,)
    """
    assert Z.ndim == 2 and y.ndim == 1 and len(Z) == len(y), "Bad shapes"
    N, D = Z.shape
    if K <= 1 or N <= K:
        return (Z.copy(), y.copy()) if K <= 1 else (Z[-1:].repeat(1, axis=0), y[-1:])
    # simple, robust loop (fast enough for typical N)
    Zk = np.empty((N - K + 1, K * D), dtype=Z.dtype)
    for i in range(K - 1, N):
        Zk[i - (K - 1)] = Z[i - K + 1:i + 1].reshape(-1)
    yk = y[K - 1:]     # label at the end of each window
    return Zk, yk

def concat_y(y: np.ndarray, K: int = 1):
    """
    y: (N,) labels aligned with Z
    K: window length. If K==1, returns inputs unchanged.

    Returns:
      yk: (N - K + 1,)
    """
    assert y.ndim == 1, "Bad shapes"
    N = y.shape
    if K <= 1 or N <= K:
        return y.copy() if K <= 1 else y[-1:]
    # simple, robust loop (fast enough for typical N)
    yk = y[K - 1:]     # label at the end of each window
    return yk


# =========================
# Dataset (robust to channel-last or no channel)
# =========================

class NumpyVideoDataset(Dataset):
    """
    X: (N, 40, 32, 32, 1) or (N, 40, 32, 32)
    y: (N,) or (N,1). Returns (C=1, T=40, H=32, W=32) float32; label int64.
    """
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None, normalize: bool = False):
        assert X.ndim in (4, 5), f"Bad X.ndim={X.ndim}, expected 4 or 5"
        if X.ndim == 5:
            assert X.shape[1:4] == (40, 32, 32), f"Bad X.shape={X.shape}"
        else:  # (N,40,32,32)
            assert X.shape[1:4] == (40, 32, 32), f"Bad X.shape={X.shape}"

        self.X = X.astype(np.float32, copy=False)
        if y is None:
            self.y = None
        else:
            y = y.squeeze()
            assert y.ndim == 1 and y.shape[0] == X.shape[0], f"Bad y shape {y.shape}"
            self.y = y.astype(np.int64, copy=False)

        self.normalize = normalize

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # (40,32,32,1) or (40,32,32)
        if x.ndim == 3:
            x = x[..., None]            # -> (40,32,32,1)
        elif x.ndim != 4:
            raise ValueError(f"Unexpected sample ndim {x.ndim}")

        if self.normalize:
            x_min, x_max = x.min(), x.max()
            if (x_max - x_min) > 1e-9:
                x = (x - x_min) / (x_max - x_min)

        x = np.moveaxis(x, -1, 0)       # (1,40,32,32)
        x = torch.from_numpy(x)         # float32

        if self.y is None:
            return x
        else:
            return x, int(self.y[idx])


# =========================
# LVMAE model
# =========================

class PatchEmbed3D(nn.Module):
    """Tubelet patch embedding via Conv3d with kernel=stride=(t_patch, p, p)."""
    def __init__(self, in_chans=1, embed_dim=128, t_patch=4, p=4):
        super().__init__()
        self.t_patch = t_patch
        self.p = p
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=(t_patch, p, p),
                              stride=(t_patch, p, p))

    def forward(self, x: torch.Tensor):
        # x: (B,1,40,32,32)
        x = self.proj(x)  # (B, D, T', H', W')
        B, D, Tn, Hn, Wn = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, Tn * Hn * Wn, D)  # (B, N, D)
        return x, (Tn, Hn, Wn)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True, bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


def random_masking(x, mask_ratio: float):
    """
    Per-sample token masking by random shuffling.
    x: (B, N, D)
    Returns: x_keep, mask (1=masked), ids_restore
    """
    B, N, D = x.shape
    len_keep = int(N * (1.0 - mask_ratio))
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_keep = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

    mask = torch.ones(B, N, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)  # unshuffle to original order
    return x_keep, mask, ids_restore


class VideoMAE(nn.Module):
    """
    LVMAE for (T,H,W)=(40,32,32) with tubelets (t_patch,p,p)=(4,4,4).
    Encoder width=embed_dim (default 128). Projection head ensures final embedding dim=proj_dim.
    """
    def __init__(
        self,
        in_chans=1,
        t_patch=4,
        p=4,
        img_size=(40, 32, 32),
        embed_dim=128,
        depth=8,
        num_heads=4,
        mlp_ratio=4.0,
        decoder_embed_dim=128,
        decoder_depth=4,
        decoder_num_heads=4,
        mask_ratio=0.9,
        use_cls_token=True,
        proj_dim=128,    # final embedding size
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.use_cls_token = use_cls_token
        self.t_patch, self.p = t_patch, p
        self.proj_dim = proj_dim

        # Patch embedding
        self.patch_embed = PatchEmbed3D(in_chans, embed_dim, t_patch=t_patch, p=p)
        Tn = img_size[0] // t_patch
        Hn = img_size[1] // p
        Wn = img_size[2] // p
        self.num_patches = Tn * Hn * Wn  # 10*8*8 = 640
        self.patch_dim = t_patch * p * p * in_chans  # 4*4*4*1 = 64

        # Encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        self.pos_embed_enc = nn.Parameter(torch.zeros(1, self.num_patches + (1 if use_cls_token else 0), embed_dim))
        self.encoder_blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm_enc = nn.LayerNorm(embed_dim)

        # Decoder (for reconstruction only; does not affect downstream embedding size)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pos_embed_dec = nn.Parameter(torch.zeros(1, self.num_patches + (1 if use_cls_token else 0), decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([TransformerBlock(decoder_embed_dim, decoder_num_heads, 4.0) for _ in range(decoder_depth)])
        self.norm_dec = nn.LayerNorm(decoder_embed_dim)
        self.head = nn.Linear(decoder_embed_dim, self.patch_dim)

        # Projection head for final embeddings (keeps downstream embedding fixed, e.g., 128)
        self.proj_head = nn.Identity() if proj_dim == embed_dim else nn.Linear(embed_dim, proj_dim)

        self._init_weights()

    def _init_weights(self):
        def _trunc_normal_(w, std=0.02): nn.init.trunc_normal_(w, std=std)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _trunc_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                _trunc_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        if self.cls_token is not None:
            nn.init.zeros_(self.cls_token)
        nn.init.zeros_(self.mask_token)
        nn.init.trunc_normal_(self.pos_embed_enc, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_dec, std=0.02)

    # ---- MAE core ----
    def forward_encoder(self, x, mask_ratio):
        x, _ = self.patch_embed(x)  # (B, N, D)
        B, N, D = x.shape
        pe = self.pos_embed_enc[:, (1 if self.use_cls_token else 0):(N + (1 if self.use_cls_token else 0))]
        x = x + pe

        x_keep, mask, ids_restore = random_masking(x, mask_ratio)
        if self.use_cls_token:
            cls_tok = self.cls_token + self.pos_embed_enc[:, :1, :]
            x_keep = torch.cat([cls_tok.expand(B, -1, -1), x_keep], dim=1)

        for blk in self.encoder_blocks:
            x_keep = blk(x_keep)
        x_keep = self.norm_enc(x_keep)
        return x_keep, mask, ids_restore

    def forward_decoder(self, x_enc, ids_restore):
        x = self.decoder_embed(x_enc)
        if self.use_cls_token:
            x_vis, cls_tok = x[:, 1:, :], x[:, :1, :]
        else:
            x_vis, cls_tok = x, None

        B, N_keep, Dd = x_vis.shape
        N = self.num_patches
        n_mask = N - N_keep
        mask_tokens = self.mask_token.expand(B, n_mask, Dd)
        x_ = torch.cat([x_vis, mask_tokens], dim=1)
        x_full = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, Dd))

        # add decoder pos emb
        x_full = x_full + self.pos_embed_dec[:, (1 if self.use_cls_token else 0):(N + (1 if self.use_cls_token else 0)), :]
        if self.use_cls_token:
            x_full = torch.cat([cls_tok + self.pos_embed_dec[:, :1, :], x_full], dim=1)

        for blk in self.decoder_blocks:
            x_full = blk(x_full)
        x_full = self.norm_dec(x_full)
        x_payload = x_full[:, 1:, :] if self.use_cls_token else x_full
        pred = self.head(x_payload)  # (B, N, patch_dim)
        return pred

    def patchify(self, videos: torch.Tensor) -> torch.Tensor:
        """
        videos: (B, 1, 40, 32, 32) -> (B, N, patch_dim)
        Uses explicit reshape based on (t_patch, p, p) to avoid unfold quirks.
        """
        B, C, T, H, W = videos.shape
        tp, p = self.t_patch, self.p

        # sanity checks
        assert T % tp == 0 and H % p == 0 and W % p == 0, \
            f"Input dims not divisible by patch sizes: (T,H,W)=({T},{H},{W}), (tp,p)=({tp},{p})"

        Tn, Hn, Wn = T // tp, H // p, W // p  # number of tubelets along each axis
        # reshape into blocks and flatten each (tp×p×p) cube
        v = videos.view(B, C, Tn, tp, Hn, p, Wn, p)                  # (B,C,Tn,tp,Hn,p,Wn,p)
        v = v.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()           # (B,Tn,Hn,Wn,C,tp,p,p)
        patches = v.view(B, Tn * Hn * Wn, C * tp * p * p)            # (B, N, patch_dim)
        return patches

    def forward(self, videos):
        x_enc, mask, ids_restore = self.forward_encoder(videos, self.mask_ratio)
        pred = self.forward_decoder(x_enc, ids_restore)
        target = self.patchify(videos)
        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss, pred, mask

    @torch.no_grad()
    def encode_no_mask(self, videos, pool: str = "cls"):
        """
        Returns final embeddings of size proj_dim (default 128).
        """
        x, _ = self.patch_embed(videos)   # (B,N,D_enc)
        B, N, D = x.shape
        if self.use_cls_token:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = x + self.pos_embed_enc[:, 1:N+1, :]
            x = torch.cat([cls_token + self.pos_embed_enc[:, :1, :], x], dim=1)
        else:
            x = x + self.pos_embed_enc[:, :N, :]

        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.norm_enc(x)

        enc_vec = x[:, 0, :] if (self.use_cls_token and pool == "cls") else x.mean(dim=1)
        return self.proj_head(enc_vec)     # -> (B, proj_dim)


# =========================
# Train / Embeddings / XGBoost
# =========================

@dataclass
class TrainCfg:
    batch_size: int = 256
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.05
    mask_ratio: float = 0.9
    num_workers: int = 0     # 0 is safer for notebooks
    amp: bool = True
    seed: int = 42

def set_seed(seed: int = 42):
    import random, os
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def train_mae(model: VideoMAE, loader: DataLoader, cfg: TrainCfg, device):
    model.train()
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for ep in range(cfg.epochs):
        epoch_loss, nsteps = 0.0, 0
        for batch in loader:
            vids = batch[0] if isinstance(batch, (tuple, list)) else batch
            vids = vids.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=cfg.amp):
                loss, _, _ = model(vids)

            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            epoch_loss += float(loss.item()); nsteps += 1
        print(f"[MAE] Epoch {ep+1}/{cfg.epochs} | loss={epoch_loss/max(1,nsteps):.4f}")
    return opt, scaler

@torch.no_grad()
def extract_embeddings(model: VideoMAE, loader: DataLoader, device, pool="cls") -> np.ndarray:
    model.eval()
    zs = []
    for batch in loader:
        vids = batch[0] if isinstance(batch, (tuple, list)) else batch
        vids = vids.to(device, non_blocking=True)
        z = model.encode_no_mask(vids, pool=pool)   # -> (B, proj_dim=128)
        zs.append(z.float().cpu().numpy())
    return np.concatenate(zs, axis=0)


# =========================
# End-to-end
# =========================
def data_sanity(X, y):
    assert X.ndim in (4, 5)
    y = y.squeeze()
    assert y.ndim == 1 and len(y) == X.shape[0]
    return X, y

def extract_embeddings_wrapper_one(model, X, y, pool="cls", normalize=False, seed = 9,
                       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    X, y = data_sanity(X, y)

    # Datasets & loaders
    #dtrain_mae = NumpyVideoDataset(X_train, y=None, normalize=normalize)
    dset_cls = NumpyVideoDataset(X, y, normalize=normalize)

    print("==> Extracting embeddings (proj_dim) ...")
    loader_emb_set = DataLoader(dset_cls, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    Z_set = extract_embeddings(model, loader_emb_set, device, pool=pool)
    print(f"Emb dims: set {Z_set.shape}")
    return Z_set, y


def extract_embeddings_wrapper(
    model, X, y, X_test, y_test,
    val_size=0.2,               # <-- add explicit val_size
    shuffle=True, pool="cls",
    normalize=False, seed=9,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    X_tmp, y_tmp = data_sanity(X, y)
    X_test, y_test = data_sanity(X_test, y_test)

    if shuffle:
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp,
            shuffle=True,
            test_size=val_size,
            random_state=seed,
            stratify=y_tmp,
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp,
            shuffle=False,
            test_size=val_size,
            random_state=seed,   # ignored when shuffle=False, but harmless
        )

    dtrain_cls = NumpyVideoDataset(X_train, y_train, normalize=normalize)
    dval_cls   = NumpyVideoDataset(X_val,   y_val,   normalize=normalize)
    dtest_cls  = NumpyVideoDataset(X_test,  y_test,  normalize=normalize)

    print("==> Extracting embeddings (proj_dim) ...")
    loader_emb_train = DataLoader(dtrain_cls, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    loader_emb_val   = DataLoader(dval_cls,   batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    loader_emb_test  = DataLoader(dtest_cls,  batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    Z_train = extract_embeddings(model, loader_emb_train, device, pool=pool)
    Z_val   = extract_embeddings(model, loader_emb_val,   device, pool=pool)
    Z_test  = extract_embeddings(model, loader_emb_test,  device, pool=pool)

    print(f"Emb dims: train {Z_train.shape}, val {Z_val.shape}, test {Z_test.shape}")
    return Z_train, y_train, Z_val, y_val, Z_test, y_test

def extract_embeddings_wrapper_old(model, X, y, X_test, y_test, shuffle=True, pool="cls", normalize=False, seed = 9,
                       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    X_tmp, y_tmp = data_sanity(X, y)
    X_test, y_test = data_sanity(X_test, y_test)

    # Splits
    if shuffle == False:
      X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, shuffle=shuffle,
            test_size=val_size/(1.0 - test_size), random_state=seed)
    else:
      X_train, X_val, y_train, y_val = train_test_split(
              X_tmp, y_tmp, shuffle=shuffle,
              test_size=val_size/(1.0 - test_size), random_state=seed, stratify=y_tmp)

    # Datasets & loaders
    dtrain_mae = NumpyVideoDataset(X_train, y=None, normalize=normalize)
    dtrain_cls = NumpyVideoDataset(X_train, y_train, normalize=normalize)
    dval_cls   = NumpyVideoDataset(X_val,   y_val,   normalize=normalize)
    dtest_cls  = NumpyVideoDataset(X_test,  y_test,  normalize=normalize)

    print("==> Extracting embeddings (proj_dim) ...")
    loader_emb_train = DataLoader(dtrain_cls, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    loader_emb_val   = DataLoader(dval_cls,   batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    loader_emb_test  = DataLoader(dtest_cls,  batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    Z_train = extract_embeddings(model, loader_emb_train, device, pool=pool)  # (N_train, proj_dim)
    Z_val   = extract_embeddings(model, loader_emb_val,   device, pool=pool)
    Z_test  = extract_embeddings(model, loader_emb_test,  device, pool=pool)

    print(f"Emb dims: train {Z_train.shape}, val {Z_val.shape}, test {Z_test.shape}")

    return Z_train, y_train, Z_val, y_val, Z_test, y_test

#Z_train, y_train, Z_val, y_val, Z_test, y_test = extract_embeddings_wrapper(model, X, y, X_test, y_test)

def run_end_to_end(X: np.ndarray, y: np.ndarray,
                   normalize=False,
                   mae_embed_dim=128,      # encoder width
                   proj_dim=128,           # final embedding size for downstream
                   mae_epochs=10, #100, #change back to 100
                   mae_batch=256,
                   mask_ratio=0.9,
                   pool="cls",
                   #val_size=0.2,
                   #test_size=0.2,
                   seed=42):
    """
    X: (N, 40, 32, 32, 1) or (N, 40, 32, 32)
    y: (N,) or (N,1) with {0,1}
    """
    set_seed(seed)
    assert X.ndim in (4, 5)
    y = y.squeeze()
    assert y.ndim == 1 and len(y) == X.shape[0]
    '''
    # Splits
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size/(1.0 - test_size), random_state=seed, stratify=y_tmp
    )
    '''
    X_train = X
    y_train = y

    # Datasets & loaders
    dtrain_mae = NumpyVideoDataset(X_train, y=None, normalize=normalize)
    dtrain_cls = NumpyVideoDataset(X_train, y_train, normalize=normalize)
    #dval_cls   = NumpyVideoDataset(X_val,   y_val,   normalize=normalize)
    #dtest_cls  = NumpyVideoDataset(X_test,  y_test,  normalize=normalize)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VideoMAE(
        in_chans=1,
        t_patch=4, p=4,
        img_size=(40,32,32),
        embed_dim=mae_embed_dim, depth=8, num_heads=4,
        decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4,
        mask_ratio=mask_ratio, use_cls_token=True,
        proj_dim=proj_dim,
    ).to(device)

    cfg = TrainCfg(batch_size=mae_batch, epochs=mae_epochs, lr=1e-3, weight_decay=0.05,
                   mask_ratio=mask_ratio, num_workers=0, amp=True, seed=seed)

    loader_mae = DataLoader(dtrain_mae, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    print("==> Training LVMAE (self-supervised) on TRAIN split ...")
    opt, scaler = train_mae(model, loader_mae, cfg, device)

    # Save a checkpoint you can resume from:
    save_lvmae_checkpoint(model, optimizer=opt, scaler=scaler,
                      epoch=cfg.epochs, path="artifacts_lvmae_1/checkpoint.pt")
  
    return dict(model=model) #, xgb=clf)


# =========================
# Example usage
# =========================
# if __name__ == "__main__":
#     # x1: (10000, 40, 32, 32, 1) or (10000, 40, 32, 32)
#     # y1: (10000, 1) or (10000,)
#     x1 = x1.astype(np.float32, copy=False)
#     y1 = y1.squeeze().astype(np.int64, copy=False)
#     results = run_end_to_end(
#         X=x1, y=y1,
#         normalize=False,
#         mae_embed_dim=128,   # encoder width
#         proj_dim=128,        # final embedding size
#         mae_epochs=100,
#         mae_batch=64, #256,
#         mask_ratio=0.9,
#         pool="cls",
#         val_size=0.2,
#         test_size=0.2,
#         seed=42
#     )
