"""
구조물 안정성 예측 - NaN 수정 버전
SAM 제거, 순수 AdamW + GradScaler
"""

import os, cv2, warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from timm.utils import ModelEmaV2
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')


# ================================================================
# CONFIG
# ================================================================
class CFG:
    DATA_DIR        = './'
    IMG_SIZE        = 288
    BATCH_SIZE      = 32
    EPOCHS          = 60
    MAX_LR          = 2e-4
    WEIGHT_DECAY    = 1e-2
    NUM_FOLDS       = 5
    EMA_DECAY       = 0.9995
    LABEL_SMOOTHING = 0.05
    MIXUP_ALPHA     = 0.3
    SEED            = 777
    BACKBONES = [
        'convnext_small_in22ft1k',
        'swinv2_tiny_window16_256',
    ]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(CFG.SEED)

print(f"Device: {CFG.DEVICE}")
print(f"PyTorch: {torch.__version__}")


# ================================================================
# OPTICAL FLOW
# ================================================================
def compute_motion_score(video_path, n_frames=6):
    if not os.path.exists(video_path):
        return -1.0
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return -1.0
    frames = []
    for _ in range(min(n_frames, total)):
        ret, f = cap.read()
        if ret:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(frames) < 2:
        return -1.0
    mags = []
    for i in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i+1], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mags.append(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean())
    return float(np.mean(mags))


def build_motion_scores(train_df):
    print("Computing optical flow motion scores...")
    scores = {}
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        img_id = str(row['id'])
        src    = row['source']
        vpath  = os.path.join(CFG.DATA_DIR, src, img_id, 'simulation.mp4')
        scores[img_id] = compute_motion_score(vpath)
    valid = [v for v in scores.values() if v >= 0]
    if valid:
        vmin, vmax = min(valid), max(valid)
        rng = vmax - vmin if vmax > vmin else 1.0
        scores = {k: (v - vmin) / rng if v >= 0 else -1.0
                  for k, v in scores.items()}
    return scores


# ================================================================
# AUGMENTATION
# ================================================================
def build_aug(is_train, view='front'):
    if not is_train:
        return A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    flips = [A.HorizontalFlip(p=0.5)]
    if view == 'top':
        flips.append(A.VerticalFlip(p=0.5))

    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        *flips,
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.12,
                           rotate_limit=8, p=0.6),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.4,
                                       contrast_limit=0.4, p=1.0),
            A.RandomGamma(gamma_limit=(60, 140), p=1.0),
            A.CLAHE(clip_limit=4.0, p=1.0),
        ], p=0.8),
        A.ColorJitter(brightness=0.35, contrast=0.35,
                      saturation=0.35, hue=0.06, p=0.6),
        A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=35,
                             val_shift_limit=35, p=0.5),
        A.OneOf([
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, p=1.0),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
        ], p=0.2),
        A.ISONoise(color_shift=(0.01, 0.06), intensity=(0.1, 0.4), p=0.25),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.25),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.CoarseDropout(max_holes=6, max_height=28, max_width=28,
                        min_holes=1, fill_value=0, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# ================================================================
# DATASET
# ================================================================
class StructureDataset(Dataset):
    def __init__(self, df, is_train=True, is_test=False, motion_scores=None):
        self.df            = df.reset_index(drop=True)
        self.is_train      = is_train
        self.is_test       = is_test
        self.motion_scores = motion_scores or {}
        self.front_tf      = build_aug(is_train, view='front')
        self.top_tf        = build_aug(is_train, view='top')

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        img_id = str(row['id'])
        folder = row.get('source', 'test')

        def load(name):
            path = os.path.join(CFG.DATA_DIR, folder, img_id, name)
            img  = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Cannot load: {path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        front_t = self.front_tf(image=load('front.png'))['image']
        top_t   = self.top_tf(image=load('top.png'))['image']
        motion  = torch.tensor([self.motion_scores.get(img_id, -1.0)],
                               dtype=torch.float32)

        if self.is_test:
            return front_t, top_t, motion, img_id

        raw = 1.0 if row['label'] == 'stable' else 0.0
        if self.is_train:
            s     = CFG.LABEL_SMOOTHING
            label = raw * (1 - s) + (1 - raw) * s
        else:
            label = raw
        return front_t, top_t, motion, torch.tensor(label, dtype=torch.float32)


# ================================================================
# MIXUP
# ================================================================
def mixup(front, top, labels, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(front.size(0), device=front.device)
    return (lam * front  + (1-lam) * front[idx],
            lam * top    + (1-lam) * top[idx],
            lam * labels + (1-lam) * labels[idx])


# ================================================================
# GeM POOLING
# ================================================================
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p).squeeze(-1).squeeze(-1)


# ================================================================
# CrossView Attention
# ================================================================
class CrossViewAttn(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        # heads가 dim을 나누어 떨어지도록 보정
        while dim % heads != 0:
            heads -= 1
        self.attn = nn.MultiheadAttention(dim, heads,
                                          batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q, kv):
        out, _ = self.attn(q.unsqueeze(1), kv.unsqueeze(1), kv.unsqueeze(1))
        return self.norm(q + out.squeeze(1))


# ================================================================
# MODEL
# ================================================================
class FusionNet(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=True, num_classes=0, global_pool='')

        with torch.no_grad():
            dummy    = torch.randn(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE)
            feat_map = self.backbone(dummy)
            if feat_map.ndim == 3:   # SwinV2: (B, N, C)
                n        = int(feat_map.size(1) ** 0.5)
                feat_map = feat_map.transpose(1, 2).reshape(1, -1, n, n)
            feat_dim = feat_map.size(1)

        print(f"  feat_dim={feat_dim}")

        self.gem        = GeM(p=3.0)
        self.cross_f    = CrossViewAttn(feat_dim)
        self.cross_t    = CrossViewAttn(feat_dim)
        self.motion_emb = nn.Sequential(
            nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 32)
        )
        in_dim = feat_dim * 3 + 32
        self.head = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def _extract(self, x):
        feat = self.backbone(x)
        if feat.ndim == 3:
            n    = int(feat.size(1) ** 0.5)
            feat = feat.transpose(1, 2).reshape(feat.size(0), -1, n, n)
        return self.gem(feat)

    def forward(self, front, top, motion):
        f     = self._extract(front)
        t     = self._extract(top)
        f_att = self.cross_f(f, t)
        t_att = self.cross_t(t, f)
        inter = f_att * t_att
        m     = self.motion_emb(motion)
        z     = torch.cat([f_att, t_att, inter, m], dim=1)
        return self.head(z).squeeze(1)


# ================================================================
# TEMPERATURE SCALING
# ================================================================
class TempScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))

    def forward(self, logits): return logits / self.T


def calibrate(model, val_loader):
    print("  Calibrating temperature...")
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for front, top, motion, labels in val_loader:
            front  = front.to(CFG.DEVICE)
            top    = top.to(CFG.DEVICE)
            motion = motion.to(CFG.DEVICE)
            logits_list.append(model(front, top, motion).cpu())
            labels_list.append(labels)
    all_logits = torch.cat(logits_list)
    all_labels = torch.cat(labels_list)

    ts   = TempScale()
    opt  = torch.optim.LBFGS([ts.T], lr=0.01, max_iter=200)
    crit = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = crit(ts(all_logits), all_labels)
        loss.backward()
        return loss

    opt.step(closure)
    T = ts.T.item()
    print(f"  → Temperature: {T:.4f}")
    return T


# ================================================================
# TRAINING
# ================================================================
def train_one_backbone(backbone_name, train_df, dev_df, motion_scores):
    print(f"\n{'#'*60}")
    print(f"  BACKBONE: {backbone_name}")
    print(f"{'#'*60}")

    skf      = StratifiedKFold(n_splits=CFG.NUM_FOLDS, shuffle=True,
                               random_state=CFG.SEED)
    train_df = train_df.copy()
    train_df['fold'] = -1
    for fold, (_, vi) in enumerate(skf.split(train_df, train_df['label'])):
        train_df.loc[vi, 'fold'] = fold

    safe_name    = backbone_name.replace('/', '_')
    temperatures = []

    for fold in range(CFG.NUM_FOLDS):
        print(f"\n{'='*55}  FOLD {fold+1}/{CFG.NUM_FOLDS}")

        trn_df = train_df[train_df['fold'] != fold].reset_index(drop=True)
        val_df = dev_df.copy()

        trn_loader = DataLoader(
            StructureDataset(trn_df, is_train=True,
                             motion_scores=motion_scores),
            batch_size=CFG.BATCH_SIZE, shuffle=True,
            num_workers=2, pin_memory=True,
            persistent_workers=True, drop_last=True
        )
        val_loader = DataLoader(
            StructureDataset(val_df, is_train=False, motion_scores={}),
            batch_size=CFG.BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=True, persistent_workers=True
        )

        model     = FusionNet(backbone_name).to(CFG.DEVICE)
        ema_model = ModelEmaV2(model, decay=CFG.EMA_DECAY)
        crit      = nn.BCEWithLogitsLoss()

        # SAM 제거 → 순수 AdamW
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CFG.MAX_LR,
            weight_decay=CFG.WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=CFG.MAX_LR,
            epochs=CFG.EPOCHS, steps_per_epoch=len(trn_loader),
            pct_start=0.1, anneal_strategy='cos'
        )
        scaler = torch.cuda.amp.GradScaler()

        best_loss = float('inf')

        for epoch in range(CFG.EPOCHS):
            model.train()
            tr_loss   = 0.0
            nan_count = 0
            pbar      = tqdm(trn_loader, desc=f'E{epoch+1:02d}/{CFG.EPOCHS}')

            for batch_idx, (front, top, motion, labels) in enumerate(pbar):
                front  = front.to(CFG.DEVICE, non_blocking=True)
                top    = top.to(CFG.DEVICE, non_blocking=True)
                motion = motion.to(CFG.DEVICE, non_blocking=True)
                labels = labels.to(CFG.DEVICE, non_blocking=True)

                # NaN 입력 체크 (첫 epoch만)
                if epoch == 0 and batch_idx == 0:
                    print(f"\n  [DEBUG] front: {front.shape} "
                          f"min={front.min():.2f} max={front.max():.2f} "
                          f"nan={front.isnan().any()}")
                    print(f"  [DEBUG] motion: {motion[:4].squeeze().tolist()}")
                    print(f"  [DEBUG] labels: {labels[:4].tolist()}")

                if CFG.MIXUP_ALPHA > 0 and np.random.rand() < 0.5:
                    front, top, labels = mixup(front, top, labels,
                                               CFG.MIXUP_ALPHA)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    out  = model(front, top, motion)
                    loss = crit(out, labels)

                # NaN loss 스킵
                if torch.isnan(loss):
                    nan_count += 1
                    optimizer.zero_grad()
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                ema_model.update(model)

                tr_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                  'nan': nan_count})

            # Validation
            ema_model.module.eval()
            vl_loss = 0.0
            with torch.no_grad():
                for front, top, motion, labels in val_loader:
                    front  = front.to(CFG.DEVICE, non_blocking=True)
                    top    = top.to(CFG.DEVICE, non_blocking=True)
                    motion = motion.to(CFG.DEVICE, non_blocking=True)
                    labels = labels.to(CFG.DEVICE, non_blocking=True)
                    with torch.cuda.amp.autocast():
                        vl_loss += crit(
                            ema_model.module(front, top, motion), labels
                        ).item()

            valid_batches = len(trn_loader) - nan_count
            avg_tr = tr_loss / max(valid_batches, 1)
            avg_vl = vl_loss / len(val_loader)
            print(f"  Epoch {epoch+1:02d} | "
                  f"tr={avg_tr:.4f} | dev={avg_vl:.4f} | "
                  f"nan_batches={nan_count}")

            if avg_vl < best_loss:
                best_loss = avg_vl
                torch.save(ema_model.module.state_dict(),
                           f'best_{safe_name}_fold{fold}.pth')
                print(f"  🔥 best dev={best_loss:.4f}  saved")

        # Temperature calibration
        best_model = FusionNet(backbone_name).to(CFG.DEVICE)
        best_model.load_state_dict(
            torch.load(f'best_{safe_name}_fold{fold}.pth'))
        T = calibrate(best_model, val_loader)
        temperatures.append(T)
        torch.save({'sd': best_model.state_dict(), 'T': T},
                   f'calib_{safe_name}_fold{fold}.pth')

    return temperatures


# ================================================================
# INFERENCE
# ================================================================
def predict_one_model(model, front, top, motion, T):
    s = CFG.IMG_SIZE
    with torch.cuda.amp.autocast():
        def sig(f, t):
            return torch.sigmoid(model(f, t, motion) / T)

        o1 = sig(front, top)
        o2 = sig(torch.flip(front, [3]), torch.flip(top, [3]))
        o3 = sig(front, torch.flip(top, [2]))
        o4 = sig(torch.flip(front, [2]), top)
        o5 = sig(torch.clamp(front * 1.15, -3, 3),
                 torch.clamp(top   * 1.15, -3, 3))
        o6 = sig(torch.clamp(front * 0.85, -3, 3),
                 torch.clamp(top   * 0.85, -3, 3))
        fs  = F.interpolate(front, scale_factor=1.1, mode='bilinear',
                            align_corners=False)
        ts_ = F.interpolate(top,   scale_factor=1.1, mode='bilinear',
                            align_corners=False)
        st  = int((s * 1.1 - s) / 2)
        o7  = sig(fs[:, :, st:st+s, st:st+s],
                  ts_[:, :, st:st+s, st:st+s])
    return (o1 + o2 + o3 + o4 + o5 + o6 + o7) / 7.0


def inference_ensemble():
    test_df = pd.read_csv(
        os.path.join(CFG.DATA_DIR, 'sample_submission.csv'))
    test_df['source'] = 'test'

    test_loader = DataLoader(
        StructureDataset(test_df, is_train=False, is_test=True,
                         motion_scores={}),
        batch_size=CFG.BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    n         = len(test_df)
    all_preds = []

    for backbone in CFG.BACKBONES:
        safe = backbone.replace('/', '_')
        for fold in range(CFG.NUM_FOLDS):
            ckpt  = torch.load(f'calib_{safe}_fold{fold}.pth',
                               map_location=CFG.DEVICE)
            model = FusionNet(backbone).to(CFG.DEVICE)
            model.load_state_dict(ckpt['sd'])
            model.eval()
            T = ckpt['T']

            fold_preds = []
            with torch.no_grad():
                for front, top, motion, _ in tqdm(
                        test_loader, desc=f'{safe} fold{fold}'):
                    front  = front.to(CFG.DEVICE, non_blocking=True)
                    top    = top.to(CFG.DEVICE, non_blocking=True)
                    motion = motion.to(CFG.DEVICE, non_blocking=True)
                    pred   = predict_one_model(model, front, top, motion, T)
                    fold_preds.extend(pred.cpu().numpy())
            all_preds.append(np.array(fold_preds))

    # Rank Averaging
    ranked = np.stack([rankdata(p) / n for p in all_preds], axis=0)
    final  = ranked.mean(axis=0)

    stable_probs   = np.clip(final, 1e-6, 1 - 1e-6)
    unstable_probs = 1.0 - stable_probs

    submission = pd.read_csv(
        os.path.join(CFG.DATA_DIR, 'sample_submission.csv'))
    submission['unstable_prob'] = unstable_probs
    submission['stable_prob']   = stable_probs

    total = submission['unstable_prob'] + submission['stable_prob']
    submission['unstable_prob'] /= total
    submission['stable_prob']   /= total

    out = 'submission_final.csv'
    submission.to_csv(out, index=False)
    print(f"\n🚀 Saved → '{out}'")
    print(submission[['unstable_prob', 'stable_prob']].describe())


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    train_df = pd.read_csv(os.path.join(CFG.DATA_DIR, 'train.csv'))
    dev_df   = pd.read_csv(os.path.join(CFG.DATA_DIR, 'dev.csv'))

    train_df['source'] = train_df['id'].apply(
        lambda x: str(x).split('_')[0].lower())
    dev_df['source'] = 'dev'

    motion_scores = build_motion_scores(train_df)

    for backbone in CFG.BACKBONES:
        train_one_backbone(backbone, train_df, dev_df, motion_scores)

    inference_ensemble()