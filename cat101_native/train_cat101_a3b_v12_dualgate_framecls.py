# -*- coding: utf-8 -*-
import argparse
import importlib.util
import json
import random
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


ROOT = Path("/root/autodl-tmp/MS-TCN2")
CAT = Path("/root/autodl-tmp/101data/cataract-101")
DATA = ROOT / "data/cataract101_m10_a0_rgb2048"
MASKD = ROOT / "data/cataract101_m10_a2c_sam2_masks_full"
OLD_TRAIN_PY = ROOT / "m8/7_train_m8_igcoattn_v1_2_dualgate_framecls.py"


def log(x):
    print(x, flush=True)


def load_old_module():
    spec = importlib.util.spec_from_file_location("m8_v12", str(OLD_TRAIN_PY))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_mapping():
    mp = {}
    p = DATA / "mapping.txt"
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if parts[0].isdigit():
            idx = int(parts[0])
            name = " ".join(parts[1:])
        else:
            name = " ".join(parts[:-1])
            idx = int(parts[-1])
        mp[name] = idx
        mp[str(idx)] = idx
    return mp



def get_num_classes():
    p = DATA / "mapping.txt"
    ids = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if parts[0].isdigit():
            ids.append(int(parts[0]))
        elif parts[-1].isdigit():
            ids.append(int(parts[-1]))
        else:
            ids.append(len(ids))
    return len(set(ids))

def read_bundle(split, mode):
    p = DATA / "splits" / f"{mode}.split{split}.bundle"
    vids = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        s = s.replace(".txt", "")
        if s.startswith("case_"):
            vids.append(int(s.replace("case_", "")))
    return vids


def read_gt(video_id, label_map):
    p = DATA / "groundTruth" / f"case_{video_id}.txt"
    labels = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s in label_map:
            labels.append(label_map[s])
        else:
            raise KeyError(f"Unknown label in {p}: {s}")
    return np.asarray(labels, dtype=np.int64)


def unpack_mask(packed, h=224, w=224):
    flat = np.unpackbits(packed, bitorder="little")[: h * w]
    return flat.reshape(h, w).astype(np.float32)


def bce_prob(pred, target):
    """
    Safe BCE for attention/gate probability maps.
    Some priors/gates may contain numerical values slightly outside [0,1],
    especially under AMP. Clamp both input and target to avoid CUDA assert.
    """
    pred = torch.nan_to_num(pred.float(), nan=0.5, posinf=1.0, neginf=0.0)
    target = torch.nan_to_num(target.float(), nan=0.0, posinf=1.0, neginf=0.0)

    pred = pred.clamp(1e-4, 1.0 - 1e-4)
    target = target.clamp(0.0, 1.0)

    if pred.is_cuda:
        with torch.cuda.amp.autocast(enabled=False):
            return F.binary_cross_entropy(pred, target)
    return F.binary_cross_entropy(pred, target)

def downsample_mask(mask, size_hw):
    return F.interpolate(mask, size=size_hw, mode="nearest")


class Cat101FrameSampleDataset(Dataset):
    """
    直接读取 Cataract101 mp4 + SAM2 packed npz。
    不生成 png，不生成新帧文件。
    """
    def __init__(
        self,
        split=1,
        mode="train",
        max_frames_per_class_per_video=80,
        seed=3407,
    ):
        super().__init__()
        self.split = split
        self.mode = mode
        self.label_map = load_mapping()
        self.video_ids = read_bundle(split, mode)

        rng = np.random.default_rng(seed + (0 if mode == "train" else 999))

        self.samples = []
        self.video_labels = {}
        self.video_frame_ids = {}

        for vid in self.video_ids:
            labels = read_gt(vid, self.label_map)
            mask_path = MASKD / "npz" / f"case_{vid}.npz"
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)

            z = np.load(mask_path)
            frame_ids = z["frame_ids"]

            if len(frame_ids) != len(labels):
                raise RuntimeError(f"case_{vid}: mask T={len(frame_ids)} != gt T={len(labels)}")

            self.video_labels[vid] = labels
            self.video_frame_ids[vid] = frame_ids

            by_cls = defaultdict(list)
            for i, y in enumerate(labels):
                by_cls[int(y)].append(i)

            for y, idxs in by_cls.items():
                idxs = np.asarray(idxs, dtype=np.int64)
                if max_frames_per_class_per_video > 0 and len(idxs) > max_frames_per_class_per_video:
                    idxs = rng.choice(idxs, size=max_frames_per_class_per_video, replace=False)
                idxs = np.sort(idxs)
                for li in idxs:
                    self.samples.append((vid, int(li), int(y)))

        # 为了减少 video seek/cache 混乱，按 video + local_idx 排序，不随机打散。
        self.samples.sort(key=lambda x: (x[0], x[1]))

        self.cur_vid = None
        self.cap = None
        self.mask_cache = None

        log(f"[DATASET] mode={mode}, split={split}, videos={len(self.video_ids)}, samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _load_video_cache(self, vid):
        if self.cur_vid == vid:
            return

        if self.cap is not None:
            self.cap.release()

        video_path = CAT / "videos" / f"case_{vid}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        mask_path = MASKD / "npz" / f"case_{vid}.npz"
        z = np.load(mask_path)

        # 注意：这里会把当前视频 mask 解压进内存，但只缓存一个视频，避免爆内存。
        self.mask_cache = {
            "frame_ids": z["frame_ids"],
            "mask_h": int(z["mask_h"]),
            "mask_w": int(z["mask_w"]),
            "eye": z["eye_mask_packed"],
            "inst": z["inst_union_mask_packed"],
        }
        self.cap = cap
        self.cur_vid = vid

    def __getitem__(self, idx):
        vid, local_idx, y = self.samples[idx]
        self._load_video_cache(vid)

        gid = int(self.mask_cache["frame_ids"][local_idx])

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, gid)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"case_{vid}: cannot read frame {gid}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

        h = self.mask_cache["mask_h"]
        w = self.mask_cache["mask_w"]

        eye = unpack_mask(self.mask_cache["eye"][local_idx], h, w)
        inst = unpack_mask(self.mask_cache["inst"][local_idx], h, w)

        x = np.concatenate([rgb, inst[..., None], eye[..., None]], axis=2).astype(np.float32)
        x = torch.from_numpy(x).permute(2, 0, 1)

        mean = torch.tensor([0.485, 0.456, 0.406, 0.0, 0.0]).view(5, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225, 1.0, 1.0]).view(5, 1, 1)
        x = (x - mean) / std

        inst_t = torch.from_numpy(inst[None, ...].astype(np.float32))
        eye_t = torch.from_numpy(eye[None, ...].astype(np.float32))

        return x, torch.tensor(y, dtype=torch.long), inst_t, eye_t


def compute_loss(logits, y, attn, inst_mask, eye_mask, lambda_inst, lambda_eye, lambda_gate):
    ce = F.cross_entropy(logits, y)

    # A_inst: [B,1,28,28]
    a_inst = attn["A_inst"]
    inst_target = downsample_mask(inst_mask, a_inst.shape[-2:])
    loss_inst = bce_prob(a_inst, inst_target)

    # A_eye: [B,1,14,14]
    a_eye = attn["A_eye"]
    eye_target = downsample_mask(eye_mask, a_eye.shape[-2:])
    loss_eye = bce_prob(a_eye, eye_target)

    # Gate_final 和 prior_final 都是 [B,1,28,28]
    gate = attn.get("Gate_final", attn.get("Gate", None))
    prior = attn.get("prior_final", None)
    if gate is not None and prior is not None:
        loss_gate = bce_prob(gate, prior.detach())
    else:
        loss_gate = torch.tensor(0.0, device=logits.device)

    total = ce + lambda_inst * loss_inst + lambda_eye * loss_eye + lambda_gate * loss_gate

    return total, {
        "ce": float(ce.detach().cpu()),
        "inst": float(loss_inst.detach().cpu()),
        "eye": float(loss_eye.detach().cpu()),
        "gate": float(loss_gate.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def run_epoch(model, loader, optimizer, device, train=True, args=None):
    model.train(train)

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not args.no_amp))

    for step, batch in enumerate(loader, 1):
        x, y, inst, eye = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        inst = inst.to(device, non_blocking=True)
        eye = eye.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                logits, attn = model(x, return_attn=True)
                loss, parts = compute_loss(
                    logits, y, attn, inst, eye,
                    args.lambda_inst, args.lambda_eye, args.lambda_gate
                )

                if not torch.isfinite(loss):
                    print("[NONFINITE LOSS DETECTED]", flush=True)
                    print("loss:", loss, flush=True)
                    print("parts:", parts, flush=True)
                    print("logits finite:", torch.isfinite(logits).all().item(), flush=True)
                    print("logits min/max:", float(torch.nan_to_num(logits).min().detach().cpu()), float(torch.nan_to_num(logits).max().detach().cpu()), flush=True)
                    print("y min/max:", int(y.min().detach().cpu()), int(y.max().detach().cpu()), flush=True)
                    raise RuntimeError("Non-finite loss detected. Stop training to avoid corrupt checkpoint.")

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()

        pred = logits.argmax(dim=1)
        total_correct += int((pred == y).sum().item())
        total_n += int(y.numel())
        total_loss += float(loss.detach().cpu()) * int(y.numel())

        if step % args.print_every == 0:
            log(
                f"[{'TRAIN' if train else 'VAL'}] step={step}/{len(loader)} "
                f"loss={parts['total']:.4f} ce={parts['ce']:.4f} "
                f"inst={parts['inst']:.4f} eye={parts['eye']:.4f} gate={parts['gate']:.4f} "
                f"acc={total_correct/max(total_n,1):.4f}"
            )

    return {
        "loss": total_loss / max(total_n, 1),
        "acc": total_correct / max(total_n, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--max_frames_per_class_per_video", type=int, default=80)
    ap.add_argument("--val_max_frames_per_class_per_video", type=int, default=40)
    ap.add_argument("--lambda_inst", type=float, default=0.10)
    ap.add_argument("--lambda_eye", type=float, default=0.08)
    ap.add_argument("--lambda_gate", type=float, default=0.08)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--print_every", type=int, default=50)
    ap.add_argument("--save_dir", type=str, default=str(ROOT / "checkpoints/m10/a3b_v12"))
    ap.add_argument("--exp_name", type=str, default="cat101_a3b_v12_dualgate")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    log("===== M10-A3b v1.2 dual-gate frame classifier training =====")
    log(json.dumps(vars(args), indent=2, ensure_ascii=False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")

    mod = load_old_module()
    model_cls = getattr(mod, "M8IGCoAttnV1Asym")

    num_classes = get_num_classes()
    log(f"num_classes: {num_classes}")

    model = model_cls(
        backbone_name="dm_nfnet_f0",
        num_classes=num_classes,
        pretrained=True,
    ).to(device)

    train_set = Cat101FrameSampleDataset(
        split=args.split,
        mode="train",
        max_frames_per_class_per_video=args.max_frames_per_class_per_video,
        seed=args.seed,
    )
    val_set = Cat101FrameSampleDataset(
        split=args.split,
        mode="test",
        max_frames_per_class_per_video=args.val_max_frames_per_class_per_video,
        seed=args.seed,
    )

    all_train_y = [s[2] for s in train_set.samples]
    all_val_y = [s[2] for s in val_set.samples]
    log(f"train label min/max: {min(all_train_y)} / {max(all_train_y)}")
    log(f"val label min/max: {min(all_val_y)} / {max(all_val_y)}")
    assert min(all_train_y) >= 0 and max(all_train_y) < num_classes
    assert min(all_val_y) >= 0 and max(all_val_y) < num_classes

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = -1.0
    best_path = save_dir / f"{args.exp_name}_split{args.split}_best.pth"
    last_path = save_dir / f"{args.exp_name}_split{args.split}_last.pth"
    log_path = save_dir / f"{args.exp_name}_split{args.split}_train_log.jsonl"

    for epoch in range(1, args.epochs + 1):
        log(f"\n===== EPOCH {epoch}/{args.epochs} =====")
        tr = run_epoch(model, train_loader, optimizer, device, train=True, args=args)

        with torch.no_grad():
            va = run_epoch(model, val_loader, optimizer, device, train=False, args=args)

        record = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
        }
        log("[EPOCH RESULT] " + json.dumps(record, ensure_ascii=False))

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "best_acc": best_acc,
        }
        torch.save(ckpt, last_path)

        if va["acc"] > best_acc:
            best_acc = va["acc"]
            ckpt["best_acc"] = best_acc
            torch.save(ckpt, best_path)
            log(f"[BEST] val_acc={best_acc:.4f}, saved to {best_path}")

    log("[DONE]")
    log(f"best_acc={best_acc:.4f}")
    log(f"best_path={best_path}")
    log(f"last_path={last_path}")


if __name__ == "__main__":
    main()
