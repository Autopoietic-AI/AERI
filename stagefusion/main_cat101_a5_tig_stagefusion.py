# -*- coding: utf-8 -*-
"""
Main for M8-v1.3-GeoIG-Gate.

This follows the same three-file style as previous M7 experiments:
    model_xxx.py
    main_xxx.py
    run_xxx.sh

Dataset alias:
    sics155_m8_v13_geoig_gate_ep6

Base M8 features:
    /root/autodl-tmp/MS-TCN2/data/sics155_m8_v12_dualgate_ep6/features

M7 assets:
    /root/autodl-tmp/MS-TCN2/data/sics155_m7_v1_assets_norm
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path('/root/autodl-tmp/MS-TCN2')
M8_DIR = ROOT_DIR / 'm8'
sys.path.insert(0, str(M8_DIR))
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import random

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"[SEED] A5_GLOBAL_SEED={seed}", flush=True)



CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model_a5_tig_stagefusion import A5TIGDFusionModel


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def fmt_sec(sec):
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def read_mapping(mapping_file):
    actions_dict = {}
    inv_actions = {}

    with open(mapping_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            idx, name = s.split(maxsplit=1)
            idx = int(idx)
            actions_dict[name] = idx
            inv_actions[idx] = name

    return actions_dict, inv_actions, len(actions_dict)


def get_bundle_path(root, dataset, split, mode):
    return os.path.join(root, "data", dataset, "splits", f"{mode}.split{split}.bundle")


def load_bundle(bundle_path):
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    vids = []
    with open(bundle_path, "r", encoding="utf-8") as f:
        for line in f:
            vid = line.strip()
            if vid:
                vids.append(clean_vid(vid))
    return vids



def find_existing_file(base_dir, vid, suffixes):
    """
    Robust file resolver for Cat101.
    Tries case_269.npy / case_269.npz and several old-style aliases.
    """
    candidates = []
    for suf in suffixes:
        candidates.append(os.path.join(base_dir, f"{vid}{suf}"))
        candidates.append(os.path.join(base_dir, f"{vid}.txt{suf}"))

    for c in candidates:
        if os.path.exists(c):
            return c

    return None

def clean_vid(vid):
    return os.path.basename(vid).replace(".txt", "")


def load_gt_sequence(gt_path, actions_dict):
    labels = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            lab = line.strip()
            if lab:
                labels.append(actions_dict[lab])
    return np.array(labels, dtype=np.int64)


def load_feature(feat_path, expected_dim=384):
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Feature not found: {feat_path}")

    feat = np.load(feat_path).astype(np.float32)

    if feat.ndim != 2:
        raise ValueError(f"Feature must be 2D, got shape={feat.shape}: {feat_path}")

    if feat.shape[0] == expected_dim:
        return feat

    if feat.shape[1] == expected_dim:
        return feat.T

    print(
        f"[WARN] feature dim not expected {expected_dim}: shape={feat.shape}, path={feat_path}",
        flush=True
    )
    return feat


def load_m7_asset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"M7 asset not found: {path}")

    x = np.load(path, allow_pickle=True)

    required = [
        "geo",
        "aux",
        "presence_mask",
        "cls_proxy",
        "cls_proxy_mask",
        "traj",
        "eye_center",
        "eye_radius",
        "inst_tips",
    ]

    for k in required:
        if k not in x:
            raise KeyError(f"Missing key '{k}' in asset: {path}")

    return {
        "geo": x["geo"].astype(np.float32),
        "aux": x["aux"].astype(np.float32),
        "presence_mask": x["presence_mask"].astype(np.float32),
        "cls_proxy": x["cls_proxy"].astype(np.float32),
        "cls_proxy_mask": x["cls_proxy_mask"].astype(np.float32),
        "traj": x["traj"].astype(np.float32),
        "eye_center": x["eye_center"].astype(np.float32),
        "eye_radius": x["eye_radius"].astype(np.float32),
        "inst_tips": x["inst_tips"].astype(np.float32),
    }


def align_all(m8_feat, asset, target):
    T_feat = m8_feat.shape[1]
    T_asset = asset["geo"].shape[-1]
    T_gt = len(target)

    L = min(T_feat, T_asset, T_gt)

    m8_feat = m8_feat[:, :L]

    for k in asset:
        if asset[k].ndim == 1:
            asset[k] = asset[k][:L]
        elif asset[k].ndim == 2:
            asset[k] = asset[k][:, :L]
        elif asset[k].ndim == 3:
            asset[k] = asset[k][:, :, :L]
        elif asset[k].ndim == 4:
            asset[k] = asset[k][:, :, :, :L]

    target = target[:L]

    return m8_feat, asset, target, L


def to_tensor_batch(x, device):
    return torch.from_numpy(x).unsqueeze(0).to(device)


def sequence_loss(outputs, target, ce, mse, smooth_w=0.15):
    num_classes = outputs[0].shape[1]
    loss = 0.0

    for p in outputs:
        loss += ce(
            p.transpose(2, 1).contiguous().view(-1, num_classes),
            target.contiguous().view(-1)
        )

        loss += smooth_w * torch.mean(
            torch.clamp(
                mse(
                    torch.log_softmax(p[:, :, 1:], dim=1),
                    torch.log_softmax(p.detach()[:, :, :-1], dim=1)
                ),
                min=0,
                max=16
            )
        )

    return loss



def load_maskshape_asset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask-shape asset not found: {path}")
    x = np.load(path, allow_pickle=True)
    required = ["mask_shape", "mask_shape_valid"]
    for k in required:
        if k not in x.files:
            raise KeyError(f"Missing key {k} in mask-shape asset: {path}")
    return {
        "mask_shape": x["mask_shape"].astype(np.float32),
        "mask_shape_valid": x["mask_shape_valid"].astype(np.float32),
    }


def align_maskshape(maskshape, T):
    return {k: v[..., :T] for k, v in maskshape.items()}

def build_model(m8_feat_dim, num_classes, args, device):
    model = A5TIGDFusionModel(
        m8_feat_dim=m8_feat_dim,
        num_classes=num_classes,
        num_stages=args.num_stages,
        num_layers=args.num_layers,
        num_f_maps=args.num_f_maps,
        main_dim=args.main_dim,
        guide_dim=args.guide_dim,
        beta_max=args.beta_max,
    ).to(device)

    return model


def train(args):
    print("[ENTRY] train()", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = args.root
    dataset = args.dataset
    split = args.split

    mapping_file = os.path.join(root, "data", dataset, "mapping.txt")
    gt_dir = os.path.join(root, "data", dataset, "groundTruth")

    model_dir = os.path.join(root, "models", f"tecno_{dataset}", f"split_{split}")
    ensure_dir(model_dir)

    actions_dict, inv_actions, num_classes = read_mapping(mapping_file)
    train_vids = load_bundle(get_bundle_path(root, dataset, split, "train"))

    first_vid = clean_vid(train_vids[0])
    first_feat = load_feature(
        os.path.join(args.m8_feature_dir, f"{first_vid}.npy"),
        expected_dim=args.expected_dim
    )
    m8_feat_dim = first_feat.shape[0]

    model = build_model(m8_feat_dim, num_classes, args, device)

    ce = nn.CrossEntropyLoss(ignore_index=-100)
    mse = nn.MSELoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_loss = 1e18
    total_start = time.time()

    print("=" * 80, flush=True)
    print("[M8-v1.3-GeoIG-Gate]", flush=True)
    print(f"[INFO] device         = {device}", flush=True)
    print(f"[INFO] dataset        = {dataset}", flush=True)
    print(f"[INFO] split          = {split}", flush=True)
    print(f"[INFO] m8_feature_dir = {args.m8_feature_dir}", flush=True)
    print(f"[INFO] m7_asset_dir   = {args.m7_asset_dir}", flush=True)
    print(f"[INFO] m8_feat_dim    = {m8_feat_dim}", flush=True)
    print(f"[INFO] num_classes    = {num_classes}", flush=True)
    print(f"[INFO] num_epochs     = {args.num_epochs}", flush=True)
    print(f"[INFO] smooth_w       = {args.smooth_w}", flush=True)
    print(f"[INFO] beta_max       = {args.beta_max}", flush=True)
    print(f"[INFO] gate_l1        = {args.gate_l1}", flush=True)
    print("=" * 80, flush=True)

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        valid = 0
        gate_sum = 0.0
        epoch_start = time.time()

        print("\n" + "#" * 80, flush=True)
        print(f"[EPOCH START] {epoch}/{args.num_epochs}", flush=True)
        print("#" * 80, flush=True)

        for vid_idx, vid in enumerate(train_vids, 1):
            base = clean_vid(vid)

            feat_path = os.path.join(args.m8_feature_dir, f"{base}.npy")
            asset_path = os.path.join(args.m7_asset_dir, f"{base}.npz")
            gt_path = os.path.join(gt_dir, f"{base}.txt")

            if not (os.path.exists(feat_path) and os.path.exists(asset_path) and os.path.exists(gt_path)):
                print(f"[WARN] missing file for {vid}", flush=True)
                continue

            m8_feat = load_feature(feat_path, expected_dim=args.expected_dim)
            asset = load_m7_asset(asset_path)
            maskshape_path = os.path.join(args.maskshape_dir, f"{base}.npz")
            maskshape = load_maskshape_asset(maskshape_path)
            target = load_gt_sequence(gt_path, actions_dict)

            m8_feat, asset, target, T = align_all(m8_feat, asset, target)
            maskshape = align_maskshape(maskshape, T)

            m8_feat_t = to_tensor_batch(m8_feat, device)
            target_t = torch.from_numpy(target).unsqueeze(0).to(device)

            geo = to_tensor_batch(asset["geo"], device)
            aux = to_tensor_batch(asset["aux"], device)
            presence_mask = to_tensor_batch(asset["presence_mask"], device)
            cls_proxy = to_tensor_batch(asset["cls_proxy"], device)
            cls_proxy_mask = to_tensor_batch(asset["cls_proxy_mask"], device)
            traj = to_tensor_batch(asset["traj"], device)
            eye_center = to_tensor_batch(asset["eye_center"], device)
            eye_radius = to_tensor_batch(asset["eye_radius"], device)
            inst_tips = to_tensor_batch(asset["inst_tips"], device)
            mask_shape = to_tensor_batch(maskshape["mask_shape"], device)
            mask_shape_valid = to_tensor_batch(maskshape["mask_shape_valid"], device)

            optimizer.zero_grad()

            outputs, extra = model(
                m8_feat=m8_feat_t,
                geo=geo,
                aux=aux,
                presence_mask=presence_mask,
                cls_proxy=cls_proxy,
                cls_proxy_mask=cls_proxy_mask,
                traj=traj,
                eye_center=eye_center,
                eye_radius=eye_radius,
                inst_tips=inst_tips,
                mask_shape=mask_shape,
                mask_shape_valid=mask_shape_valid,
                return_extra=True,
            )

            loss = sequence_loss(outputs, target_t, ce, mse, smooth_w=args.smooth_w)
            loss = loss + args.gate_l1 * extra["gate"].mean()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            valid += 1
            gate_sum += float(extra["gate"].mean().detach().cpu())

            if vid_idx == 1 or vid_idx % 5 == 0 or vid_idx == len(train_vids):
                print(
                    f"[TRAIN][Epoch {epoch}/{args.num_epochs}] "
                    f"{vid_idx}/{len(train_vids)} {vid} | "
                    f"T={T} | loss={loss.item():.4f} | "
                    f"gate_mean={extra['gate'].mean().item():.5f}",
                    flush=True
                )

        avg_loss = epoch_loss / max(valid, 1)
        avg_gate = gate_sum / max(valid, 1)
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - total_start

        print("-" * 80, flush=True)
        print(
            f"[EPOCH END] {epoch}/{args.num_epochs} | "
            f"epoch_loss={epoch_loss:.4f} | avg_loss={avg_loss:.4f} | "
            f"avg_gate={avg_gate:.5f} | valid={valid} | "
            f"epoch_time={fmt_sec(epoch_time)} | elapsed={fmt_sec(elapsed)}",
            flush=True
        )
        print("-" * 80, flush=True)

        torch.save(model.state_dict(), os.path.join(model_dir, f"epoch-{epoch}.model"))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "best.model"))
            print(f"[BEST] updated best.model | best_loss={best_loss:.4f}", flush=True)


def predict(args):
    print("[ENTRY] predict()", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = args.root
    dataset = args.dataset
    split = args.split

    mapping_file = os.path.join(root, "data", dataset, "mapping.txt")
    gt_dir = os.path.join(root, "data", dataset, "groundTruth")

    model_dir = os.path.join(root, "models", f"tecno_{dataset}", f"split_{split}")
    result_dir = os.path.join(root, "results", f"tecno_{dataset}", f"split_{split}")
    guide_dir = os.path.join(root, "artifacts", "m10_a4_geoig_gate", args.dataset, f"split_{split}", "guide_npz")

    ensure_dir(result_dir)
    ensure_dir(guide_dir)

    actions_dict, inv_actions, num_classes = read_mapping(mapping_file)
    test_vids = load_bundle(get_bundle_path(root, dataset, split, "test"))

    first_vid = clean_vid(test_vids[0])
    first_feat = load_feature(
        os.path.join(args.m8_feature_dir, f"{first_vid}.npy"),
        expected_dim=args.expected_dim
    )
    m8_feat_dim = first_feat.shape[0]

    model = build_model(m8_feat_dim, num_classes, args, device)

    ckpt = os.path.join(model_dir, f"epoch-{args.num_epochs}.model")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(model_dir, "best.model")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print("=" * 80, flush=True)
    print("[PREDICT M8-v1.3-GeoIG-Gate]", flush=True)
    print(f"[INFO] checkpoint     = {ckpt}", flush=True)
    print(f"[INFO] result_dir     = {result_dir}", flush=True)
    print(f"[INFO] guide_dir      = {guide_dir}", flush=True)
    print(f"[INFO] m8_feature_dir = {args.m8_feature_dir}", flush=True)
    print(f"[INFO] m7_asset_dir   = {args.m7_asset_dir}", flush=True)
    print(f"[INFO] test videos    = {len(test_vids)}", flush=True)
    print("=" * 80, flush=True)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    with torch.no_grad():
        for vid_idx, vid in enumerate(test_vids, 1):
            base = clean_vid(vid)

            feat_path = os.path.join(args.m8_feature_dir, f"{base}.npy")
            asset_path = os.path.join(args.m7_asset_dir, f"{base}.npz")
            gt_path = os.path.join(gt_dir, f"{base}.txt")

            if not (os.path.exists(feat_path) and os.path.exists(asset_path) and os.path.exists(gt_path)):
                print(f"[WARN] missing file for {vid}", flush=True)
                continue

            m8_feat = load_feature(feat_path, expected_dim=args.expected_dim)
            asset = load_m7_asset(asset_path)
            maskshape_path = os.path.join(args.maskshape_dir, f"{base}.npz")
            maskshape = load_maskshape_asset(maskshape_path)

            with open(gt_path, "r", encoding="utf-8") as f:
                gt_lines = [line.strip() for line in f if line.strip()]
            target = np.array([actions_dict[x] for x in gt_lines], dtype=np.int64)

            m8_feat, asset, target, T = align_all(m8_feat, asset, target)
            maskshape = align_maskshape(maskshape, T)

            m8_feat_t = to_tensor_batch(m8_feat, device)

            geo = to_tensor_batch(asset["geo"], device)
            aux = to_tensor_batch(asset["aux"], device)
            presence_mask = to_tensor_batch(asset["presence_mask"], device)
            cls_proxy = to_tensor_batch(asset["cls_proxy"], device)
            cls_proxy_mask = to_tensor_batch(asset["cls_proxy_mask"], device)
            traj = to_tensor_batch(asset["traj"], device)
            eye_center = to_tensor_batch(asset["eye_center"], device)
            eye_radius = to_tensor_batch(asset["eye_radius"], device)
            inst_tips = to_tensor_batch(asset["inst_tips"], device)
            mask_shape = to_tensor_batch(maskshape["mask_shape"], device)
            mask_shape_valid = to_tensor_batch(maskshape["mask_shape_valid"], device)

            outputs, extra = model(
                m8_feat=m8_feat_t,
                geo=geo,
                aux=aux,
                presence_mask=presence_mask,
                cls_proxy=cls_proxy,
                cls_proxy_mask=cls_proxy_mask,
                traj=traj,
                eye_center=eye_center,
                eye_radius=eye_radius,
                inst_tips=inst_tips,
                mask_shape=mask_shape,
                mask_shape_valid=mask_shape_valid,
                return_extra=True,
            )

            pred = torch.argmax(outputs[-1], dim=1).squeeze(0).cpu().numpy()
            pred_labels = [inv_actions[int(p)] for p in pred]

            out_path = os.path.join(result_dir, vid)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("### Frame level recognition: ###\n")
                f.write(" ".join(pred_labels) + "\n")


            def to_np_extra(extra_dict, key):
                v = extra_dict.get(key, None)
                if v is None:
                    return np.array([], dtype=np.float32)
                if hasattr(v, "detach"):
                    v = v.detach().cpu().numpy()
                return np.asarray(v)

            np.savez_compressed(
                os.path.join(guide_dir, f"{base}.npz"),
                gate=extra["gate"].squeeze(0).cpu().numpy(),
                dfusion_gate=to_np_extra(extra, "dfusion_gate"),
                tig_attn=to_np_extra(extra, "tig_attn"),
                dfusion_rel=to_np_extra(extra, "dfusion_rel"),
                pred=pred,
                target=target,
            )

            print(
                f"[OK] predicted {vid_idx}/{len(test_vids)}: {vid} | "
                f"T={len(pred_labels)} | gate_mean={extra['gate'].mean().item():.5f}",
                flush=True
            )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, default="/root/autodl-tmp/MS-TCN2")
    ap.add_argument("--dataset", type=str, default="sics155_m8_v13_geoig_gate_ep6")
    ap.add_argument("--m8_feature_dir", type=str, default="/root/autodl-tmp/MS-TCN2/data/sics155_m8_v12_dualgate_ep6/features")
    ap.add_argument("--m7_asset_dir", type=str, default="/root/autodl-tmp/MS-TCN2/data/sics155_m7_v1_assets_norm")
    ap.add_argument("--maskshape_dir", type=str, default="/root/autodl-tmp/MS-TCN2/data/cataract101_m10_a5_maskshape_assets_v1_safe/npz")
    ap.add_argument("--split", type=int, default=1)

    ap.add_argument("--action", type=str, choices=["train", "predict"], required=True)
    ap.add_argument("--num_epochs", type=int, default=100)
    ap.add_argument("--learning_rate", type=float, default=5e-4)

    ap.add_argument("--num_stages", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=10)
    ap.add_argument("--num_f_maps", type=int, default=64)

    ap.add_argument("--main_dim", type=int, default=256)
    ap.add_argument("--guide_dim", type=int, default=64)
    ap.add_argument("--beta_max", type=float, default=0.20)
    ap.add_argument("--gate_l1", type=float, default=0.001)

    ap.add_argument("--smooth_w", type=float, default=0.15)
    ap.add_argument("--expected_dim", type=int, default=384)

    args = ap.parse_args()

    seed = int(os.environ.get("A5_GLOBAL_SEED", "2026"))
    set_global_seed(seed)

    if args.action == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()