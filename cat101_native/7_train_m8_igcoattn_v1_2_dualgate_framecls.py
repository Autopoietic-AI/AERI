#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Auto-generated M8-X5-IGCoAttn-v1.2-dual-gate version.

Generated from:
3_train_m8_igcoattn_v1_asym_framecls.py

Main idea:
- A_inst uses medium instrument target.
- Gate_wide learns wide interaction context.
- Gate_thin learns compact interaction core.
- Gate_final fuses wide/thin gates for final feature enhancement.
"""


"""
M8-X5-IGCoAttn-v1.2-dual-gate frame-level classifier

核心目标：
1. 5-channel input: RGB + instrument mask + eye mask
2. NFNet backbone: dm_nfnet_f0
3. Asymmetric branch:
   - instrument branch uses shallow/mid feature
   - eye branch uses deeper feature
4. Explicit alignment loss:
   - A_inst aligns to dilated instrument mask
   - A_eye aligns to eye mask
   - Gate aligns to interaction prior
5. Presence gate suppresses instrument/gate branch when no instrument exists.

建议路径：
/root/autodl-tmp/MS-TCN2/m8/3_train_m8_igcoattn_v1_2_dualgate_framecls.py
"""

import os
import glob
import json
import math
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import timm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# =========================
# Utils
# =========================

def set_seed(seed: int = 3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_mapping(mapping_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    mapping.txt format:
    0 background
    1 peritomy
    ...
    """
    name_to_id = {}
    id_to_name = {}

    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            cls_name = parts[1]
            id_to_name[cls_id] = cls_name
            name_to_id[cls_name] = cls_id

    return name_to_id, id_to_name


def read_bundle(bundle_path: str) -> List[str]:
    """
    MS-TCN bundle usually contains:
    video_name.txt
    or
    video_id
    """
    video_ids = []
    if not os.path.exists(bundle_path):
        print(f"[WARN] bundle not found: {bundle_path}")
        return video_ids

    with open(bundle_path, "r", encoding="utf-8") as f:
        for line in f:
            x = line.strip()
            if not x:
                continue
            x = os.path.basename(x)
            x = x.replace(".txt", "")
            video_ids.append(x)

    return sorted(list(set(video_ids)))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_gray_mask(path: str, size_hw: Tuple[int, int]) -> np.ndarray:
    """
    Return float mask in [0,1], shape H,W.
    """
    h, w = size_hw
    if path is None or (not os.path.exists(path)):
        return np.zeros((h, w), dtype=np.float32)

    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((h, w), dtype=np.float32)

    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.float32)
    return mask


def combine_inst_masks(inst_root: str, video_id: str, phase_name: str, frame_stem: str, size_hw: Tuple[int, int]) -> np.ndarray:
    """
    Instrument masks may be:
    frame_00747_inst0_mask.png
    frame_00747_inst1_mask.png
    ...
    """
    h, w = size_hw
    pattern = os.path.join(inst_root, video_id, phase_name, f"{frame_stem}_inst*_mask.png")
    paths = sorted(glob.glob(pattern))

    if len(paths) == 0:
        return np.zeros((h, w), dtype=np.float32)

    mask_all = np.zeros((h, w), dtype=np.float32)
    for p in paths:
        m = load_gray_mask(p, size_hw)
        mask_all = np.maximum(mask_all, m)

    return mask_all.astype(np.float32)


def dilate_mask_tensor(mask: torch.Tensor, k: int = 9) -> torch.Tensor:
    """
    mask: B,1,H,W in [0,1]
    use max_pool as dilation.
    """
    if k <= 1:
        return mask
    pad = k // 2
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)


def make_interaction_prior(inst_mask: torch.Tensor, eye_mask: torch.Tensor, k_inst: int = 13, k_eye: int = 9) -> torch.Tensor:
    """
    inst_mask, eye_mask: B,1,H,W
    prior = dilate(inst) intersection dilate(eye)
    fallback: if no intersection, use dilated inst.
    """
    inst_d = dilate_mask_tensor(inst_mask, k_inst)
    eye_d = dilate_mask_tensor(eye_mask, k_eye)

    inter = inst_d * eye_d
    inter_area = inter.flatten(1).sum(dim=1, keepdim=True).view(-1, 1, 1, 1)

    fallback = inst_d
    prior = torch.where(inter_area > 1.0, inter, fallback)
    prior = prior.clamp(0, 1)
    return prior


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred/target: B,1,H,W
    """
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)

    inter = (pred * target).flatten(1).sum(dim=1)
    denom = pred.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()

def bce_dice_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
    """
    pred/target: B,1,H,W

    注意：
    pred 已经经过 Sigmoid，是概率图。
    在 AMP autocast 下，torch.nn.functional.binary_cross_entropy 不安全，
    所以这里强制关闭 autocast，并转为 float32 计算 BCE + Dice。
    """
    device_type = "cuda" if pred.is_cuda else "cpu"

    with torch.amp.autocast(device_type=device_type, enabled=False):
        pred_f = pred.float().clamp(1e-5, 1.0 - 1e-5)
        target_f = target.float().clamp(0.0, 1.0)

        bce = F.binary_cross_entropy(pred_f, target_f, reduction="none")

        if weight is not None:
            weight_f = weight.float()
            bce = bce * weight_f

        bce = bce.mean()

        dloss = dice_loss(pred_f, target_f)

        return bce + dloss


# =========================
# Dataset
# =========================

class SICSFrameDataset(Dataset):
    def __init__(
        self,
        frame_root: str,
        eye_root: str,
        inst_root: str,
        mapping_path: str,
        video_ids: List[str],
        img_size: int = 224,
        resize_size: int = 256,
        max_frames_per_phase: int = -1,
        is_train: bool = True,
    ):
        self.frame_root = frame_root
        self.eye_root = eye_root
        self.inst_root = inst_root
        self.img_size = img_size
        self.resize_size = resize_size
        self.is_train = is_train

        self.name_to_id, self.id_to_name = read_mapping(mapping_path)
        self.video_ids = set(video_ids)

        self.samples = []
        self._collect_samples(max_frames_per_phase=max_frames_per_phase)

        if is_train:
            self.rgb_tf = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            self.mask_tf = transforms.Compose([
                transforms.Resize((resize_size, resize_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1),
                                             interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ])
        else:
            self.rgb_tf = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            self.mask_tf = transforms.Compose([
                transforms.Resize((resize_size, resize_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
            ])

        print(f"[Dataset] train={is_train}, videos={len(self.video_ids)}, samples={len(self.samples)}")

    def _collect_samples(self, max_frames_per_phase: int):
        for video_id in sorted(os.listdir(self.frame_root)):
            if video_id not in self.video_ids:
                continue

            video_dir = os.path.join(self.frame_root, video_id)
            if not os.path.isdir(video_dir):
                continue

            for phase_name in sorted(os.listdir(video_dir)):
                phase_dir = os.path.join(video_dir, phase_name)
                if not os.path.isdir(phase_dir):
                    continue
                if phase_name not in self.name_to_id:
                    continue

                frame_paths = sorted(glob.glob(os.path.join(phase_dir, "*.jpg")))
                frame_paths += sorted(glob.glob(os.path.join(phase_dir, "*.png")))
                frame_paths = sorted(list(set(frame_paths)))

                if max_frames_per_phase is not None and max_frames_per_phase > 0:
                    if len(frame_paths) > max_frames_per_phase:
                        idx = np.linspace(0, len(frame_paths) - 1, max_frames_per_phase).astype(int)
                        frame_paths = [frame_paths[i] for i in idx]

                cls_id = self.name_to_id[phase_name]

                for fp in frame_paths:
                    stem = Path(fp).stem
                    self.samples.append({
                        "frame_path": fp,
                        "video_id": video_id,
                        "phase_name": phase_name,
                        "frame_stem": stem,
                        "label": cls_id,
                    })

    def __len__(self):
        return len(self.samples)

    def _load_raw(self, item):
        frame_path = item["frame_path"]
        video_id = item["video_id"]
        phase_name = item["phase_name"]
        frame_stem = item["frame_stem"]

        img = Image.open(frame_path).convert("RGB")
        w, h = img.size

        eye_path = os.path.join(self.eye_root, video_id, phase_name, f"{frame_stem}_eye_mask.png")
        eye_mask = load_gray_mask(eye_path, (h, w))

        inst_mask = combine_inst_masks(
            self.inst_root, video_id, phase_name, frame_stem, (h, w)
        )

        eye_pil = Image.fromarray((eye_mask * 255).astype(np.uint8))
        inst_pil = Image.fromarray((inst_mask * 255).astype(np.uint8))

        return img, inst_pil, eye_pil

    def __getitem__(self, idx):
        item = self.samples[idx]

        # 为了保证 RGB 和 mask 做相同随机变换，这里用同一个随机种子。
        seed = np.random.randint(0, 2 ** 31 - 1)

        img, inst_pil, eye_pil = self._load_raw(item)

        random.seed(seed)
        torch.manual_seed(seed)
        rgb = self.rgb_tf(img)

        random.seed(seed)
        torch.manual_seed(seed)
        inst = self.mask_tf(inst_pil)
        inst = (inst > 0.5).float()

        random.seed(seed)
        torch.manual_seed(seed)
        eye = self.mask_tf(eye_pil)
        eye = (eye > 0.5).float()

        x5 = torch.cat([rgb, inst, eye], dim=0)

        inst_area = inst.flatten().mean()
        presence = (inst_area > 0.0005).float()

        return {
            "x": x5,
            "rgb": rgb,
            "inst_mask": inst,
            "eye_mask": eye,
            "presence": presence,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "meta": item,
        }


# =========================
# Model
# =========================

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)



class M8IGCoAttnV1Asym(nn.Module):
    """
    M8-X5-IGCoAttn-v1.2-dual-gate.

    设计思想：
    1. A_inst 使用中等尺度器械监督，避免 v1 过粗、v1.1 过窄；
    2. Gate_wide 建模宽交互上下文；
    3. Gate_thin 建模紧凑交互核心；
    4. Gate_final 学习融合 wide/thin，作为最终特征增强门控。

    注意：
    为了兼容已有可视化脚本，类名仍保持 M8IGCoAttnV1Asym。
    """
    def __init__(
        self,
        backbone_name: str = "dm_nfnet_f0",
        num_classes: int = 19,
        pretrained: bool = True,
        mid_ch: int = 128,
        out_indices=(1, 2, 3),
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=5,
            out_indices=out_indices,
        )

        chs = self.backbone.feature_info.channels()
        if len(chs) < 3:
            raise RuntimeError(f"Need at least 3 feature scales, got channels={chs}")

        c0, c1, c2 = chs[0], chs[1], chs[2]
        self.c0, self.c1, self.c2 = c0, c1, c2

        # Instrument branch: S0 + S1, output at S1 resolution
        self.s0_proj = ConvBNAct(c0, mid_ch, k=1, p=0)
        self.s1_proj = ConvBNAct(c1, mid_ch, k=1, p=0)

        self.inst_att = nn.Sequential(
            ConvBNAct(mid_ch * 2 + 1, mid_ch, k=3, p=1),
            nn.Conv2d(mid_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Eye branch: S2, output at S2 then upsample to S1
        self.s2_proj = ConvBNAct(c2, mid_ch, k=1, p=0)

        self.eye_att = nn.Sequential(
            ConvBNAct(mid_ch + 1, mid_ch, k=3, p=1),
            nn.Conv2d(mid_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Gate_wide: wider interaction context
        self.gate_wide_net = nn.Sequential(
            ConvBNAct(mid_ch * 2 + 5, mid_ch, k=3, p=1),
            ConvBNAct(mid_ch, mid_ch, k=3, p=1),
            nn.Conv2d(mid_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Gate_thin: compact interaction core
        self.gate_thin_net = nn.Sequential(
            ConvBNAct(mid_ch * 2 + 5, mid_ch, k=3, p=1),
            ConvBNAct(mid_ch, mid_ch, k=3, p=1),
            nn.Conv2d(mid_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Gate_final: learnable fusion of wide/thin gates
        # Inputs:
        # s1p, s2_up = 2*mid_ch
        # a_inst, a_eye, gate_wide, gate_thin, prior_wide, prior_thin = 6 channels
        self.gate_fusion_net = nn.Sequential(
            ConvBNAct(mid_ch * 2 + 6, mid_ch, k=3, p=1),
            ConvBNAct(mid_ch, mid_ch, k=3, p=1),
            nn.Conv2d(mid_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # residual strength, initialized as near identity
        self.gamma_inst = nn.Parameter(torch.tensor(0.0))
        self.gamma_eye = nn.Parameter(torch.tensor(0.0))
        self.gamma_gate = nn.Parameter(torch.tensor(0.0))

        self.classifier = nn.Sequential(
            nn.LayerNorm(mid_ch * 3),
            nn.Linear(mid_ch * 3, mid_ch),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(mid_ch, num_classes),
        )

    def forward(self, x, inst_mask=None, eye_mask=None, return_attn=False):
        """
        x: B,5,H,W
        inst_mask/eye_mask: B,1,H,W
        """
        if inst_mask is None:
            inst_mask = x[:, 3:4]
        if eye_mask is None:
            eye_mask = x[:, 4:5]

        feats = self.backbone(x)
        s0, s1, s2 = feats[0], feats[1], feats[2]

        s0p = self.s0_proj(s0)
        s1p = self.s1_proj(s1)
        s2p = self.s2_proj(s2)

        h1, w1 = s1p.shape[-2:]
        h2, w2 = s2p.shape[-2:]

        s0p_up = F.interpolate(s0p, size=(h1, w1), mode="bilinear", align_corners=False)
        s2p_up = F.interpolate(s2p, size=(h1, w1), mode="bilinear", align_corners=False)

        inst_s1 = F.interpolate(inst_mask, size=(h1, w1), mode="nearest")
        eye_s1 = F.interpolate(eye_mask, size=(h1, w1), mode="nearest")

        # 中等器械目标：比 v1 细，比 v1.1 宽
        inst_d_s1 = dilate_mask_tensor(inst_s1, k=7)
        eye_d_s1 = dilate_mask_tensor(eye_s1, k=7)

        # 三种 prior
        prior_wide = make_interaction_prior(inst_s1, eye_s1, k_inst=13, k_eye=9)
        prior_thin = make_interaction_prior(inst_s1, eye_s1, k_inst=7, k_eye=5)
        prior_final = make_interaction_prior(inst_s1, eye_s1, k_inst=9, k_eye=7)

        # presence gate
        presence = (inst_mask.flatten(1).mean(dim=1) > 0.0005).float().view(-1, 1, 1, 1)

        # Instrument attention
        inst_input = torch.cat([s0p_up, s1p, inst_d_s1], dim=1)
        a_inst = self.inst_att(inst_input)
        a_inst_eff = a_inst * presence

        # Eye attention
        eye_s2 = F.interpolate(eye_mask, size=(h2, w2), mode="nearest")
        eye_input = torch.cat([s2p, eye_s2], dim=1)
        a_eye = self.eye_att(eye_input)
        a_eye_s1 = F.interpolate(a_eye, size=(h1, w1), mode="bilinear", align_corners=False)

        prior_wide_eff = prior_wide * presence
        prior_thin_eff = prior_thin * presence
        prior_final_eff = prior_final * presence

        # Gate wide
        gate_wide_input = torch.cat([
            s1p,
            s2p_up,
            a_inst_eff,
            a_eye_s1,
            inst_d_s1,
            eye_d_s1,
            prior_wide_eff,
        ], dim=1)
        gate_wide = self.gate_wide_net(gate_wide_input)
        gate_wide_eff = gate_wide * presence

        # Gate thin
        gate_thin_input = torch.cat([
            s1p,
            s2p_up,
            a_inst_eff,
            a_eye_s1,
            inst_d_s1,
            eye_d_s1,
            prior_thin_eff,
        ], dim=1)
        gate_thin = self.gate_thin_net(gate_thin_input)
        gate_thin_eff = gate_thin * presence

        # Gate final fusion
        gate_final_input = torch.cat([
            s1p,
            s2p_up,
            a_inst_eff,
            a_eye_s1,
            gate_wide_eff,
            gate_thin_eff,
            prior_wide_eff,
            prior_thin_eff,
        ], dim=1)
        gate_final = self.gate_fusion_net(gate_final_input)
        gate_final_eff = gate_final * presence

        # Enhanced features
        s1_enh = s1p + torch.tanh(self.gamma_inst) * s1p * a_inst_eff
        s2_enh = s2p + torch.tanh(self.gamma_eye) * s2p * a_eye

        # Final gate modulates mid-level interaction feature
        s1_gate = s1_enh + torch.tanh(self.gamma_gate) * s1_enh * gate_final_eff

        token_inst = F.adaptive_avg_pool2d(s1_gate, 1).flatten(1)
        token_eye = F.adaptive_avg_pool2d(s2_enh, 1).flatten(1)
        token_global = F.adaptive_avg_pool2d(s2p, 1).flatten(1)

        token = torch.cat([token_inst, token_eye, token_global], dim=1)
        logits = self.classifier(token)

        if return_attn:
            return logits, {
                "A_inst": a_inst,
                "A_eye": a_eye,
                "A_eye_s1": a_eye_s1,

                # 兼容旧可视化脚本：Gate 指向 final gate
                "Gate": gate_final,
                "Gate_eff": gate_final_eff,
                "prior_s1": prior_final,

                # v1.2 新增可视化字段
                "Gate_wide": gate_wide,
                "Gate_wide_eff": gate_wide_eff,
                "Gate_thin": gate_thin,
                "Gate_thin_eff": gate_thin_eff,
                "Gate_final": gate_final,
                "Gate_final_eff": gate_final_eff,

                "prior_wide": prior_wide,
                "prior_thin": prior_thin,
                "prior_final": prior_final,

                "presence": presence,
                "inst_s1": inst_s1,
                "eye_s1": eye_s1,
                "inst_d_s1": inst_d_s1,
                "eye_d_s1": eye_d_s1,

                "gamma_inst": torch.tanh(self.gamma_inst).detach(),
                "gamma_eye": torch.tanh(self.gamma_eye).detach(),
                "gamma_gate": torch.tanh(self.gamma_gate).detach(),
            }

        return logits


# =========================
# Train / Val
# =========================


def run_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
    train=True,
    lambda_inst=0.10,
    lambda_eye=0.08,
    lambda_gate=0.08,
    amp=True,
):
    """
    v1.2-dual-gate loss:

    L = CE
        + lambda_inst * L_inst
        + lambda_eye  * L_eye
        + lambda_gate * (0.35*L_wide + 0.45*L_thin + 0.20*L_final)

    解释：
    - L_wide 保证交互上下文不漏；
    - L_thin 保证核心区域紧凑；
    - L_final 让最终 Gate 保持平衡。
    """
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_inst = 0.0
    total_eye = 0.0
    total_gate = 0.0
    total_correct = 0
    total_num = 0

    for step, batch in enumerate(loader):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        inst = batch["inst_mask"].to(device, non_blocking=True)
        eye = batch["eye_mask"].to(device, non_blocking=True)
        presence = batch["presence"].to(device, non_blocking=True).view(-1, 1, 1, 1)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp):
                logits, attn = model(x, inst, eye, return_attn=True)

                ce = F.cross_entropy(logits, y)

                a_inst = attn["A_inst"]
                a_eye = attn["A_eye"]

                gate_wide = attn["Gate_wide"]
                gate_thin = attn["Gate_thin"]
                gate_final = attn["Gate_final"]

                h1, w1 = a_inst.shape[-2:]
                h2, w2 = a_eye.shape[-2:]

                inst_s1 = F.interpolate(inst, size=(h1, w1), mode="nearest")
                eye_s1 = F.interpolate(eye, size=(h1, w1), mode="nearest")
                eye_s2 = F.interpolate(eye, size=(h2, w2), mode="nearest")

                # medium instrument target
                inst_target = dilate_mask_tensor(inst_s1, k=7)
                eye_target = eye_s2

                # three gate targets
                target_wide = make_interaction_prior(inst_s1, eye_s1, k_inst=13, k_eye=9)
                target_thin = make_interaction_prior(inst_s1, eye_s1, k_inst=7, k_eye=5)
                target_final = make_interaction_prior(inst_s1, eye_s1, k_inst=9, k_eye=7)

                loss_inst = bce_dice_loss(a_inst, inst_target, weight=presence)
                loss_eye = bce_dice_loss(a_eye, eye_target, weight=None)

                loss_wide = bce_dice_loss(gate_wide, target_wide, weight=presence)
                loss_thin = bce_dice_loss(gate_thin, target_thin, weight=presence)
                loss_final = bce_dice_loss(gate_final, target_final, weight=presence)

                loss_gate = 0.35 * loss_wide + 0.45 * loss_thin + 0.20 * loss_final

                loss = ce + lambda_inst * loss_inst + lambda_eye * loss_eye + lambda_gate * loss_gate

            if train:
                optimizer.zero_grad(set_to_none=True)
                if amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

        pred = logits.argmax(dim=1)
        correct = (pred == y).sum().item()
        bs = x.size(0)

        total_loss += loss.item() * bs
        total_ce += ce.item() * bs
        total_inst += loss_inst.item() * bs
        total_eye += loss_eye.item() * bs
        total_gate += loss_gate.item() * bs
        total_correct += correct
        total_num += bs

        if train and step % 50 == 0:
            print(
                f"[Epoch {epoch:03d}][Step {step:04d}/{len(loader)}] "
                f"loss={loss.item():.4f} ce={ce.item():.4f} "
                f"inst={loss_inst.item():.4f} eye={loss_eye.item():.4f} "
                f"gate={loss_gate.item():.4f} "
                f"acc={correct / max(bs, 1):.4f}"
            )

    metrics = {
        "loss": total_loss / max(total_num, 1),
        "ce": total_ce / max(total_num, 1),
        "loss_inst": total_inst / max(total_num, 1),
        "loss_eye": total_eye / max(total_num, 1),
        "loss_gate": total_gate / max(total_num, 1),
        "acc": total_correct / max(total_num, 1),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dataset", type=str, default="sics155")
    parser.add_argument("--ms_tcn_root", type=str, default="/root/autodl-tmp/MS-TCN2")
    parser.add_argument("--frame_root", type=str, required=True)
    parser.add_argument("--eye_root", type=str, required=True)
    parser.add_argument("--inst_root", type=str, required=True)

    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--backbone", type=str, default="dm_nfnet_f0")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_frames_per_phase", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--no_amp", action="store_true")

    parser.add_argument("--lambda_inst", type=float, default=0.10)
    parser.add_argument("--lambda_eye", type=float, default=0.08)
    parser.add_argument("--lambda_gate", type=float, default=0.08)

    parser.add_argument("--save_dir", type=str, default="/root/autodl-tmp/MS-TCN2/checkpoints/m8")
    parser.add_argument("--exp_name", type=str, default="m8_igcoattn_v1_2_dualgate")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.save_dir)

    mapping_path = os.path.join(args.ms_tcn_root, "data", args.base_dataset, "mapping.txt")
    splits_dir = os.path.join(args.ms_tcn_root, "data", args.base_dataset, "splits")
    train_bundle = os.path.join(splits_dir, f"train.split{args.split}.bundle")
    test_bundle = os.path.join(splits_dir, f"test.split{args.split}.bundle")

    train_vids = read_bundle(train_bundle)
    val_vids = read_bundle(test_bundle)

    if len(train_vids) == 0:
        raise RuntimeError(f"No train videos found from {train_bundle}")
    if len(val_vids) == 0:
        raise RuntimeError(f"No val videos found from {test_bundle}")

    print("=" * 80)
    print("[M8-X5-IGCoAttn-v1.2-dual-gate]")
    print(f"mapping_path: {mapping_path}")
    print(f"train videos: {len(train_vids)}")
    print(f"val videos  : {len(val_vids)}")
    print(f"backbone    : {args.backbone}")
    print("=" * 80)

    train_set = SICSFrameDataset(
        frame_root=args.frame_root,
        eye_root=args.eye_root,
        inst_root=args.inst_root,
        mapping_path=mapping_path,
        video_ids=train_vids,
        img_size=args.img_size,
        resize_size=args.resize_size,
        max_frames_per_phase=args.max_frames_per_phase,
        is_train=True,
    )

    val_set = SICSFrameDataset(
        frame_root=args.frame_root,
        eye_root=args.eye_root,
        inst_root=args.inst_root,
        mapping_path=mapping_path,
        video_ids=val_vids,
        img_size=args.img_size,
        resize_size=args.resize_size,
        max_frames_per_phase=args.max_frames_per_phase,
        is_train=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
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

    name_to_id, id_to_name = read_mapping(mapping_path)
    num_classes = len(id_to_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = M8IGCoAttnV1Asym(
        backbone_name=args.backbone,
        num_classes=num_classes,
        pretrained=True,
        mid_ch=128,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.lr * 0.05,
    )

    amp = not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_acc = -1.0
    best_path = os.path.join(args.save_dir, f"{args.exp_name}_best.pth")
    last_path = os.path.join(args.save_dir, f"{args.exp_name}_last.pth")
    log_path = os.path.join(args.save_dir, f"{args.exp_name}_train_log.jsonl")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            train=True,
            lambda_inst=args.lambda_inst,
            lambda_eye=args.lambda_eye,
            lambda_gate=args.lambda_gate,
            amp=amp,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            scaler=None,
            device=device,
            epoch=epoch,
            train=False,
            lambda_inst=args.lambda_inst,
            lambda_eye=args.lambda_eye,
            lambda_gate=args.lambda_gate,
            amp=amp,
        )

        scheduler.step()

        print("=" * 80)
        print(f"[Epoch {epoch:03d}]")
        print(f"Train: {train_metrics}")
        print(f"Val  : {val_metrics}")
        print(
            f"Gamma: inst={torch.tanh(model.gamma_inst).item():.5f}, "
            f"eye={torch.tanh(model.gamma_eye).item():.5f}, "
            f"gate={torch.tanh(model.gamma_gate).item():.5f}"
        )
        print("=" * 80)

        ckpt = {
            "epoch": epoch,
            "args": vars(args),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "num_classes": num_classes,
            "id_to_name": id_to_name,
        }

        torch.save(ckpt, last_path)

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            torch.save(ckpt, best_path)
            print(f"[SAVE] best checkpoint updated: {best_path}")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "best_acc": best_acc,
            }, ensure_ascii=False) + "\n")

    print(f"[DONE] best_acc={best_acc:.4f}")
    print(f"[BEST] {best_path}")
    print(f"[LAST] {last_path}")


if __name__ == "__main__":
    main()