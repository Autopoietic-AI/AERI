# -*- coding: utf-8 -*-
"""
A5-maskShape-TIG-Dfusion-v0

Core idea:
    Do not residual-inject new cues into A4 guide_token.

    Build:
        visual branch       from m8_feat
        local interaction   from geo/traj/mask_shape/reliability at frame t
        temporal interaction from TIG:
            Q_t = current mask_shape_core + reliability
            K,V = local temporal window of geo/traj/mask_delta/reliability + relative time
        aux branch          from aux/reliability

    Fuse:
        z(t) = D-Fusion(visual, local+temporal interaction, aux)
        MS-TCN++ decodes phase sequence
"""

import os
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model import MS_TCN2


class GeoIGConditionEncoder(nn.Module):
    """
    Keep the original A4 GeoIG encoder for compatibility.
    In TIG-Dfusion, we do not use its residual guide injection as the main route.
    """
    def __init__(self, hidden=128, sem_dim=32, guide_dim=64):
        super().__init__()

        self.cls_proj = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, sem_dim, kernel_size=1),
        )

        # geo: 2*7=14
        # aux: 4
        # presence: 2
        # traj: 2*8=16
        # cls sem: 2*sem_dim
        cond_dim = 14 + 4 + 2 + 16 + 2 * sem_dim

        self.cond = nn.Sequential(
            nn.Conv1d(cond_dim, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.guide_head = nn.Conv1d(hidden, guide_dim, kernel_size=1)
        self.gate_head = nn.Conv1d(hidden, 1, kernel_size=1)

        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -2.0)

    def forward(self, geo, aux, presence_mask, cls_proxy, cls_proxy_mask, traj):
        B, _, _, T = geo.shape

        geo_flat = geo.reshape(B, 14, T)
        traj_flat = traj.reshape(B, 16, T)

        cls = cls_proxy.reshape(B * 2, 512, T)
        cls_sem = self.cls_proj(cls).reshape(B, 2, -1, T)

        mask = cls_proxy_mask.unsqueeze(2).clamp(0, 1)
        cls_sem = cls_sem * mask
        cls_flat = cls_sem.reshape(B, -1, T)

        cond = torch.cat([
            geo_flat,
            aux,
            presence_mask,
            traj_flat,
            cls_flat,
        ], dim=1)

        h = self.cond(cond)
        guide_token = self.guide_head(h)

        beta_max = 0.35
        active = presence_mask.max(dim=1, keepdim=True).values
        gate = beta_max * torch.sigmoid(self.gate_head(h))
        gate = gate * active

        return guide_token, gate, h



class DilatedTemporalIGBranch(nn.Module):
    """
    Dilated Temporal Interaction Guidance.

    Compared with the original contiguous local window:
        [t-4, ..., t+4]

    DilatedTIG attends to sparse long-range offsets:
        [-16, -8, -4, -2, 0, 2, 4, 8, 16]

    This enlarges temporal receptive field without increasing the number of
    attention positions.
    """
    def __init__(
        self,
        q_in_dim=15,
        evidence_dim=30,
        q_dim=64,
        out_dim=12,
        hidden=128,
        window=9,
    ):
        super().__init__()

        offsets_str = os.environ.get("A5_TIG_DILATED_OFFSETS", "-16,-8,-4,-2,0,2,4,8,16")
        self.offsets = [int(x.strip()) for x in offsets_str.split(",") if x.strip()]
        assert len(self.offsets) > 0, "A5_TIG_DILATED_OFFSETS cannot be empty"
        print("[DilatedTIG] offsets =", self.offsets, flush=True)

        self.window = len(self.offsets)
        self.q_dim = int(q_dim)
        self.out_dim = int(out_dim)

        self.q_proj = nn.Sequential(
            nn.Conv1d(q_in_dim, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, q_dim, kernel_size=1),
        )

        self.k_proj = nn.Sequential(
            nn.Conv1d(evidence_dim, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, q_dim, kernel_size=1),
        )

        self.v_proj = nn.Sequential(
            nn.Conv1d(evidence_dim, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, out_dim, kernel_size=1),
        )

        self.out_norm = nn.GroupNorm(1, out_dim)

    def gather_offsets(self, x):
        """
        x: (B,C,T)
        return: (B,C,T,W), where W=len(offsets)
        """
        B, C, T = x.shape
        device = x.device
        base = torch.arange(T, device=device)

        gathered = []
        for off in self.offsets:
            idx = torch.clamp(base + int(off), min=0, max=T - 1)
            gathered.append(x.index_select(dim=2, index=idx))

        return torch.stack(gathered, dim=-1)

    def forward(self, q_feat, evidence_feat, rel_score):
        """
        q_feat:        (B, Cq, T)
        evidence_feat: (B, Ce, T)
        rel_score:     (B, 1, T)
        """
        B, _, T = q_feat.shape
        W = self.window

        q = self.q_proj(q_feat)              # (B,D,T)
        k = self.k_proj(evidence_feat)       # (B,D,T)
        v = self.v_proj(evidence_feat)       # (B,O,T)

        k_win = self.gather_offsets(k)          # (B,D,T,W)
        v_win = self.gather_offsets(v)          # (B,O,T,W)
        r_win = self.gather_offsets(rel_score)  # (B,1,T,W)

        score = (q.unsqueeze(-1) * k_win).sum(dim=1) / math.sqrt(self.q_dim)  # (B,T,W)

        rel_bias = torch.log(r_win.squeeze(1).clamp_min(1e-4))  # (B,T,W)
        score = score + 0.5 * rel_bias

        attn = torch.softmax(score, dim=-1)  # (B,T,W)

        e = (attn.unsqueeze(1) * v_win).sum(dim=-1)  # (B,O,T)
        e = self.out_norm(e)

        return e, attn


class TIGDFusionBranch(nn.Module):
    """
    D-Fusion with Temporal IG.

    Branches:
        visual branch:      projected visual feature
        interaction branch: local interaction + temporal interaction
        aux branch:         auxiliary reliability
    """
    def __init__(
        self,
        main_dim=64,
        f_dim=48,
        e_local_dim=12,
        e_temp_dim=12,
        a_dim=4,
        hidden=128,
        tig_q_dim=64,
        tig_window=9,
        fixed_gate=True,
    ):
        super().__init__()

        self.f_dim = f_dim
        self.e_local_dim = e_local_dim
        self.e_temp_dim = e_temp_dim
        self.a_dim = a_dim
        self.fixed_gate = fixed_gate

        self.visual_proj = nn.Sequential(
            nn.Conv1d(main_dim, f_dim, kernel_size=1),
            nn.GroupNorm(1, f_dim),
            nn.GELU(),
        )

        # local per-track interaction:
        # geo 7 + traj 8 + mask_core 9 + rel 6 = 30
        self.track_mlp = nn.Sequential(
            nn.Conv1d(30, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, e_local_dim, kernel_size=1),
            nn.GroupNorm(1, e_local_dim),
            nn.GELU(),
        )

        # TIG evidence:
        # geo_pool 7 + traj_pool 8 + mask_temporal 4 + rel_summary 4 + relative_time_context 7? 
        # We construct a compact 30D evidence to keep consistent.
        self.evidence_proj_in_dim = 30

        # Q feature:
        # mask_core pooled 9 + rel summary 4 + aux 2 = 15
        self.tig = DilatedTemporalIGBranch(
            q_in_dim=20,
            evidence_dim=self.evidence_proj_in_dim,
            q_dim=tig_q_dim,
            out_dim=e_temp_dim,
            hidden=hidden,
            window=tig_window,
        )

        self.interaction_fuse = nn.Sequential(
            nn.Conv1d(e_local_dim + e_temp_dim, e_local_dim + e_temp_dim, kernel_size=1),
            nn.GroupNorm(1, e_local_dim + e_temp_dim),
            nn.GELU(),
        )

        # aux 4 + rel_max + rel_mean + presence_any = 7
        self.aux_proj = nn.Sequential(
            nn.Conv1d(7, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, a_dim, kernel_size=1),
            nn.GroupNorm(1, a_dim),
            nn.GELU(),
        )

        if fixed_gate:
            # global learnable scalar gate: avoid frame-wise gate flicker
            self.gate_logits = nn.Parameter(torch.tensor([2.0, 0.0, -0.5], dtype=torch.float32))
            self.dynamic_gate = None
        else:
            self.dynamic_gate = nn.Sequential(
                nn.Conv1d(7, hidden, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(hidden, 3, kernel_size=1),
            )
            self.gate_logits = None

        # Stage-aware Dynamic Fusion:
        # Memory is NOT injected into TIG query; it only modulates fusion weights.
        self.stage_gate_net = nn.Sequential(
            nn.Conv1d(12, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, 3, kernel_size=1),
        )

        self.memory_summary_proj = nn.Sequential(
            nn.Conv1d(self.evidence_proj_in_dim, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, 4, kernel_size=1),
        )

        self.stage_gate_residual = float(os.environ.get("A5_STAGE_GATE_RESIDUAL", "0.5"))
        print("[StageFusion] residual =", self.stage_gate_residual, flush=True)

        fusion_in = f_dim + e_local_dim + e_temp_dim + a_dim
        self.fusion_proj = nn.Sequential(
            nn.Conv1d(fusion_in, main_dim, kernel_size=1),
            nn.GroupNorm(1, main_dim),
            nn.GELU(),
        )

    @staticmethod
    def temporal_delta(x):
        d = torch.zeros_like(x)
        d[..., 1:] = x[..., 1:] - x[..., :-1]
        return d

    def forward(self, main_token, geo, traj, aux, presence_mask, mask_shape, mask_shape_valid):
        B = main_token.shape[0]

        T = min(
            main_token.shape[-1],
            geo.shape[-1],
            traj.shape[-1],
            aux.shape[-1],
            presence_mask.shape[-1],
            mask_shape.shape[-1],
            mask_shape_valid.shape[-1],
        )

        main_token = main_token[..., :T].float()
        geo = geo[..., :T].float()
        traj = traj[..., :T].float()
        aux = aux[..., :T].float()
        presence = presence_mask[..., :T].float().clamp(0, 1)
        mask_shape = mask_shape[..., :T].float()
        mask_shape_valid = mask_shape_valid[..., :T].float().clamp(0, 1)

        # cleanQ mask core:
        # 0 area, 1 area_delta, 2 union_area, 3 union_delta,
        # 4 inst_score, 5 eye_score, 6 inst_valid, 7 eye_valid, 14 compactness
        q_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 14], device=mask_shape.device, dtype=torch.long)
        mask_core = mask_shape.index_select(dim=2, index=q_idx)  # (B,2,9,T)

        inst_score = mask_shape[:, :, 4, :].clamp(0, 1)
        eye_score = mask_shape[:, :, 5, :].clamp(0, 1)
        inst_valid = mask_shape[:, :, 6, :].clamp(0, 1)
        eye_valid = mask_shape[:, :, 7, :].clamp(0, 1)

        rel_score = (
            mask_shape_valid *
            inst_score *
            eye_score *
            inst_valid *
            eye_valid *
            presence
        ).clamp(0, 1)  # (B,2,T)

        rel_feat = torch.stack([
            mask_shape_valid,
            inst_score,
            eye_score,
            inst_valid,
            eye_valid,
            presence,
        ], dim=2)  # (B,2,6,T)

        # local track interaction
        track_in = torch.cat([
            geo,
            traj,
            mask_core,
            rel_feat,
        ], dim=2)  # (B,2,30,T)

        track_in_flat = track_in.contiguous().view(B * 2, 30, T)
        e_track = self.track_mlp(track_in_flat).view(B, 2, self.e_local_dim, T)

        w = rel_score
        denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)

        e_local = (e_track * w.unsqueeze(2)).sum(dim=1) / denom  # (B,e_local_dim,T)

        # pooled evidence for temporal IG
        geo_pool = (geo * w.unsqueeze(2)).sum(dim=1) / denom       # (B,7,T)
        traj_pool = (traj * w.unsqueeze(2)).sum(dim=1) / denom     # (B,8,T)
        mask_pool = (mask_core * w.unsqueeze(2)).sum(dim=1) / denom  # (B,9,T)

        rel_max = w.max(dim=1, keepdim=True).values
        rel_mean = w.mean(dim=1, keepdim=True)
        presence_any = presence.max(dim=1, keepdim=True).values
        rel_stability = 1.0 - torch.abs(self.temporal_delta(rel_mean)).clamp(0, 1)

        # mask temporal cues: area / union area / compactness + their local change
        # mask_pool indices: 0 inst_area, 1 inst_area_delta, 2 union_area, 3 union_delta, 8 compactness
        mask_temporal = torch.cat([
            mask_pool[:, 0:1, :],
            mask_pool[:, 2:3, :],
            mask_pool[:, 8:9, :],
            torch.abs(self.temporal_delta(mask_pool[:, 2:3, :])),
        ], dim=1)  # (B,4,T)

        rel_summary = torch.cat([
            rel_max,
            rel_mean,
            presence_any,
            rel_stability,
        ], dim=1)  # (B,4,T)

        # Optional trajectory dimension ablation.
        # Example:
        #   export A5_ABLATE_TRAJ_DIMS=0,1,4,5,7
        # This MUST be before dist_speed and evidence construction.
        ablate_dims = os.environ.get("A5_ABLATE_TRAJ_DIMS", "").strip()
        if ablate_dims:
            traj_pool = traj_pool.clone()
            for s in ablate_dims.split(","):
                s = s.strip()
                if not s:
                    continue
                idx = int(s)
                if 0 <= idx < traj_pool.shape[1]:
                    traj_pool[:, idx:idx+1, :] = 0.0

            if os.environ.get("A5_ABLATE_DEBUG", "0") == "1":
                print("[A5_ABLATE_DEBUG] A5_ABLATE_TRAJ_DIMS =", ablate_dims, flush=True)
                print("[A5_ABLATE_DEBUG] traj_pool abs mean after ablation =",
                      float(traj_pool.detach().abs().mean().cpu()), flush=True)

        # evidence = 7 + 8 + 4 + 4 + 7 filler from aux/geo summaries = 30
        # use aux 4 + dist/speed summaries 3
        dist_speed = torch.cat([
            geo_pool[:, 2:3, :],    # dist
            traj_pool[:, 2:3, :],   # speed
            traj_pool[:, 3:4, :],   # rel_dist_delta
        ], dim=1)

        # ------------------------------------------------------------
        # Inference-time / train-time group ablation switches.
        # Set env var to 1 to zero out a group:
        #   A5_ABLATE_GEO=1
        #   A5_ABLATE_TRAJ=1
        #   A5_ABLATE_MASKDYN=1
        #   A5_ABLATE_REL=1
        #   A5_ABLATE_AUX=1
        #   A5_ABLATE_DISTSPEED=1
        # ------------------------------------------------------------
        if os.environ.get("A5_ABLATE_GEO", "0") == "1":
            geo_pool = torch.zeros_like(geo_pool)
        if os.environ.get("A5_ABLATE_TRAJ", "0") == "1":
            traj_pool = torch.zeros_like(traj_pool)
        if os.environ.get("A5_ABLATE_MASKDYN", "0") == "1":
            mask_temporal = torch.zeros_like(mask_temporal)
        if os.environ.get("A5_ABLATE_REL", "0") == "1":
            rel_summary = torch.zeros_like(rel_summary)
        if os.environ.get("A5_ABLATE_AUX", "0") == "1":
            aux = torch.zeros_like(aux)
        if os.environ.get("A5_ABLATE_DISTSPEED", "0") == "1":
            dist_speed = torch.zeros_like(dist_speed)

        evidence = torch.cat([
            geo_pool,        # 7
            traj_pool,       # 8
            mask_temporal,   # 4
            rel_summary,     # 4
            aux,             # 4
            dist_speed,      # 3
        ], dim=1)             # 30

        # ProgressTIG query:
        # current mask/reliability/aux state + low-frequency workflow progress prior.
        Bq, _, Tq = mask_pool.shape
        device = mask_pool.device
        dtype = mask_pool.dtype
        denom = max(Tq - 1, 1)
        p_idx = torch.arange(Tq, device=device, dtype=dtype) / float(denom)
        pi = math.pi

        progress_feat = torch.stack([
            p_idx,
            torch.sin(pi * p_idx),
            torch.cos(pi * p_idx),
            torch.sin(2.0 * pi * p_idx),
            torch.cos(2.0 * pi * p_idx),
        ], dim=0).unsqueeze(0).repeat(Bq, 1, 1)  # (B,5,T)

        q_feat = torch.cat([
            mask_pool,       # 9
            rel_summary,     # 4
            aux[:, :2, :],    # 2
            progress_feat,   # 5
        ], dim=1)             # 20

        e_temp, tig_attn = self.tig(q_feat=q_feat, evidence_feat=evidence, rel_score=rel_mean)

        e_inter = self.interaction_fuse(torch.cat([e_local, e_temp], dim=1))

        aux_in = torch.cat([aux, rel_max, rel_mean, presence_any], dim=1)  # 7

        f_token = self.visual_proj(main_token)
        a_token = self.aux_proj(aux_in)

        # Stage-aware Dynamic Fusion:
        # Build memory summary from evidence, then use progress/reliability/memory to modulate fusion gate.
        Bsf, _, Tsf = f_token.shape
        device = f_token.device
        dtype = f_token.dtype

        denom_g = max(Tsf - 1, 1)
        p_idx_gate = torch.arange(Tsf, device=device, dtype=dtype) / float(denom_g)
        pi = math.pi

        progress_gate_feat = torch.stack([
            p_idx_gate,
            torch.sin(pi * p_idx_gate),
            torch.cos(pi * p_idx_gate),
            torch.sin(2.0 * pi * p_idx_gate),
            torch.cos(2.0 * pi * p_idx_gate),
        ], dim=0).unsqueeze(0).repeat(Bsf, 1, 1)  # (B,5,T)

        # 4-bin low-frequency video memory from evidence.
        Bm, Ce, Te = evidence.shape
        assert Te == Tsf, "evidence length must match fusion token length"

        bin_edges = [
            (0.00, 0.25),
            (0.25, 0.50),
            (0.50, 0.75),
            (0.75, 1.00),
        ]

        memory_tokens = []
        for l, r in bin_edges:
            s = int(round(l * Tsf))
            e = int(round(r * Tsf))
            s = max(0, min(s, Tsf - 1))
            e = max(s + 1, min(e, Tsf))
            memory_tokens.append(evidence[:, :, s:e].mean(dim=2))

        memory_tokens = torch.stack(memory_tokens, dim=2)  # (B,Ce,4)

        centers = torch.tensor([0.125, 0.375, 0.625, 0.875], device=device, dtype=dtype)
        sigma = float(os.environ.get("A5_MEMORY_SIGMA", "0.22"))

        p_for_weight = p_idx_gate.unsqueeze(1)
        w = torch.exp(-0.5 * ((p_for_weight - centers.view(1, 4)) / sigma) ** 2)
        w = w / torch.clamp(w.sum(dim=1, keepdim=True), min=1e-6)
        w = w.transpose(0, 1).unsqueeze(0).repeat(Bsf, 1, 1)

        memory_context = torch.einsum("bck,bkt->bct", memory_tokens, w)
        memory_summary = self.memory_summary_proj(memory_context)  # (B,4,T)

        stage_gate_input = torch.cat([
            progress_gate_feat,  # 5
            rel_mean,            # 1
            rel_max,             # 1
            presence_any,        # 1
            memory_summary,      # 4
        ], dim=1)                # 12

        dynamic_logits = self.stage_gate_net(stage_gate_input)
        dynamic_gate = torch.softmax(dynamic_logits, dim=1)

        if self.gate_logits is not None:
            base_gate = torch.softmax(self.gate_logits, dim=0).view(1, 3, 1).to(device).expand(Bsf, 3, Tsf)
        else:
            base_gate = torch.ones(Bsf, 3, Tsf, device=device, dtype=dtype) / 3.0

        r = float(self.stage_gate_residual)
        gate = (1.0 - r) * base_gate + r * dynamic_gate
        gate = gate / torch.clamp(gate.sum(dim=1, keepdim=True), min=1e-6)

        f_token = f_token * gate[:, 0:1, :]
        e_inter = e_inter * gate[:, 1:2, :]
        a_token = a_token * gate[:, 2:3, :]

        z = torch.cat([f_token, e_inter, a_token], dim=1)
        fused = self.fusion_proj(z)

        extra = {
            "dfusion_gate": gate,
            "tig_attn": tig_attn,
            "dfusion_rel": rel_score,
            "gate_mean": gate.detach().mean(),
            "gate_visual_mean": gate[:, 0, :].detach().mean(),
            "gate_interaction_mean": gate[:, 1, :].detach().mean(),
            "gate_aux_mean": gate[:, 2, :].detach().mean(),
            "gate_std": gate.detach().std(),
        }

        return fused, extra


class A5TIGDFusionModel(nn.Module):
    def __init__(
        self,
        m8_feat_dim,
        num_classes,
        num_f_maps=64,
        num_layers_PG=11,
        num_layers_R=10,
        num_R=4,
        main_dim=64,
        guide_dim=64,
        num_stages=None,
        num_layers=None,
        dim=None,
        **kwargs,
    ):
        super().__init__()

        self.main_proj = nn.Conv1d(m8_feat_dim, main_dim, kernel_size=1)

        self.geoig_encoder = GeoIGConditionEncoder(
            hidden=128,
            sem_dim=32,
            guide_dim=guide_dim,
        )

        self.geo_to_main = nn.Conv1d(guide_dim, main_dim, kernel_size=1)
        nn.init.zeros_(self.geo_to_main.weight)
        nn.init.zeros_(self.geo_to_main.bias)

        self.dfusion_branch = TIGDFusionBranch(
            main_dim=main_dim,
            f_dim=int(os.environ.get("A5_DFUSION_F_DIM", "48")),
            e_local_dim=int(os.environ.get("A5_DFUSION_E_LOCAL_DIM", "12")),
            e_temp_dim=int(os.environ.get("A5_TIG_E_TEMP_DIM", "12")),
            a_dim=int(os.environ.get("A5_DFUSION_A_DIM", "4")),
            hidden=int(os.environ.get("A5_DFUSION_HIDDEN", "128")),
            tig_q_dim=int(os.environ.get("A5_TIG_Q_DIM", "64")),
            tig_window=int(os.environ.get("A5_TIG_WINDOW", "9")),
            fixed_gate=os.environ.get("A5_DFUSION_FIXED_GATE", "1") == "1",
        )

        # Compatibility with different MS-TCN++ argument naming styles.
        if num_layers is not None:
            num_layers_PG = num_layers
            num_layers_R = num_layers
        if num_stages is not None:
            num_R = num_stages
        if dim is not None:
            num_f_maps = dim

        self.temporal_model = MS_TCN2(
            num_layers_PG,
            num_layers_R,
            num_R,
            num_f_maps,
            main_dim,
            num_classes,
        )

    def forward(
        self,
        m8_feat,
        geo,
        aux,
        presence_mask,
        cls_proxy,
        cls_proxy_mask,
        traj,
        eye_center=None,
        eye_radius=None,
        inst_tips=None,
        mask_shape=None,
        mask_shape_valid=None,
        return_extra=False,
    ):
        main_token = self.main_proj(m8_feat)

        guide_token, gate, h = self.geoig_encoder(
            geo=geo,
            aux=aux,
            presence_mask=presence_mask,
            cls_proxy=cls_proxy,
            cls_proxy_mask=cls_proxy_mask,
            traj=traj,
        )

        # A5-TIG-Dfusion:
        # do not residual-inject guide_token;
        # use independent temporal interaction branch.
        dfusion_extra = {}
        if mask_shape is not None and mask_shape_valid is not None:
            main_token, dfusion_extra = self.dfusion_branch(
                main_token=main_token,
                geo=geo,
                traj=traj,
                aux=aux,
                presence_mask=presence_mask,
                mask_shape=mask_shape,
                mask_shape_valid=mask_shape_valid,
            )

        outputs = self.temporal_model(main_token)

        if not return_extra:
            return outputs

        extra = {
            "gate": gate,
            "guide_token": guide_token,
            "dfusion_gate": dfusion_extra.get("dfusion_gate", None),
            "tig_attn": dfusion_extra.get("tig_attn", None),
            "dfusion_rel": dfusion_extra.get("dfusion_rel", None),
        }
        return outputs, extra
