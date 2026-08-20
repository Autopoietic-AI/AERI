import os
# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import json

ROOT = Path(os.environ.get("ROOT", "/path/to/MS-TCN2"))

BASE = ROOT / "data/cataract101_m10_a4_geo_assets_sam2/npz"
NEW = ROOT / "data/cataract101_m10_a4_geo_assets_sam2_yolo_tiptrack_cons/npz"

OUT_ROOT = ROOT / "data/cataract101_m10_a4_geo_assets_sam2_yolo_tiptrack_cons_compat"
OUT = OUT_ROOT / "npz"
OUT.mkdir(parents=True, exist_ok=True)

bad = []
rows = []

files = sorted(NEW.glob("case_*.npz"))
print("new files:", len(files), flush=True)
print("base:", BASE, flush=True)
print("out:", OUT, flush=True)

for i, npz_new in enumerate(files, 1):
    case = npz_new.stem
    npz_base = BASE / f"{case}.npz"
    out_path = OUT / f"{case}.npz"

    with np.load(npz_new, allow_pickle=True) as n:
        data = {}

        # 先从旧 A4 asset 继承完整结构
        if npz_base.exists():
            with np.load(npz_base, allow_pickle=True) as b:
                for k in b.files:
                    data[k] = b[k]
        else:
            bad.append((case, "missing base asset"))
            data = {}

        required_new = ["geo", "traj", "presence_mask", "eye_center", "eye_radius", "inst_tips"]
        missing_new = [k for k in required_new if k not in n.files]
        if missing_new:
            bad.append((case, "missing new keys", missing_new))
            continue

        geo = n["geo"].astype(np.float32)
        traj = n["traj"].astype(np.float32)
        presence = n["presence_mask"].astype(np.float32)
        T = geo.shape[-1]

        # 用新 v1c-cons 的几何和轨迹覆盖旧内容
        data["geo"] = geo
        data["traj"] = traj
        data["presence_mask"] = presence
        data["eye_center"] = n["eye_center"].astype(np.float32)
        data["eye_radius"] = n["eye_radius"].astype(np.float32)
        data["inst_tips"] = n["inst_tips"].astype(np.float32)

        if "inst_centers" in n.files:
            data["inst_centers"] = n["inst_centers"].astype(np.float32)
        if "frame_ids" in n.files:
            data["frame_ids"] = n["frame_ids"]

        # aux: 旧模型需要 4 x T
        # aux[0] 保留旧 eye_valid，如果没有则默认 1
        # aux[1] any instrument presence
        # aux[2] instrument count / 2
        # aux[3] 保留旧面积类辅助量，如果没有则 0
        old_aux = data.get("aux", None)
        if old_aux is not None and getattr(old_aux, "shape", None) == (4, T):
            aux = old_aux.astype(np.float32).copy()
        else:
            aux = np.zeros((4, T), dtype=np.float32)
            aux[0] = 1.0

        aux[1] = (presence.max(axis=0) > 0.5).astype(np.float32)
        aux[2] = presence.sum(axis=0).astype(np.float32) / 2.0
        data["aux"] = aux.astype(np.float32)

        # cls_proxy / cls_proxy_mask：旧 A4 模型可能会读
        # cls_proxy 继承旧 asset；mask 用新 presence 对齐
        if "cls_proxy" not in data:
            # 理论上不应该发生；兜底避免 KeyError
            data["cls_proxy"] = np.zeros((2, 512, T), dtype=np.float32)

        cp = data["cls_proxy"]
        if getattr(cp, "shape", None) != (2, 512, T):
            # 如果 T 不一致，兜底补零，避免训练直接炸
            data["cls_proxy"] = np.zeros((2, 512, T), dtype=np.float32)

        data["cls_proxy_mask"] = presence.astype(np.float32)

        # 检查关键字段
        required_final = [
            "geo", "traj", "aux", "presence_mask",
            "cls_proxy", "cls_proxy_mask",
            "eye_center", "eye_radius", "inst_tips"
        ]

        missing_final = [k for k in required_final if k not in data]
        if missing_final:
            bad.append((case, "missing final keys", missing_final))
            continue

        if data["geo"].shape != (2, 7, T):
            bad.append((case, "bad geo", data["geo"].shape))
        if data["traj"].shape != (2, 8, T):
            bad.append((case, "bad traj", data["traj"].shape))
        if data["aux"].shape != (4, T):
            bad.append((case, "bad aux", data["aux"].shape))
        if data["presence_mask"].shape != (2, T):
            bad.append((case, "bad presence", data["presence_mask"].shape))
        if data["cls_proxy_mask"].shape != (2, T):
            bad.append((case, "bad cls_proxy_mask", data["cls_proxy_mask"].shape))

        np.savez_compressed(out_path, **data)

        rows.append({
            "case": case,
            "T": int(T),
            "presence_mean": float(presence.mean()),
            "geo_dist_p95": float(np.percentile(geo[:, 2][geo[:, 2] > 0], 95)) if np.any(geo[:, 2] > 0) else 0.0,
            "traj_speed_p95": float(np.percentile(traj[:, 2][traj[:, 2] > 0], 95)) if np.any(traj[:, 2] > 0) else 0.0,
        })

    if i % 10 == 0:
        print(f"[{i}/{len(files)}] done", flush=True)

summary = {
    "base": str(BASE),
    "new": str(NEW),
    "out": str(OUT),
    "num_files": len(files),
    "num_written": len(list(OUT.glob("case_*.npz"))),
    "bad_count": len(bad),
    "bad_samples": bad[:10],
    "note": "A4-v1c-cons compatible assets: inherit aux/cls_proxy from old A4 asset, override geo/traj/tip/presence with YOLO-clean tip-track."
}

with open(OUT_ROOT / "compat_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

if bad:
    raise RuntimeError(f"Bad compat assets: {bad[:5]}")
