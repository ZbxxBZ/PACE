"""
Export the experience for the LIBERO-PRO true-d check (10-23): successes of the V10-0 run (plus_l10_V10f_acc3) on the
light / sensor-noise / background variants of every libero_10 base task, all from init state 0 (physically the
original libero_10 scene), as env-unit action sequences (the closed loop's unnormalization: policy postprocessor,
then the env postprocessor), so they can be replayed in another code base (LIBERO-PRO, Python 3.8) without lerobot.
Up to --per_base per base task. CPU only.
    python v10_export_sources.py -> runs/rectest/v10_pro/sources_l10.npz (+ .json meta)
"""
import argparse
import collections
import glob
import json
import os

import numpy as np
import torch

RUN = "/root/pace/runs/eval/plus_l10_V10f_acc3"
SOURCE_CATS = ("Light Conditions", "Sensor Noise", "Background Textures")
OUT = "/root/pace/runs/rectest/v10_pro"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_base", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    eps = []
    for f in sorted(glob.glob(f"{RUN}/rollouts/rollouts_*.pt")):
        eps += torch.load(f, map_location="cpu", weights_only=False)["episodes"]
    rows = {r["task_id"]: r for r in map(json.loads, open(f"{RUN}/episodes.jsonl"))}
    by_base = collections.defaultdict(dict)
    for e in eps:
        m = e["meta"]
        if e["success"] and m["category"] in SOURCE_CATS and m["task_id"] not in by_base[m["base"]]:
            by_base[m["base"]][m["task_id"]] = e
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoPlusEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.libero import _get_suite
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import ACTION

    cfg = PreTrainedConfig.from_pretrained("lerobot/pi05_libero_finetuned")
    _, policy_post = make_pre_post_processors(cfg, pretrained_path="lerobot/pi05_libero_finetuned",
                                              preprocessor_overrides={"device_processor": {"device": "cpu"},
                                                                      "rename_observations_processor": {"rename_map": {}}})
    _, env_post = make_env_pre_post_processors(LiberoPlusEnv(task="libero_10"), cfg)
    suite = _get_suite("libero_10")

    def post(a):
        out = env_post({ACTION: policy_post(a)})[ACTION]
        return (out if torch.is_tensor(out) else torch.as_tensor(np.asarray(out))).float().cpu().numpy()

    arrays, meta = {}, []
    for base, d in sorted(by_base.items()):
        order = sorted(d.values(), key=lambda e: (SOURCE_CATS.index(e["meta"]["category"]), e["meta"]["task_id"]))
        for e in order[:args.per_base]:
            m = e["meta"]
            task = suite.get_task(int(m["task_id"]))
            acts = e["actions"][:, :10].reshape(-1, e["actions"].shape[-1]).float()
            key = f"s{len(meta):03d}"
            arrays[key] = post(acts).astype(np.float64)
            meta.append(dict(key=key, base=base, task_id=int(m["task_id"]), variant=task.name, category=m["category"],
                             init_index=int(m.get("init_index", 0)), seed=int(m["seed"]), steps=int(arrays[key].shape[0]),
                             env_steps=int(rows.get(int(m["task_id"]), {}).get("env_steps", -1))))
    np.savez_compressed(f"{OUT}/sources_l10.npz", **arrays)
    json.dump(meta, open(f"{OUT}/sources_l10.json", "w"), indent=1)
    print(f"{len(meta)} sources over {len({m['base'] for m in meta})} base tasks -> {OUT}/sources_l10.npz")
    print(collections.Counter(m["base"][:50] for m in meta))


if __name__ == "__main__":
    main()
