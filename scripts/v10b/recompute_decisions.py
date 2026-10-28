"""
Recover the follow decision of every V10-1b closed-loop episode (10-24). closed_loop.py wrote the wrapper log only when
an episode ended with success / termination, so failed episodes (stopped by lerobot at the step limit) have no log and
their "follow" was recorded as None -- counted as "not followed" in closed_report.py. The decision is made at the
first call, at step 0, before any action: the frames depend only on the env seed, so it is recomputed exactly here
(same env construction, same 224 render, same model) and written next to the episodes.
    python recompute_decisions.py -> runs/v10b/closed/run1/decisions.json
"""
import json

import numpy as np
import torch

import closed_loop as C
import collect_pairs as P
import v10_oracle_shift2 as O

R = "/root/pace/runs"
RUN = f"{R}/v10b/closed/run1"


def main():
    P.worker_init()
    G = O.G
    from lerobot.envs.configs import LiberoPlusEnv
    from pace.pi05 import InstructionText

    env_cfg = LiberoPlusEnv(task="libero_10")
    gk = dict(env_cfg.gym_kwargs)
    gk.pop("task_ids", None)
    rows = [json.loads(l) for l in open(f"{R}/v10b/noise_sources/episodes.jsonl")]
    srcs = {}
    for r in rows:
        if not r["success"]:
            continue
        s = dict(key=(r["base"], r["task_id"]), task_id=int(r["task_id"]), init_index=int(r["init"]), seed=int(r["seed"]),
                 language=r["language"], actions=np.load(f"{R}/v10b/noise_sources/{r['actions']}").astype(np.float64))
        rec = O.record_source(s)
        if not rec["success"]:
            continue
        s["phases"] = [(int(a), int(b)) for a, b, _ in O.phase_targets(rec)]
        n = len(s["actions"])
        s["path"] = np.concatenate([rec["path"], np.repeat(rec["path"][-1:], max(0, n - len(rec["path"])), 0)])
        z = np.load(f"{RUN}/refs/t{s['task_id']}_i{s['init_index']}.npz")
        s.update(feats=z["feats"], t=z["t"], eef=z["eef"], grip=z["grip"])
        srcs[s["task_id"]] = s
    idx_lang = [json.loads(l)["language"] for l in open(f"{R}/v10b/refs/episodes.jsonl")]
    text = InstructionText.from_checkpoint("lerobot/pi05_libero_finetuned", tasks=list(dict.fromkeys(idx_lang)))
    lang_of = {r["task_id"]: r["base"] for r in map(json.loads, open(f"{R}/rectest/v10_shift/base_targets/episodes.jsonl"))}
    eps = [json.loads(l) for l in open(f"{RUN}/episodes.jsonl")]
    out = {}
    for e in eps:
        if e["mode"] != "v10":
            continue
        s = srcs[e["source"]]
        cfg = dict(mode="v10", margin=0.0, ckpt=f"{R}/v10b/train/run1/best.pt", pca=f"{R}/v10b/feats/pca.npz",
                   tower=f"{R}/v10b/closed/siglip_tower.pt",
                   src={k: s[k] for k in ("actions", "path", "phases", "feats", "t", "eef", "grip")},
                   lang=[text.embed(lang_of[e["task_id"]]).numpy(), text.embed(s["language"]).numpy()])
        fn = G["make_fns"](suite=G["suite"], suite_name="libero_10", task_id=e["task_id"], n_envs=1, camera_names=G["cams"],
                           episode_length=env_cfg.episode_length, init_states=env_cfg.init_states, gym_kwargs=gk,
                           control_mode=env_cfg.control_mode, camera_name_mapping=env_cfg.camera_name_mapping,
                           is_libero_plus=env_cfg.is_libero_plus)[0]
        env = C.V10Env(fn.func(e["init"], **fn.keywords), cfg)
        try:
            env.reset(seed=e["seed"])
            d, pf, pb = env._infer(env.env.unwrapped._env.env)
        finally:
            env.close()
        out[e["key"]] = dict(pf=pf, pb=pb, follow=bool(pf - pb > 0.0), d0=[float(x) for x in d], logged=e["follow"])
        print(e["key"], round(pf, 3), round(pb, 3), out[e["key"]]["follow"], "logged", e["follow"], flush=True)
    json.dump(out, open(f"{RUN}/decisions.json", "w"), indent=1)
    agree = [v["follow"] == v["logged"] for v in out.values() if v["logged"] is not None]
    print(f"[check] recomputed decision == logged decision on {sum(agree)}/{len(agree)} logged episodes")
    print(f"-> {RUN}/decisions.json")


if __name__ == "__main__":
    main()
