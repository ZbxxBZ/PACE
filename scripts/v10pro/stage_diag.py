"""Why "stage + pi0.5" fails on the fixture tasks (route B, 10-23): runs pro_stage.py's vla mode on a few cells with
the same plan, seeds and code, and records per phase the staged hand vs. its target, the goal predicates at the end,
and a video (agentview | wrist, every 4th step). Diagnosis only, not a result.
    python stage_diag.py --cells 9:0 8:0 3:0 -> runs/rectest/v10_pro/stage_diag.json, videos/stage_<t>_<k>.mp4"""
import argparse
import json

import imageio
import numpy as np

import pro_common as C
import pro_oracle as P
import pro_stage as S

VID = "/root/pace/runs/rectest/v10_pro/videos/stage_%d_%d.mp4"


def goals(env):
    e = env.env
    return {" ".join(g): bool(e._eval_predicate(g)) for g in e.parsed_problem["goal_state"]}


def run(t, k, src, port):
    seed = 1000 + 100 * t + k
    client = C.PolicyClient(port)
    ep = S.Episode(t, k, seed, client)
    frames, step0 = [], ep.step

    def step(a):
        out = step0(a)
        if ep.t % 4 == 0:
            o = ep.obs
            frames.append(np.concatenate([o["agentview_image"][::-1, ::-1], o["robot0_eye_in_hand_image"][::-1, ::-1]], 1))
        return out
    ep.step = step
    rec, path = src["rec"], src["rec"]["path"]
    z_safe = float(path[:, 2].max())
    log = []
    try:
        for i, ph in enumerate(S.plan(rec)):
            d = np.zeros(3)
            if ph["anchor_pos"] is not None:
                d[:2] = ep.sim.positions()[ph["anchor"]][:2] - ph["anchor_pos"][:2]
            tgt = path[min(ph["pre"], len(path) - 1)] + d
            t0 = ep.t
            stop = ep.stage(tgt, z_safe)
            row = dict(phase=i, kind=ph["kind"], anchor=ph["anchor"], d_cm=(100 * d[:2]).round(1).tolist(),
                       stage_err_cm=round(100 * float(np.linalg.norm(ep.sim.hand() - tgt)), 1), stage_steps=ep.t - t0)
            if stop:
                log.append(row)
                break
            t1 = ep.t
            done = ep.vla_until(ph["kind"])
            row.update(vla_steps=ep.t - t1, hand_after=ep.sim.hand().round(3).tolist(),
                       anchor_after=ep.sim.positions()[ph["anchor"]].round(3).tolist() if ph["anchor"] in ep.sim.positions() else None)
            log.append(row)
            if done:
                break
            for _ in range(S.GRIP_SETTLE):
                a = np.zeros(7)
                a[6] = ep.grip
                if ep.step(a):
                    break
        if not ep.success and ep.t < S.BUDGET:
            ep.vla_until(None)
        res = dict(task=t, init=k, success=ep.success, steps=ep.t, phases=log, goals=goals(ep.env))
        imageio.mimwrite(VID % (t, k), frames, fps=10)
        return res
    finally:
        ep.close()
        client.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["9:0", "8:0", "3:0"])
    ap.add_argument("--port", type=int, default=18765)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(f"{P.EXP}/episodes.jsonl")]
    arrays = np.load(f"{P.EXP}/actions.npz")
    out = []
    for cell in args.cells:
        t, k = map(int, cell.split(":"))
        r = next(r for r in sorted(rows, key=lambda r: (r["task"], r["init"])) if r["task"] == t and r["success"])
        key = f"t{t}_i{r['init']}"
        src = dict(key=key, task=t, init=r["init"], seed=r["seed"], actions=arrays[key])
        src["rec"] = P.record_source(src)
        res = run(t, k, src, args.port)
        out.append(res)
        print("DIAG", json.dumps(res), flush=True)
        json.dump(out, open("/root/pace/runs/rectest/v10_pro/stage_diag.json", "w"), indent=1)


if __name__ == "__main__":
    main()
