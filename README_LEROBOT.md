# PACE V10 — LeRobot-π0.5 base

Branch `v10`: the V10 code exactly as it ran (the 2026-10-28 snapshot of `lerobot-pi05` without `scripts/v11/`).
V11 is on branch `v11`, V11b on `v11b`.

## LeRobot-π0.5 snapshot (V8–V10, frozen 2026-10-28)

This branch freezes the code of every PACE version that ran on the LeRobot port of π0.5,
`lerobot/pi05_libero_finetuned` (LeRobot's own LIBERO fine-tune: 50-step action chunks, 10 executed per call,
mean/std normalization). On 2026-10-28 the project moved its base model to Physical Intelligence's official
`pi05_libero` (openpi; 10-step chunks, 5 executed per call) served through openpi itself; that work lives on
other branches. Numbers obtained with this snapshot must not be mixed with official-model results.

| Path | Contents |
|---|---|
| `pace/` | The PACE package as it ran on the servers (`code_v10/pace`): experience adapter (`core.py`, V9's branch also reused by V11), π0.5 hooks (`pi05.py`), training, LIBERO / LIBERO-plus evaluation, retrieval and key studies, V10-0 rule follower (`follow.py`). |
| `scripts/v10/` | V10 premise checks and upper bounds: oracle shift / noise, localization probe, reference keys. |
| `scripts/v10b/` | V10 data line and method: π0.5 references and self-evaluation, pair collection, augmentation, replay rendering, feature extraction, the comparator (`train_v10b.py`), value head (`fit_value_low.py`), closed loops (`closed_loop_v3.py` = V10-3-1 hard follow, `closed_loop_v32.py` = V10-3-2 soft follow). |
| `scripts/v10pro/` | LIBERO-Pro side line (reported only). |

The commit before this one keeps a local working copy of `pace/core.py`, `pace/pi05.py`, `pace/train.py` as they were
on the development machine. Machine-specific transfer and setup scripts are not included. Third-party code
(`lerobot/`, `openpi/`) is cloned separately, as on `main`.
