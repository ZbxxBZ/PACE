# PACE V11b — LeRobot-π0.5 base

Branch `v11b`: V11 (branch `v11`) plus V11b, as it ran on the 4090 server up to 2026-11-03. Numbers come from
`lerobot/pi05_libero_finetuned` and must not be mixed with official-π0.5 results.

V11b keeps V11's branch on the frozen π0.5 and changes how the memory is read:

- **Reference step (PMA):** the step of the reference is found by a state-only Progressive Memory Alignment (hand position and
  gripper, no image, nothing learned; RTCF, arXiv 2608.04527). Every part reads that one step.
- **Compare head:** one token per image cell from current − reference, bias-free, read by every action step; its output is added to
  the branch's correction.
- **Match gate:** g in (0, 1) from the compare tokens and the state difference, supervised by how the sample was paired (BCE);
  the executed correction is g·Δ. No d head.
- **Data:** camera 30 %, wide camera 20 %, replay 15 %, non-camera successes 15 % (target 0), other-task reference 15 % and
  time-shifted reference 5 % (target 0, gate 0); the reference action chunk is dropped in 40 % of training samples.

| Path | Contents |
|---|---|
| `scripts/v11/train_v11b.py` | Training, validation, checkpoint choice, the image-swap probe run after training. |
| `scripts/v11/closed_loop_v11b.py` | Closed loop (wraps `closed_loop_v11.py`, online PMA). |
| `scripts/v11/v11b_swap.py` | Swap probe: moved-camera, same-task, constant and other-task reference images. |
| `scripts/v11/v11b_replay.py`, `v11b_noncam.py`, `v11b_recenv.py`, `v11b_data_check.py` | V11b data: replays (keep, pure, wide camera), non-camera episodes, checks. |
| `scripts/v11/v11_imgpath.py`, `v11b_probe.py`, `v11_refvis_orig.py` | Why V11 did not use the images: attention shares, overfit tests, compare-head prototype; references with the camera in place. |
| `scripts/v11/train_v11_v9.py`, `scripts/v10b/augment.py` | V11's trainer with `--ref_isolation`, augmentation with the wide camera (`sample_wide`). |
| `scripts/v11b_queue/` | The queue scripts that ran the stages on GPU1 (short trainings, 30000-step training, swap probes, closed loop, continuation to 40000). |

Run outputs (`runs/v11b/...`) are not in the repository.
