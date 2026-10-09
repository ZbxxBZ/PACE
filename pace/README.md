# PACE: Progress-Aligned Cross-episode Experience

A plug-in experience memory for pi0.5 (LeRobot `PI05Policy`), ported from the MemoryVLA experience-retrieval branch
(v2 adapter). pi0.5 stays frozen and unmodified; PACE attaches through forward hooks and two instance wrappers and
`PI05Experience.detach()` removes them.

| file | what |
|---|---|
| `core.py` | host-independent: PMA alignment, experience store, runtime (one per running episode), adapter |
| `pi05.py` | pi0.5 attachment: keys from the image tokens, hooks on `action_in_proj` and the action expert's layers, training loss on the inference path |
| `build_library.py` | LeRobot dataset -> demo library (keys, action chunks, frame indices per policy call) |
| `train.py` | adapter training (pi0.5 frozen) with paired validation: real / unaligned / other_task / none |
| `eval.py` | `lerobot-eval` with PACE: conditions, prompt override (hidden goal), rollout accumulation |
| `common.py` | loading helpers, prompt override |

Run from `D:\project\PACE` (the `pace` package next to `lerobot/`), with lerobot installed (`pip install -e lerobot[pi,libero]`).

## Workflow (LIBERO, `lerobot/pi05_libero_finetuned`, n_action_steps 10)

```bash
# 1. library of the training demos (vision tower only)
python -m pace.build_library --policy_path lerobot/pi05_libero_finetuned --dataset_repo_id lerobot/libero \
    --suite libero_goal --n_action_steps 10 --out libs/goal_demo.pt

# 2. adapter (hidden goal: the policy is told a generic prompt; the references reveal the goal)
python -m pace.train --policy_path lerobot/pi05_libero_finetuned --dataset_repo_id lerobot/libero \
    --library libs/goal_demo.pt --n_action_steps 10 --out_dir runs/goal_hidden \
    --prompt_override "complete the task shown in the demonstration"

# 3. paired evaluation: same --eval.batch_size and --seed for every condition
python -m pace.eval --pace.adapter runs/goal_hidden/adapter_last.pt --pace.library libs/goal_demo.pt \
    --pace.condition real --pace.prompt_override "complete the task shown in the demonstration" \
    --pace.out_dir eval/goal_hidden_real -- \
    --env.type=libero --env.task=libero_goal --eval.batch_size=10 --eval.n_episodes=20 \
    --policy.path=lerobot/pi05_libero_finetuned --policy.n_action_steps=10 --output_dir=eval/goal_hidden_real/lerobot
```

Checks before any result:
- `--pace.condition none` (or no `--pace.adapter`) must reproduce plain `lerobot-eval` episode by episode;
- step 0 of training logs equal losses in every validation condition (zero-initialized adapter);
- `--ref_content self` must drive `val_self` far below `val_none` (the adapter can copy).

## Design notes
- Key of a call: SigLIP + projector image tokens (before the language model, so independent of the prompt and of the
  state in the prompt), mean-pooled per camera (default cameras 0 and 1 = agent view and wrist), normalized and
  concatenated. Scene signature: the first camera's token grid pooled to 4 x 4. Stores record this in `info` and
  refuse to mix.
- Adapter: gated cross-attention after every 2nd of the 18 expert layers (zero-initialized output projections) and
  input injection of the score-weighted reference chunk into the action tokens (zero-initialized). Without
  references the residual is exactly zero, so the policy output is bit-identical.
- Training runs on the inference path (prefix KV cache without gradients, one denoising step at a sampled flow time),
  because the joint training forward of pi0.5 bypasses the expert layers' `forward` and therefore the hooks.
- Not supported: `compile_model=True`, MEM memory inputs, RTC guidance with the adapter.
