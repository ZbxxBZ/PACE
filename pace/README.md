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
| `key_study.py` | offline comparison of retrieval-key transforms and alignment methods |
| `retrieval_study.py` | offline comparison of the classic and v3 retrievals (held-out demos replayed through the runtime) |
| `val_probe.py` | offline probes of a trained adapter on its run's held-out demos (perfect / near-identical / content-free references) |
| `eval_suite.py` | LIBERO / LIBERO-plus evaluation over a fixed variant list, rounds and rollout accumulation |
| `compare_runs.py` | paired comparison of eval_suite runs (McNemar, per round and category) |
| `rollouts.py` | v7: recording the policy's own rollouts with their frames (eval_suite) and reading them back (train.py) |

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
    --policy.path=lerobot/pi05_libero_finetuned --policy.n_action_steps=10 --policy.compile_model=false \
    --output_dir=eval/goal_hidden_real/lerobot
```

Cross-episode experience in evaluation (accumulation; use `pace.eval_suite` for every run that is compared, so
episodes pair across runs):

```bash
# base policy, 3 rounds = its run-to-run noise
python -m pace.eval_suite --env_type libero_plus --suite libero_10 --per_category 50 --rounds 3 --out_dir eval/plus_base3
# an adapter, the store growing with its own rollouts (successes are read back; failures are stored with their outcome)
python -m pace.eval_suite --env_type libero_plus --suite libero_10 --tasks_file eval/plus_base3/sample.json --rounds 3 \
    --pace.condition real --pace.accumulate --pace.adapter runs/train/v2_libero4/adapter_last.pt \
    --pace.library runs/libs/libero4_grid_pca128_feat.pt --out_dir eval/plus_v2_acc3
python -m pace.compare_runs eval/plus_base3 eval/plus_v2_acc3 --by category
```

Every round repeats the same init states and seeds; each batch's rollouts join the store when it ends and are saved
to `<out_dir>/rollouts/` (a resumed run reloads them). `compare_runs` pairs episodes, gives McNemar p-values against
the first run and every round against round 1 of the same run.

Checks before any result:
- `--pace.condition none` (or no `--pace.adapter`) must reproduce plain `lerobot-eval` episode by episode;
- step 0 of training logs equal losses in every validation condition (zero-initialized adapter);
- `--ref_content self` must drive `val_self` far below `val_none` (the adapter can copy).

## Design notes
- Key of a call: SigLIP + projector image tokens (before the language model, so independent of the prompt and of the
  state in the prompt) of cameras 0 and 1 (agent view and wrist), pooled to a 4 x 4 grid per camera (`--key grid`;
  `mean` = one token mean per camera), then the library's fitted transform (`--key_transform pca --pca_dim 128`:
  centering removes the component all frames share, which made raw cosines ~0.996 and let PMA's prior swamp them).
  The library stores the fit and deployment reuses it. Chosen with `key_study.py` (offline: align held-out demos to
  the rest, score the matched reference chunks). Scene signature: the first camera's grid pooled to 4 x 4. Stores
  record all of this in `info` and refuse to mix.
- Alignment: PMA over the call history (RTCF's) or frame nearest neighbour (`--align nn` in train.py,
  `--pace.align nn` in eval.py). On libero_goal (short tasks) frame-NN read slightly better chunks offline.
- Adapter: gated cross-attention in the 18-layer action expert (zero-initialized output projections) and input
  injection of the score-weighted reference chunk into the action tokens (zero-initialized). Without references the
  residual is exactly zero, so the policy output is bit-identical. Every branch is pre-norm (no norm on the residual
  stream, as in the host), so there is no Add & Norm after it. Ablation switches of `train.py`, stored in the adapter:
  - `--inject_layers`: indices and/or presets `all`, `every<N>` (default `every2`: layers 1, 3, ..., 17), `last<N>`,
    `first<N>`, `last_half`, `first_half`;
  - `--position`: `post_layer` (after the block, default) or `pre_mlp` (between the expert's joint attention and its
    MLP, so the block's own MLP processes the experience);
  - `--time_cond 1`: each cross-attention is conditioned on the flow-time embedding the expert feeds adaRMS
    (adaLN shift/scale of its input and a (1 + g) output gate, zero-initialized);
  - `--encoder v1|v2`: how references become the tokens the cross-attention reads (both keep the cross-attention,
    the input injection and the zero initialization):
    - v1 (default; 2.2M parameters outside the cross-attention): per reference, one token per chunk step (MLP of
      the action + MLP of the 6 scalars + success / step / slot embeddings); references never see each other.
    - v2 (8.8M; `--enc_dim 512 --enc_layers 2 --enc_heads 8 --act_segments 10`): per reference a summary token
      (scalars, success, its robot state and the state difference to the current call), 10 action-segment tokens
      and its 32 image tokens (2 cameras x 4 x 4 grid cells, shared projection); the current call adds its state
      token and image tokens; a 2-layer pre-norm transformer attends over all of them (references compared with
      each other and with the current scene), and the references' outputs are projected to the expert width. The
      input injection scores references from their summary outputs. Needs a library built with
      `build_library.py --features 1` (per call: image tokens, fp16, and normalized robot state; ~3.5 GB for all
      LIBERO demos); rollouts accumulated during evaluation keep them too.
- Training runs on the inference path (prefix KV cache without gradients, one denoising step at a sampled flow time),
  because the joint training forward of pi0.5 bypasses the expert layers' `forward` and therefore the hooks.
- Retrieval (`--retrieval` in train.py, `--pace.retrieval` in eval, default: the adapter's training value; it fixes
  the per-reference scalars, so an adapter is used with the retrieval it was trained with; independent of
  `--encoder`):
  - classic (v1 and v2 were trained so): candidates = the successful episodes stored under the exact instruction,
    aligned on image keys;
  - v3: candidates = the episodes of the `--text_tasks` tasks whose instruction is closest to the one the policy is
    told (`pi05.InstructionText`: pi0.5's own token embeddings, IDF-weighted, centered on the library's tasks) and of
    the `--scene_tasks` tasks with the closest first frames; the local alignment cost mixes in the robot state
    (`--state_weight`, `--state_sigma`), which a changed camera or lighting does not move; the ranking adds
    `--text_weight x (1 - instruction similarity)`; the adapter also reads the task similarity, whether an
    instruction was used and the state cost. Training withholds the instruction from a sample's retrieval with
    probability `--p_text_drop` and draws distractors / adherence negatives from the best-aligned candidates of other
    tasks (`--hard_distractor`); validation adds `notext`. In evaluation `--pace.retrieval_text shown` (default)
    compares what the policy is told (LIBERO-plus language variants are then a fair test), `true` the env's
    instruction (oracle), `none` nothing. Needs libraries with robot states (build_library always stores them now;
    libraries built with `--features 1` before have them too).
- Generic branch (v4, `--generic mlp`; independent of `--encoder` and `--retrieval`): without it the adapter acts only
  on samples that have references, so it can learn a plain, reference-independent fine-tune of pi0.5 gated by "there
  are references" (v1 / v2 validation: real - none ~ -0.016, of which real - other_task only ~ -0.001). The generic
  branch (`core.GenericResidual`: a zero-initialized, time-conditioned residual MLP of width `--generic_dim` at every
  injection layer (its flow-time projection low-rank, `--generic_cond_rank`), plus a per-step bias on the action tokens' input) acts on every sample, so that fine-tune goes
  there; `none` then means pi0.5 + the generic branch and `real - none` is what references add. Two stages:
  `--train_parts generic` (every sample without references, only the generic branch learns), then
  `--train_parts reference --init_from <stage-1 adapter>` (same configuration; generic frozen, only the reference
  branch learns); `all` = joint. Evaluation: `--pace.generic on` (default) / `off`; with `--pace.condition none`
  and `on` it is the generic-only control, so full vs generic-only isolates the references' effect in closed loop.
- Relevance gate (v6, `--relevance gate`; on top of the generic branch): the reference branch can decline its
  references. Without it, a reference branch trained on data its host does not fit (lerobot/libero_plus: the host's
  loss is ~3x what a content-free reference reaches) learns to use whatever it is given, so in closed loop it follows
  poorly matching experience. v6 gives every reference a weight g in (0, 1) from its summary (match quality included);
  log g biases the cross-attention and the input-injection score, the null token's value is zero (its attention logit
  gets log T, so it competes with a reference's T tokens as one option, as the injection's null does), the output /
  injection projections have no bias and the injection softmax has a learned null option, so with every g small the
  branch adds ~0 (host + generic). Training mixes good references (retrieved; `--p_near_self`: near-identical
  experience) and bad ones (`--p_misaligned`: read at random steps; `--p_distractor`: other tasks; `--p_sparse`: a
  sparse, loosely related store); the generic branch makes declining free. In stage 2 use `--p_none 0` (a sample
  without references has no path to the reference branch). Validation and `pace.val_probe` report the mean g
  (`relevance_<condition>`) and the injection's total weight on the references (`refmass_<condition>`) per condition.
  `eval.py` also has a hard gate, `--pace.max_image_cost` (skip references whose image cost at the matched step
  exceeds it; rows record `declined_frac`), off by default; comparisons between versions keep it off.
- Training on the policy's own rollouts (v7, `--rollouts`; `rollouts.py`): `eval_suite --record_frames` with
  `--pace.condition none --pace.accumulate` and an adapter with the v2 encoder runs the base policy and stores every
  rollout (keys, image tokens, states, executed chunks, outcome) plus every policy call's observation
  (`frames_<tag>.pt`), so rounds are repeated attempts of the same variants. `train.py --rollouts <dirs> --library
  <the evaluation's library>` then trains on the successful attempts (target: the executed chunk), holds out whole
  variants, reads the training variants' attempts as references (`--ref_failures 1`: failed ones too, with their
  outcome; the evaluation then retrieves failures as well, stored in the adapter) and keeps a sample's other attempts
  of its own variant with probability `--p_same_variant` (validation: `real` with them, `novariant` without). No
  generic branch: the host fits its own rollouts, so a relevance gate can decline for free. Training variants must
  exclude the evaluation's.
- Offline probes of a trained adapter (`pace.val_probe`): its run's held-out demos read with a perfect reference
  (`self`), near-identical scenes allowed (`near`), a fixed content-free reference (`fixed`), the top reference only,
  other-task references at random steps, next to train.py's own conditions (whose first 32 episodes reproduce its log).
- Several GPUs: `torchrun --nproc_per_node N -m pace.train ...` (data parallel over the adapter; each process draws its own
  batch, so a step is N x the batch of one GPU; process 0 validates, logs and saves).
- Not supported: `compile_model=True`, MEM memory inputs, RTC guidance with the adapter.
