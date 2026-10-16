"""
Training on the policy's own rollouts (v7): what an evaluation run records and how pace/train.py reads it back.

Recording (pace.eval_suite --record_frames, with --pace.accumulate): every policy call's observation as the policy
preprocessor receives it (the env preprocessor's output: images flipped, state as 8 floats) is kept by `FrameRecorder`,
and each batch's episodes are written next to their rollout file: <out_dir>/rollouts/frames_<tag>.pt holds, per stored
rollout name, its calls' images (uint8 [n, 3, H, W] per camera key, lossless: the env images are k / 255) and robot
states (float [n, Ds]) and the instruction the policy was told. The rollout file (rollouts_<tag>.pt) already holds
the rest: per call the retrieval key, image tokens, normalized robot state and the executed action chunk (the actions
the episode actually ran from that call on, `core.executed_chunks`), plus its outcome. Run it with
--pace.condition none and an adapter with the v2 encoder (so image tokens are recorded): the policy then runs exactly
as the base policy. With --pace.write_lock (a lock file shared by the collectors writing to one disk) every batch's two
files are written and flushed to the disk under that lock (`exclusive_writes`), one collector at a time, and with
--pace.write_gap at least that many seconds after the previous collector's write ended.

Reading (`RolloutFrames`): a dataset over the recorded calls whose items look like a LeRobotDataset frame without the
action (the training target is the rollout's executed chunk, which pace/train.py takes from the store).
"""

import contextlib
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch

IMAGE_PREFIX = "observation.images."
STATE_KEY = "observation.state"


class FrameRecorder:
    """
    The env preprocessor, keeping its output at every `every`-th env step (the policy calls; LIBERO: every
    n_action_steps steps from step 0). One recorder per rollout batch.
    """

    def __init__(self, inner, every: int) -> None:
        self.inner, self.every = inner, int(every)
        self.step = 0
        self.calls: List[Dict[str, torch.Tensor]] = []  # per recorded step: key -> [B, ...] (CPU)

    def __call__(self, observation):
        out = self.inner(observation)
        if self.step % self.every == 0:
            frame = {}
            for key, value in out.items():
                if key.startswith(IMAGE_PREFIX):
                    frame[key] = to_uint8(value.detach())
            frame[STATE_KEY] = out[STATE_KEY].detach().float().cpu()
            self.calls.append(frame)
        self.step += 1
        return out

    def episodes(self, rows: Sequence[int], num_calls: Sequence[int]) -> List[Dict[str, torch.Tensor]]:
        """Per batch row rows[k]: its first num_calls[k] recorded calls, key -> [n, ...]."""
        out = []
        for b, n in zip(rows, num_calls):
            assert n <= len(self.calls), f"row {b}: {n} policy calls but only {len(self.calls)} recorded steps"
            out.append({key: torch.stack([call[key][b] for call in self.calls[:n]]) for key in self.calls[0]})
        return out

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    """Float images in [0, 1] (k / 255, as the env produces them) -> uint8, refusing anything that would lose data."""
    if images.dtype == torch.uint8:
        return images.cpu()
    scaled = images.float().cpu() * 255.0
    rounded = scaled.round()
    err = float((scaled - rounded).abs().max()) if scaled.numel() else 0.0
    if err > 1e-2 or float(rounded.min()) < 0 or float(rounded.max()) > 255:
        raise ValueError(f"images are not k / 255 values in [0, 1] (max rounding error {err:.3g}); cannot store as uint8")
    return rounded.to(torch.uint8)


def save_frames(path, names: Sequence[str], frames: Sequence[Dict[str, torch.Tensor]], tasks: Sequence[str],
                sync: bool = False) -> Path:
    """
    Write one batch's recorded calls (`FrameRecorder.episodes`) to `path` (a complete file or none); with `sync` the
    file is on the disk when this returns (`fsync`), not only in the page cache.
    """
    path = Path(path)
    episodes = {}
    for name, frame, task in zip(names, frames, tasks):
        episodes[name] = {"images": {k: v.contiguous() for k, v in frame.items() if k.startswith(IMAGE_PREFIX)},
                          "state": frame[STATE_KEY].contiguous(), "task": task}
    tmp = path.with_suffix(".tmp")
    torch.save({"version": 1, "episodes": episodes}, tmp)
    if sync:
        fsync_path(tmp)
    os.replace(tmp, path)
    if sync:
        fsync_path(path.parent)
    return path


def fsync_path(path) -> None:
    """Flush a file (or a directory's entries) from the page cache to the disk."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def exclusive_writes(lock_path, gap: float = 0.0):
    """
    Hold an exclusive lock on `lock_path` (a file shared by the processes that write to the same disk; None: no lock),
    so their large writes, flushed inside the lock (`sync`), reach the disk one process at a time instead of at once.
    With `gap` (seconds) a writer also waits until `gap` seconds have passed since the previous writer finished (its
    end time is kept in the lock file), so the disk is quiet between two writes.
    """
    if not lock_path:
        yield
        return
    import fcntl

    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            if gap > 0:
                f.seek(0)
                try:
                    last_end = float(f.read().strip() or 0.0)
                except ValueError:
                    last_end = 0.0
                wait = last_end + gap - time.time()
                if wait > 0:
                    time.sleep(min(wait, gap))
            yield
            if gap > 0:
                f.seek(0)
                f.truncate()
                f.write(f"{time.time():.3f}\n")
                f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def frame_files(rollout_dirs: Sequence) -> List[Path]:
    files = []
    for d in rollout_dirs:
        files += sorted(Path(d).glob("frames_*.pt"))
    return files


def rollout_store_files(rollout_dirs: Sequence) -> List[Path]:
    files = []
    for d in rollout_dirs:
        files += sorted(Path(d).glob("rollouts_*.pt"))
    return files


class RolloutFrames(torch.utils.data.Dataset):
    """
    The calls recorded in frames_*.pt files: item i = one call, {camera key: float [3, H, W] in [0, 1], state, task}
    (as a LeRobotDataset frame reaches the policy preprocessor, minus the action). `index[name]` = the global item
    indices of a rollout's calls, in order. A name recorded twice (a resumed batch) keeps its last recording, as the
    store keeps the last rollout of that name. Files are memory-mapped, opened lazily in each process.
    """

    def __init__(self, files: Sequence) -> None:
        self.files = [str(f) for f in files]
        latest = {}
        for f_i, f in enumerate(self.files):
            data = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
            for name, ep in data["episodes"].items():
                latest[name] = (f_i, int(ep["state"].shape[0]))
        self.items, self.index = [], {}
        for name, (f_i, n) in latest.items():
            self.index[name] = list(range(len(self.items), len(self.items) + n))
            self.items += [(f_i, name, c) for c in range(n)]
        self._open: Dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _episode(self, f_i: int, name: str) -> dict:
        if f_i not in self._open:
            self._open[f_i] = torch.load(self.files[f_i], map_location="cpu", weights_only=False, mmap=True)
        return self._open[f_i]["episodes"][name]

    def __getitem__(self, i):
        f_i, name, c = self.items[i]
        ep = self._episode(f_i, name)
        item = {key: images[c].float() / 255.0 for key, images in ep["images"].items()}
        item[STATE_KEY] = ep["state"][c].float()
        item["task"] = ep["task"]
        return item

    def __getstate__(self):  # DataLoader workers open their own maps
        state = dict(self.__dict__)
        state["_open"] = {}
        return state


def rekey(rollout_dirs: Sequence, library: str) -> None:
    """
    Rollouts recorded without the library (eval_suite --pace.record_features, no --pace.library) carry untransformed
    grid keys. The grid key of a call is its image tokens flattened (`pi05.raw_keys` with key "grid" = `vis_tokens`:
    every camera's 4 x 4 cells, each unit-normalized, cameras in order), so the library's keys follow from the stored
    image tokens with the library's fitted transform, as build_library computes them (fp16 raw key, then the
    transform). Each rollouts_*.pt is rewritten with those keys and the library's key info; the original is kept as
    rollouts_*.pt.rawkeys. Frames are unchanged.
    """
    from .core import ExperienceStore, apply_key_transform

    lib = ExperienceStore.load(library)
    assert lib.info.get("key") == "grid", f"{library}: key {lib.info.get('key')} (only grid keys are image tokens)"
    for path in rollout_store_files(rollout_dirs):
        store = ExperienceStore.load(path, mmap=False)
        if store.info == lib.info:
            print(f"[rekey] {path.name}: already keyed like the library")
            continue
        assert store.info.get("transform") in (None, "none") and store.info.get("key") == "grid", \
            f"{path}: key info {store.info} (expected untransformed grid keys)"
        for ep in store.episodes:
            raw = ep["vis"].reshape(ep["vis"].shape[0], -1)  # [T, 16 * cameras * D] fp16
            stored = torch.nn.functional.normalize(ep["keys"].float(), dim=-1)
            cos = float((torch.nn.functional.normalize(raw.float(), dim=-1) * stored).sum(-1).min())
            assert cos > 0.999, f"{path.name} {ep['name']}: stored keys are not the image tokens (min cos {cos:.4f})"
            ep["keys"] = apply_key_transform(raw.half(), lib.key_transform).half()
        os.replace(path, path.with_name(path.name + ".rawkeys"))
        ExperienceStore(store.episodes, lib.info).save(path)
        print(f"[rekey] {path.name}: {len(store.episodes)} episodes -> keys {tuple(store.episodes[0]['keys'].shape[1:])}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="v7 rollouts: rekey rollouts recorded without the library (see rekey)")
    ap.add_argument("command", choices=["rekey"])
    ap.add_argument("--rollouts", nargs="+", required=True, help="<out_dir>/rollouts directories")
    ap.add_argument("--library", required=True, help="The evaluation's library (its fitted key transform)")
    cli = ap.parse_args()
    rekey(cli.rollouts, cli.library)
