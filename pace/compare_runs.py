"""
Paired comparison of evaluation runs (pace.eval_suite out_dirs, or pace.eval / lerobot-eval out_dirs with a
pace_episodes.jsonl). Episodes are paired by (round, task_id, init_index) for eval_suite rows and by (task, episode)
for pace.eval rows, so compare runs made with the same task list, --seed and --batch_size.

For every run and round: success rate (and per category with --by category); for every run against the reference run
(the first one) and every round against the same run's round 1: the episodes only one side solved and McNemar's exact
two-sided p-value (binomial test on the discordant pairs). Rounds of an accumulating run (--pace.accumulate) show
whether experience helps as it grows; a base run with --rounds gives the run-to-run noise of the same comparison.

    python -m pace.compare_runs runs/eval/plus_base runs/eval/plus_v2_acc runs/eval/plus_v2_demo --by category
"""

import argparse
import collections
import json
import math
from pathlib import Path


def load_rows(run: str):
    path = Path(run)
    for name in ("episodes.jsonl", "pace_episodes.jsonl"):
        if (path / name).exists():
            return [json.loads(line) for line in open(path / name)]
    raise SystemExit(f"{run}: no episodes.jsonl / pace_episodes.jsonl")


def key_of(row):
    if "task_id" in row:
        return row.get("round", 1), row["task_id"], row.get("init_index", 0)
    return row.get("round", 1), row["task"], row["episode"]


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value: b, c = discordant pairs (binomial(b + c, 1/2))."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(a: dict, b: dict, keys) -> dict:
    keys = [k for k in keys if k in a and k in b]
    both = sum(a[k] and b[k] for k in keys)
    a_only = sum(a[k] and not b[k] for k in keys)
    b_only = sum(b[k] and not a[k] for k in keys)
    return {"n": len(keys), "both": both, "ref_only": a_only, "run_only": b_only,
            "diff_pct": 100 * (b_only - a_only) / max(len(keys), 1), "p": mcnemar_p(a_only, b_only)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="Run directories; the first is the reference")
    parser.add_argument("--by", choices=("none", "category"), default="none")
    parser.add_argument("--out", default=None, help="Write the tables as JSON")
    args = parser.parse_args()

    runs = {}
    for run in args.runs:
        rows = load_rows(run)
        table = {}
        for row in rows:
            table[key_of(row)] = row
        runs[run] = table
    ref = args.runs[0]
    report = {"runs": {}, "vs_reference": {}, "vs_round1": {}}

    def groups(table):
        out = collections.defaultdict(list)
        for k, row in table.items():
            out[(k[0], "all")].append(k)
            if args.by == "category":
                out[(k[0], row.get("category", "?"))].append(k)
        return out

    print(f"{'run':48s} {'round':>5s} {'group':28s} {'n':>5s} {'succ%':>7s} {'rollout_top':>11s}")
    for run, table in runs.items():
        for (rnd, group), keys in sorted(groups(table).items(), key=lambda x: (x[0][0], x[0][1] != "all", x[0][1])):
            succ = [table[k]["success"] for k in keys]
            frac = [table[k]["top_ref_rollout_frac"] for k in keys if "top_ref_rollout_frac" in table[k]]
            row = {"n": len(keys), "success": 100 * sum(succ) / len(keys),
                   "top_ref_rollout_frac": sum(frac) / len(frac) if frac else None}
            report["runs"].setdefault(run, {})[f"{rnd}/{group}"] = row
            print(f"{run[-48:]:48s} {rnd:5d} {group[:28]:28s} {row['n']:5d} {row['success']:7.1f} "
                  f"{'' if row['top_ref_rollout_frac'] is None else format(row['top_ref_rollout_frac'], '11.3f'):>11s}")

    def compare(title, pairs, store):
        print(f"\n{title}")
        print(f"{'pair':72s} {'group':28s} {'n':>5s} {'ref_only':>8s} {'run_only':>8s} {'diff%':>7s} {'p':>7s}")
        for name, a, b, rnd_a, rnd_b in pairs:
            ga, gb = groups(a), groups(b)
            for (rnd, group), keys in sorted(ga.items()):
                if rnd != rnd_a:
                    continue
                other = gb.get((rnd_b, group))
                if not other:
                    continue
                left = {k[1:]: a[k]["success"] for k in keys}
                right = {k[1:]: b[k]["success"] for k in other}
                res = paired(left, right, left.keys())
                store[f"{name}/{group}"] = res
                print(f"{name[-72:]:72s} {group[:28]:28s} {res['n']:5d} {res['ref_only']:8d} {res['run_only']:8d} "
                      f"{res['diff_pct']:7.1f} {res['p']:7.3f}")

    rounds_of = {run: sorted({k[0] for k in t}) for run, t in runs.items()}
    pairs = []
    for run in args.runs[1:]:
        for rnd in rounds_of[run]:
            ref_round = rnd if rnd in rounds_of[ref] else rounds_of[ref][0]
            pairs.append((f"{Path(ref).name} r{ref_round} -> {Path(run).name} r{rnd}", runs[ref], runs[run],
                          ref_round, rnd))
    if pairs:
        compare("vs reference (paired by episode)", pairs, report["vs_reference"])
    pairs = [(f"{Path(run).name} r1 -> r{rnd}", runs[run], runs[run], 1, rnd)
             for run in args.runs for rnd in rounds_of[run] if rnd > 1 and 1 in rounds_of[run]]
    if pairs:
        compare("later rounds vs round 1 of the same run", pairs, report["vs_round1"])
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
