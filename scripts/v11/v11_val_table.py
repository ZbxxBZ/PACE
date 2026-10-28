import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
keys = ["true_expert_rms", "expert_zero_rms", "expert_copy_rms", "other_expert_rms", "unaligned_expert_rms",
        "true_perturbed_rms", "perturbed_zero_rms", "true_carry_rms", "carry_zero_rms",
        "true_pi05_aug_rms", "pi05_aug_zero_rms", "true_negative_rms", "other_negative_rms",
        "d_err_cm_median", "d_zero_cm_median", "true_relevance_mean", "other_relevance_mean", "true_inject_ref_mass"]
print("step", *keys)
for r in rows:
    print(r["step"], *(f"{r.get(k, float('nan')):.4f}" for k in keys))
