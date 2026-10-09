# PACE

Progress-Aligned Cross-episode Experience: a plug-in experience memory for VLA policies, starting with pi0.5
(LeRobot `PI05Policy`). The code is in [`pace/`](pace/README.md).

PACE runs against upstream LeRobot, which is not vendored here. Clone it next to `pace/` and install it:

```bash
git clone https://github.com/huggingface/lerobot.git   # developed against b9cb121c
pip install -e "lerobot[pi,libero]"
```

openpi (https://github.com/Physical-Intelligence/openpi, 215abfb2) is only a reference for the official pi0.5
configs and evaluation protocol.
