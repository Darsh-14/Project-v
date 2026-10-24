"""Kaggle throughput pilot (Checkpoint 2, option A). Self-contained kernel script.

Clones the public repo, installs requirements.lock, records the hardware, then runs
scripts/bench_fps.py single and concurrently. Writes /kaggle/working/results/pilot/.
MODE is set by the folder's copy: "gpu" or "cpu".
"""

import os
import subprocess
import sys
import time

MODE = os.environ.get("PV_MODE", "cpu")
REPO = "https://github.com/Darsh-14/Project-v.git"
SRC = "/kaggle/tmp/pv"
OUT = "/kaggle/working/results/pilot"


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


os.makedirs(OUT, exist_ok=True)
sh(f"rm -rf {SRC} && git clone --depth 1 {REPO} {SRC}")
sh(f"cd {SRC} && pip install -q -r requirements.lock && pip install -q -e . --no-deps")

hw = subprocess.run(
    "python --version; nproc; lscpu | grep 'Model name'; free -g | head -2; "
    "nvidia-smi --query-gpu=name,memory.total --format=csv || true; "
    "python -c 'import torch;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.device_count())'",
    shell=True, capture_output=True, text=True,
).stdout
print(hw, flush=True)
open(f"{OUT}/hardware_{MODE}.txt", "w").write(hw)

env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
csv = f"{OUT}/fps_kaggle_{MODE}.csv"


def bench(label, exp, variants, device, n_procs, timed, warmup=2000):
    procs = []
    for i in range(n_procs):
        dev = device
        if device == "cuda-spread":
            import torch

            dev = f"cuda:{i % max(torch.cuda.device_count(), 1)}"
        cmd = [
            sys.executable, "scripts/bench_fps.py", "--exp", f"configs/exp/{exp}.yaml",
            "--variants", variants, "--device", dev, "--threads", "1", "--timed", str(timed),
            "--warmup", str(warmup), "--host", f"kaggle-{MODE}-{label}-p{n_procs}", "--out", csv,
        ]
        procs.append(subprocess.Popen(cmd, cwd=SRC, env=env))
    t = time.time()
    codes = [p.wait() for p in procs]
    print(f"[{label} x{n_procs}] exit={codes} wall={time.time() - t:.0f}s", flush=True)


if MODE == "gpu":
    bench("single", "push_dr", "2048x512x3", "cuda:0", 1, timed=1500)
    bench("drhist", "push_drhist", "2048x512x3", "cuda:0", 1, timed=1500)
    bench("share1gpu", "push_dr", "2048x512x3", "cuda:0", 2, timed=1500)
    bench("spread", "push_dr", "2048x512x3", "cuda-spread", 4, timed=1500)
else:
    bench("single", "push_dr", "256x256x3", "cpu", 1, timed=500)
    bench("conc", "push_dr", "256x256x3", "cpu", 3, timed=500)
    bench("drhist", "push_drhist", "256x256x3", "cpu", 1, timed=500)
    bench("zoo", "push_dr", "2048x512x3", "cpu", 1, timed=100)
    bench("reach", "reach_dr", "256x256x2", "cpu", 3, timed=1000)

print(open(csv).read(), flush=True)
