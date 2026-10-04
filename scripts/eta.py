"""Rough time-remaining estimate for scripts/run_minimal.sh, read from the logs.

  python scripts/eta.py
"""
import re
import statistics as st
from pathlib import Path

LOGS = {"GPU0 SegNet": Path("logs/gpu0_segnet.log"), "GPU1 DenseNet121": Path("logs/gpu1_densenet121.log")}
CHAIN = ["seg", "cls", "tune", "opt", "fl"]


def hm(sec):
    sec = max(0, int(sec))
    return f"{sec // 3600}h {sec % 3600 // 60:02d}m"


def tune_trials_total():
    """Hyperband's trial count is fixed by max_epochs/factor; use the finished SegNet search if available."""
    log = LOGS["GPU0 SegNet"]
    if log.exists() and Path("results/tune_seg.done.json").exists():
        t = log.read_text(errors="ignore")
        n = len(re.findall(r"Trial \d+ Complete", t)) + len(re.findall(r"ResourceExhaustedError", t))
        if n:
            return n
    return 96


def report(name, log):
    if not log.exists():
        print(f"{name}: no log yet")
        return
    t = log.read_text(errors="ignore")
    stages = re.findall(r"GPU\d (?:seg|cls): (\w+)", t)
    if "finished" in t.splitlines()[-1] if t.strip() else False:
        print(f"{name}: finished")
        return
    stage = stages[-1] if stages else "?"
    seg_t = t[t.rfind(f": {stage}"):] if stages else t          # log text of the current stage
    ep_times = [int(x) for x in re.findall(r"^\d+/\d+ - (\d+)s", seg_t, re.M)][1:]  # skip compile epoch
    ep = st.median(ep_times[-10:]) if ep_times else None
    print(f"{name}: current stage = {stage}")

    if stage == "tune":
        durs = [int(h) * 3600 + int(m) * 60 + int(s)
                for h, m, s in re.findall(r"Trial \d+ Complete \[(\d+)h (\d+)m (\d+)s\]", seg_t)]
        done, total = len(durs), tune_trials_total()
        if durs:
            # later Hyperband trials train longer, so weight toward recent trial durations
            per = st.mean(durs[-15:])
            left = max(0, total - done) * per
            print(f"   trials {done}/~{total}, recent trial ≈ {hm(per)}, tuning left ≈ {hm(left)}")
        else:
            left = 0
            print("   first trial still running")
        after = ("opt", "fl")
    elif stage == "opt":
        m = re.findall(r"Epoch (\d+)/(\d+)", seg_t)
        k, n = (int(m[-1][0]), int(m[-1][1])) if m else (0, 100)
        left = (n - k) * ep if ep else 0
        print(f"   epoch {k}/{n} at ≈{ep}s/epoch -> training left ≤ {hm(left)} "
              f"(early stopping usually ends sooner) + quantization eval ≈ 0h30m–1h")
        left += 2700
        after = ("fl",)
    elif stage == "fl":
        m = re.findall(r"\[fl\] \w+ round (\d+)/(\d+)", seg_t)
        r, n = (int(m[-1][0]), int(m[-1][1])) if m else (0, 50)
        left = (n - r) * (ep or 60) * 1.1 + 1800
        print(f"   round {r}/{n} -> FL left ≈ {hm(left)} (incl. final INT8 evaluation)")
        after = ()
    else:
        left, after = 0, ("tune", "opt", "fl")
        if ep:
            m = re.findall(r"Epoch (\d+)/(\d+)", seg_t)
            if m:
                k, n = int(m[-1][0]), int(m[-1][1])
                left = (n - k) * ep
                print(f"   epoch {k}/{n} at ≈{ep}s/epoch -> stage left ≤ {hm(left)}")

    rough = {"tune": 6 * 3600, "opt": 3 * 3600, "fl": 2 * 3600}
    later = sum(rough[s] for s in after)
    if after:
        print(f"   then {' + '.join(after)}: very roughly {hm(later)}")
    print(f"   => estimated time to finish this GPU: ~{hm(left + later)}\n")


if __name__ == "__main__":
    for name, log in LOGS.items():
        report(name, log)
