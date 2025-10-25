#!/usr/bin/env python3
"""
Plot box position (x,y,z) vs time from the CSV logged by pose_transformer.

Default CSV path:
  ~/ds_yash/bimanual_ws/src/fr3_controllers/dynamic_box_lifting/data_for_ee_force/utils/aruco_box_positions.csv
"""

import os
import csv
import argparse
import math
import matplotlib.pyplot as plt

DEFAULT_CSV = os.path.expanduser(
    "~/ds_yash/bimanual_ws/src/fr3_controllers/dynamic_box_lifting/data_for_ee_force/utils/aruco_box_positions.csv"
)

def load_csv(path):
    t, x, y, z = [], [], [], []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        need = {"time_sec", "x", "y", "z"}
        if not need.issubset(set(r.fieldnames or [])):
            raise ValueError(f"CSV missing required columns; found {r.fieldnames}, need {sorted(need)}")
        for row in r:
            try:
                t.append(float(row["time_sec"]))
                x.append(float(row["x"]))
                y.append(float(row["y"]))
                z.append(float(row["z"]))
            except ValueError:
                # skip malformed lines
                continue
    return t, x, y, z

def moving_avg(vals, k):
    if k <= 1:
        return vals[:]
    out = []
    s = 0.0
    from collections import deque
    q = deque()
    for v in vals:
        q.append(v); s += v
        if len(q) > k:
            s -= q.popleft()
        out.append(s / len(q))
    return out

def unit_scale(units: str):
    """
    Return (scale, label) where scale multiplies meters -> target units.
    Supported: m, cm, mm
    """
    u = units.lower()
    if u == "m":
        return 1.0, "m"
    if u == "cm":
        return 100.0, "cm"
    if u == "mm":
        return 1000.0, "mm"
    raise ValueError("units must be one of: m, cm, mm")

def main():
    ap = argparse.ArgumentParser(description="Plot x/y/z vs time from ArUco CSV")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="Path to CSV (default: %(default)s)")
    ap.add_argument("--units", default="cm", choices=["m", "cm", "mm"],
                    help="Position units to plot (default: %(default)s)")
    ap.add_argument("--smooth", type=int, default=1, help="Moving average window (samples). 1 = no smoothing")
    ap.add_argument("--start", type=float, default=None, help="Start time (sec) to plot from")
    ap.add_argument("--end", type=float, default=None, help="End time (sec) to plot to")
    ap.add_argument("--out", default="", help="Save plot to this PNG (if omitted, just shows the window)")
    ap.add_argument("--no-show", action="store_true", help="Do not open a window; useful on headless servers")
    args = ap.parse_args()

    path = os.path.expanduser(args.csv)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    t, x, y, z = load_csv(path)
    if not t:
        raise RuntimeError("No data rows found in CSV.")

    # Crop by time if requested
    if args.start is not None or args.end is not None:
        t0 = args.start if args.start is not None else -math.inf
        t1 = args.end   if args.end   is not None else  math.inf
        sel = [i for i, ti in enumerate(t) if t0 <= ti <= t1]
        if not sel:
            raise RuntimeError("No samples in the requested time range.")
        t = [t[i] for i in sel]
        x = [x[i] for i in sel]
        y = [y[i] for i in sel]
        z = [z[i] for i in sel]

    # Smoothing (on meters)
    x_s = moving_avg(x, args.smooth)
    y_s = moving_avg(y, args.smooth)
    z_s = moving_avg(z, args.smooth)

    # Units conversion
    scale, ulabel = unit_scale(args.units)
    x_s = [v * scale for v in x_s]
    y_s = [v * scale for v in y_s]
    z_s = [v * scale for v in z_s]

    # Plot
    plt.figure(figsize=(9, 5))
    plt.plot(t, x_s, label=f"x [{ulabel}]")
    plt.plot(t, y_s, label=f"y [{ulabel}]")
    plt.plot(t, z_s, label=f"z [{ulabel}]")
    plt.xlabel("time [s]")
    plt.ylabel(f"position [{ulabel}]")
    plt.title("Box position over time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    # Force y-axis from 0 to 50 cm
    plt.ylim(0, 50)
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"Saved plot → {args.out}")

    if not args.no_show or not args.out:
        plt.show()

if __name__ == "__main__":
    main()
