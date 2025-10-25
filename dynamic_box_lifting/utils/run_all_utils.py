#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import signal
import subprocess

# --- CONFIG ---
WS_SETUP = os.path.expanduser("~/ds_yash/bimanual_ws/devel/setup.bash")
PY = sys.executable
START_DELAY = 3.0   # seconds between launches

SCRIPTS = [
    ("pos",  "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/utils/auruco_obj_position.py"),
    ("vel",  "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/utils/auruco_obj_velocity.py"),
    ("filt", "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/utils/fr3_wrench_filter.py"),
    ("ft",   "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/utils/ft_raw_values.py"),
    ("wr",   "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/utils/wrench.py"),
]
# -----------

def launch(label, path):
    if not os.path.isfile(path):
        print(f"[{label}] ERROR: not found: {path}")
        return None
    cmd = f'source "{WS_SETUP}" && {PY} "{path}"'
    print(f"→ Starting {label}: {os.path.basename(path)}")
    p = subprocess.Popen(
        ["bash", "-lc", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return p

def kill_all(procs):
    print("\nStopping all nodes …")
    for label, p in procs:
        if p and p.poll() is None:
            try:
                print(f"  SIGINT -> {label}")
                p.send_signal(signal.SIGINT)
            except Exception:
                pass
    time.sleep(2.0)
    for label, p in procs:
        if p and p.poll() is None:
            try:
                print(f"  SIGTERM -> {label}")
                p.terminate()
            except Exception:
                pass
    time.sleep(2.0)
    for label, p in procs:
        if p and p.poll() is None:
            try:
                print(f"  SIGKILL -> {label}")
                p.kill()
            except Exception:
                pass

def main():
    procs = []
    try:
        for label, path in SCRIPTS:
            p = launch(label, path)
            if p is None:
                print("Aborting sequence due to missing file.")
                break
            procs.append((label, p))
            time.sleep(START_DELAY)

        print("\nAll nodes launched. Press Ctrl-C to stop them.\n")

        while True:
            alive = any(p.poll() is None for _, p in procs)
            if not alive:
                print("All nodes have exited.")
                break
            time.sleep(1.0)

    except KeyboardInterrupt:
        pass
    finally:
        kill_all(procs)

if __name__ == "__main__":
    main()
