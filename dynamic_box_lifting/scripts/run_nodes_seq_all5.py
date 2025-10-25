#!/usr/bin/env python3
import os
import sys
import time
import subprocess
from pathlib import Path

# ================= CONFIG =================
WORKSPACE = os.path.expanduser(os.environ.get("ROS_WS", "~/ds_yash/bimanual_ws"))
PYTHON    = os.environ.get("PYTHON", "python3")

# Absolute paths to your scripts (edit if your filenames differ)
V_PATH = f"{WORKSPACE}/src/fr3_controllers/dynamic_box_lifting/data_for_ee_force/utils/auruco_obj_velocity.py"
P_PATH = f"{WORKSPACE}/src/fr3_controllers/dynamic_box_lifting/data_for_ee_force/utils/auruco_obj_position.py"
F_FILTER = f"{WORKSPACE}/src/fr3_controllers/dynamic_box_lifting/data_for_ee_force/scripts/fr3_wrench_filter.py"
F_RAW    = f"{WORKSPACE}/src/fr3_controllers/dynamic_box_lifting/data_for_ee_force/scripts/ft_raw_values.py"
F_WRENCH = f"{WORKSPACE}/src/fr3_controllers/dynamic_box_lifting/data_for_ee_force/utils/wrench.py"

LOGDIR = Path("/tmp/ros_seq_keep_py")
READY_TIMEOUT_SEC = 20.0  # how long to wait for each node to appear
# =========================================

def say(x): print(f"\033[1;36m{x}\033[0m")
def err(x): print(f"\033[1;31m{x}\033[0m", file=sys.stderr)

def file_ok(path, allow_missing=False):
    if not os.path.isfile(path):
        if allow_missing:
            say(f"SKIP (missing): {path}")
            return False
        err(f"Missing file: {path}")
        sys.exit(2)
    if not os.access(path, os.X_OK):
        err(f"Not executable: {path}  (fix: chmod +x '{path}')")
        sys.exit(2)
    return True

def bash_popen(cmd, env=None, log_file=None):
    # Detach group so Ctrl-C on launcher doesn't kill children
    stdout = open(log_file, "w") if log_file else subprocess.PIPE
    return subprocess.Popen(
        ["bash", "-lc", cmd],
        stdout=stdout,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setpgrp,
        env=env or os.environ.copy(),
    )

def rosnode_list():
    try:
        out = subprocess.check_output(["bash","-lc","rosnode list || true"], text=True, timeout=2)
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        return []

def wait_for_master(timeout=10.0):
    say("Waiting for ROS master…")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            subprocess.check_call(["bash","-lc","rosnode list >/dev/null 2>&1 || true"])
            return True
        except Exception:
            time.sleep(0.2)
    return True  # best effort

def start_roscore():
    LOGDIR.mkdir(parents=True, exist_ok=True)
    say("Starting roscore (detached)…")
    p = bash_popen("roscore", log_file=str(LOGDIR / "roscore.log"))
    if not wait_for_master(10.0):
        err("Warning: could not confirm ROS master; continuing.")
    return p

def source_cmd():
    return f"source {WORKSPACE}/devel/setup.bash"

def start_and_wait(label, node_fq, command, extra_env=None):
    """
    Start a node and wait until /node_fq shows up in `rosnode list`.
    Never stops the node once it's up.
    """
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGDIR / f"{label}.log"
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    say(f"Launching {label} → {node_fq}")
    p = bash_popen(f"{source_cmd()} && exec {command}", env=env, log_file=str(log_path))
    # Wait for node to appear or process to die
    t0 = time.time()
    while time.time() - t0 < READY_TIMEOUT_SEC:
        if p.poll() is not None:
            err(f"{label} exited early with code {p.returncode}. Check log: {log_path}")
            try:
                tail = (log_path.read_text().splitlines() or ["<empty>"])[-30:]
                print("\n".join(tail))
            except Exception:
                pass
            sys.exit(1)
        if node_fq in rosnode_list():
            say(f"{label} is up (PID {p.pid}). Logs → {log_path}")
            with open(LOGDIR / "pids.txt", "a") as f:
                f.write(f"{label}\tPID={p.pid}\tNODE={node_fq}\tLOG={log_path}\n")
            return p
        time.sleep(0.25)

    err(f"Timeout waiting for node {node_fq}. Check log: {log_path}")
    sys.exit(1)

def main():
    # Validate files (ArUco ones are optional individually, but at least one may be present)
    v_ok = file_ok(V_PATH, allow_missing=True)
    p_ok = file_ok(P_PATH, allow_missing=True)
    if not (v_ok or p_ok):
        say("No ArUco scripts found; continuing with the remaining nodes.")

    file_ok(F_FILTER)
    file_ok(F_RAW)
    file_ok(F_WRENCH)

    # Source the workspace once so the environment is sane
    rc = subprocess.call(["bash","-lc", source_cmd()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rc != 0:
        err(f"Could not source {WORKSPACE}/devel/setup.bash")
        sys.exit(2)

    # Start roscore
    start_roscore()

    # Start each node sequentially, never stopping them afterwards.
    # 1) ArUco velocity under /vel namespace (node will be /vel/pose_transformer)
    if v_ok:
        start_and_wait(
            "aruco_velocity",
            "/vel/pose_transformer",
            f"env ROS_NAMESPACE=/vel {PYTHON} {V_PATH}",
        )

    # 2) ArUco position under /pos namespace (node will be /pos/pose_transformer)
    if p_ok:
        start_and_wait(
            "aruco_position",
            "/pos/pose_transformer",
            f"env ROS_NAMESPACE=/pos {PYTHON} {P_PATH}",
        )

    # 3) Wrench filter
    start_and_wait(
        "fr3_wrench_filter",
        "/fr3_wrench_filter",
        f"{PYTHON} {F_FILTER}",
    )

    # 4) Raw FT publisher
    start_and_wait(
        "robotous_ft_publisher",
        "/robotous_ft_publisher",
        f"{PYTHON} {F_RAW}",
    )

    # 5) Wrench aggregator
    start_and_wait(
        "wrench_aggregator",
        "/wrench_aggregator",
        f"{PYTHON} {F_WRENCH}",
    )

    say("All nodes launched and running. PIDs recorded in /tmp/ros_seq_keep_py/pids.txt")
    say("This launcher will stay alive; Ctrl-C will exit the launcher, nodes keep running.")

    # Keep the launcher alive without sending signals to children
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        say("Launcher exiting. Nodes remain running.")
        sys.exit(0)

if __name__ == "__main__":
    main()
