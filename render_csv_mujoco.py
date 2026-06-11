#!/usr/bin/env python3
"""Render normalized Panda joint CSV files to MP4 using MuJoCo."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

LOCAL_PYDEPS = Path(__file__).resolve().parents[3] / "mujoco_pydeps"
if LOCAL_PYDEPS.exists():
    sys.path.insert(0, str(LOCAL_PYDEPS))

import cv2
import mujoco
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_DIR = SCRIPT_DIR / "DP_experiment" / "Re__CSV_format_slow_smooth"
DEFAULT_MENAGERIE_MODEL = (
    SCRIPT_DIR
    / "DP_experiment"
    / "robot_descriptions_cache"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)
DEFAULT_MODEL = (
    DEFAULT_MENAGERIE_MODEL
    if DEFAULT_MENAGERIE_MODEL.exists()
    else SCRIPT_DIR / "DP_experiment" / "render_assets" / "panda_abs.urdf"
)
DEFAULT_OUT_DIR = SCRIPT_DIR / "DP_experiment" / "render_mp4"


def read_player_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        required = ["timestamp"] + [f"panda_joint{i}_pos" for i in range(1, 8)]
        missing = [column for column in required if column not in header]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

        times = []
        positions = []
        for row in reader:
            times.append(float(row["timestamp"]))
            positions.append([float(row[f"panda_joint{i}_pos"]) for i in range(1, 8)])

    times_arr = np.asarray(times, dtype=float)
    positions_arr = np.asarray(positions, dtype=float)
    if len(times_arr) < 2:
        raise ValueError(f"{path} needs at least two rows")
    if np.any(np.diff(times_arr) <= 0):
        raise ValueError(f"{path} has non-increasing timestamps")
    return times_arr, positions_arr


def sample_positions(times: np.ndarray, positions: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    duration = float(times[-1] - times[0])
    frame_count = int(np.floor(duration * fps)) + 1
    frame_times = np.arange(frame_count, dtype=float) / fps
    frame_times[-1] = min(frame_times[-1], duration)

    sampled = np.empty((len(frame_times), positions.shape[1]), dtype=float)
    source_times = times - times[0]
    for joint_idx in range(positions.shape[1]):
        sampled[:, joint_idx] = np.interp(frame_times, source_times, positions[:, joint_idx])
    return frame_times, sampled


def build_joint_address_map(model: mujoco.MjModel) -> dict[str, int]:
    joint_to_qpos = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name:
            joint_to_qpos[name] = int(model.jnt_qposadr[joint_id])
    return joint_to_qpos


def resolve_arm_joint_addresses(joint_to_qpos: dict[str, int]) -> list[int]:
    addresses = []
    missing = []
    for i in range(1, 8):
        aliases = [f"panda_joint{i}", f"joint{i}"]
        for alias in aliases:
            if alias in joint_to_qpos:
                addresses.append(joint_to_qpos[alias])
                break
        else:
            missing.append("/".join(aliases))
    if missing:
        raise ValueError(f"Model is missing arm joints: {missing}")
    return addresses


def configure_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.15, 0.0, 0.55]
    camera.distance = 2.2
    camera.azimuth = 135.0
    camera.elevation = -25.0
    return camera


def configure_visuals(model: mujoco.MjModel) -> None:
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:] = [0.85, 0.85, 0.85]
    model.vis.headlight.specular[:] = [0.25, 0.25, 0.25]


def configure_scene_option() -> mujoco.MjvOption:
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[:] = 1
    # MuJoCo Menagerie puts visual meshes in group 2 and collision meshes in
    # group 3. Hiding group 3 gives a cleaner inspection video.
    scene_option.geomgroup[3] = 0
    return scene_option


def render_csv(
    csv_path: Path,
    model: mujoco.MjModel,
    out_path: Path,
    fps: float,
    width: int,
    height: int,
) -> None:
    times, positions = read_player_csv(csv_path)
    frame_times, frame_positions = sample_positions(times, positions, fps)

    data = mujoco.MjData(model)
    joint_to_qpos = build_joint_address_map(model)
    arm_qpos_addresses = resolve_arm_joint_addresses(joint_to_qpos)
    for finger_name in ("finger_joint1", "finger_joint2", "panda_finger_joint1", "panda_finger_joint2"):
        if finger_name in joint_to_qpos:
            data.qpos[joint_to_qpos[finger_name]] = 0.04

    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = configure_camera()
    scene_option = configure_scene_option()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = float(frame_times[-1])
    for frame_idx, (t, q) in enumerate(zip(frame_times, frame_positions)):
        for qpos_address, value in zip(arm_qpos_addresses, q):
            data.qpos[qpos_address] = value
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        rgb = renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.convertScaleAbs(bgr, alpha=1.35, beta=18)

        label = f"{csv_path.name}  t={t:05.2f}s / {duration:05.2f}s"
        cv2.putText(
            bgr,
            label,
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr,
            label,
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        writer.write(bgr)

        if frame_idx and frame_idx % int(max(fps * 10, 1)) == 0:
            print(f"  {csv_path.name}: frame {frame_idx}/{len(frame_times)}")

    writer.release()
    renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--only", nargs="*", default=None, help="Optional CSV basenames to render")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Missing MuJoCo/URDF model: {args.model}")

    csv_paths = sorted(args.csv_dir.glob("*.csv"))
    if args.only:
        wanted = set(args.only)
        csv_paths = [path for path in csv_paths if path.name in wanted]
    if not csv_paths:
        raise SystemExit(f"No CSV files to render in {args.csv_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    configure_visuals(model)
    print(f"model={args.model} nq={model.nq} nv={model.nv} joints={model.njnt}")

    for csv_path in csv_paths:
        out_path = args.out_dir / f"{csv_path.stem}.mp4"
        print(f"rendering {csv_path} -> {out_path}")
        render_csv(csv_path, model, out_path, args.fps, args.width, args.height)


if __name__ == "__main__":
    main()
