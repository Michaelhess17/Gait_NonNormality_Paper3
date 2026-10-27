"""Convert selected loco-mujoco-datasets mocap .npz files to CSV joint-angle trials.

Source: https://huggingface.co/datasets/robfiras/loco-mujoco-datasets
        DefaultDatasets/mocap/{Robot}/{motion}.npz
License: CC-BY-NC-ND-4.0 (non-commercial, no-derivatives on the *dataset itself*;
         using it to compute/plot derived metrics for a paper is fine, but do not
         redistribute modified copies of the raw mocap data).

Each npz stores MuJoCo qpos for a free-floating base (7 dof: xyz + quat) followed by
one column per hinge joint, in the order given by `joint_names` (joint_names[0] is the
free "root"/"reference" joint). So qpos column for joint_names[j] (j >= 1) is 6 + j.

We pick a matched set of leg joint names per robot (hip flexion/adduction/rotation or
pitch/roll/yaw, knee, ankle) so the same walking task can be compared across robots
with different mechanical DOF counts.
"""
import json
import os

import numpy as np

# robot -> ordered list of (channel_label, joint_name) to extract, left then right
CHANNEL_MAP = {
    "UnitreeH1": [
        ("hip_rotation_l", "hip_rotation_l"),
        ("hip_adduction_l", "hip_adduction_l"),
        ("hip_flexion_l", "hip_flexion_l"),
        ("knee_l", "knee_angle_l"),
        ("ankle_l", "ankle_angle_l"),
        ("hip_rotation_r", "hip_rotation_r"),
        ("hip_adduction_r", "hip_adduction_r"),
        ("hip_flexion_r", "hip_flexion_r"),
        ("knee_r", "knee_angle_r"),
        ("ankle_r", "ankle_angle_r"),
    ],
    "Atlas": [
        ("hip_rotation_l", "hip_rotation_l"),
        ("hip_adduction_l", "hip_adduction_l"),
        ("hip_flexion_l", "hip_flexion_l"),
        ("knee_l", "knee_angle_l"),
        ("ankle_l", "ankle_angle_l"),
        ("ankle_abduction_l", "l_leg_akx"),
        ("hip_rotation_r", "hip_rotation_r"),
        ("hip_adduction_r", "hip_adduction_r"),
        ("hip_flexion_r", "hip_flexion_r"),
        ("knee_r", "knee_angle_r"),
        ("ankle_r", "ankle_angle_r"),
        ("ankle_abduction_r", "r_leg_akx"),
    ],
    "UnitreeG1": [
        ("hip_pitch_l", "left_hip_pitch_joint"),
        ("hip_roll_l", "left_hip_roll_joint"),
        ("hip_yaw_l", "left_hip_yaw_joint"),
        ("knee_l", "left_knee_joint"),
        ("ankle_pitch_l", "left_ankle_pitch_joint"),
        ("ankle_roll_l", "left_ankle_roll_joint"),
        ("hip_pitch_r", "right_hip_pitch_joint"),
        ("hip_roll_r", "right_hip_roll_joint"),
        ("hip_yaw_r", "right_hip_yaw_joint"),
        ("knee_r", "right_knee_joint"),
        ("ankle_pitch_r", "right_ankle_pitch_joint"),
        ("ankle_roll_r", "right_ankle_roll_joint"),
    ],
}

MOTIONS = {
    "UnitreeH1": ["walk", "run", "random_walk", "stepinplace1"],
    "Atlas": ["walk", "run"],
    "UnitreeG1": ["walk"],
}


def qpos_column_for(joint_names, jnt_type, name):
    assert jnt_type[0] == 0, "expected a free root joint"
    assert set(jnt_type[1:].tolist()) == {3}, "expected all-hinge remaining joints"
    j = list(joint_names).index(name)
    assert j >= 1, f"{name} should not be the root joint"
    return 6 + j


def convert(robot, motion, outdir):
    src = os.path.join("raw", robot, f"{motion}.npz")
    d = np.load(src, allow_pickle=True)
    joint_names = list(d["joint_names"])
    jnt_type = d["jnt_type"]
    qpos = d["qpos"]
    freq = float(d["frequency"])

    channels = CHANNEL_MAP[robot]
    cols = [qpos_column_for(joint_names, jnt_type, jn) for _, jn in channels]
    labels = [lab for lab, _ in channels]
    X = qpos[:, cols]

    # Base-frame translational speed (m/s), from finite-differenced free-joint
    # xy position, smoothed over ~1 stride so it reflects walking speed rather
    # than frame-to-frame jitter. This is metadata, not a DMD input channel.
    base_xy = qpos[:, 0:2]
    raw_speed = np.linalg.norm(np.diff(base_xy, axis=0), axis=1) * freq
    raw_speed = np.concatenate([raw_speed, raw_speed[-1:]])  # pad to n_samples
    stride_win = max(1, round(freq * 1.1))  # ~1 stride at ~1.1 s period
    kernel = np.ones(stride_win) / stride_win
    speed = np.convolve(raw_speed, kernel, mode="same")

    X_out = np.column_stack([X, speed])
    out_labels = labels + ["speed_mps"]

    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"{robot}_{motion}.csv")
    header = ",".join(out_labels)
    np.savetxt(csv_path, X_out, delimiter=",", header=header, comments="", fmt="%.8f")

    meta = {
        "robot": robot,
        "motion": motion,
        "frequency_hz": freq,
        "n_samples": int(X.shape[0]),
        "n_channels": int(X.shape[1]),
        "channels": labels,
        "non_dmd_columns": ["speed_mps"],
        "source": f"DefaultDatasets/mocap/{robot}/{motion}.npz",
        "repo": "robfiras/loco-mujoco-datasets",
        "license": "CC-BY-NC-ND-4.0",
        "speed_range_mps": [float(speed.min()), float(speed.max())],
        "speed_mean_mps": float(speed.mean()),
    }
    with open(os.path.join(outdir, f"{robot}_{motion}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{robot}/{motion}: {X.shape[0]} samples x {X.shape[1]} channels @ {freq} Hz -> {csv_path}")


if __name__ == "__main__":
    outdir = "processed"
    for robot, motions in MOTIONS.items():
        for motion in motions:
            convert(robot, motion, outdir)
