"""Schematic sagittal-plane stick-figure animation comparing how UnitreeH1,
Atlas, and UnitreeG1 realize the *same* underlying retargeted walk trial.

This is NOT a true-scale/true-geometry robot render -- we only have joint
angles (qpos) in the mocap npz, not link lengths or meshes. All three robots
are drawn with identical schematic segment lengths so the only thing that
differs between panels is the joint-angle trajectory itself, which is exactly
the point: showing how the same source motion gets redistributed across
hip/knee/ankle differently per robot (see Figure 4 panel D discussion).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ROBOTS = {
    "UnitreeH1": dict(hip="hip_flexion", knee="knee", ankle="ankle"),
    "Atlas": dict(hip="hip_flexion", knee="knee", ankle="ankle"),
    "UnitreeG1": dict(hip="hip_pitch", knee="knee", ankle="ankle_pitch"),
}

L_THIGH, L_SHANK, L_FOOT = 1.0, 1.0, 0.35
FS = 40.0
N_FRAMES = 240  # 6 s
START = 400  # skip a bit of transient at the very start of the recording


def load(robot):
    path = f"processed/{robot}_walk.csv"
    with open(path) as f:
        labels = f.readline().strip().split(",")
    labels = labels[:-1]
    X = np.loadtxt(path, delimiter=",", skiprows=1)[:, : len(labels)]
    return {l: X[:, i] for i, l in enumerate(labels)}


def leg_points(hip_angle, knee_angle, ankle_angle, side_sign):
    # hip_angle: +ve = thigh swings forward; knee_angle: +ve = shank flexes
    # backward relative to thigh; ankle_angle: +ve = foot dorsiflexes.
    hip = np.array([0.0, 0.0])
    thigh_dir = hip_angle
    knee = hip + L_THIGH * np.array([np.sin(thigh_dir), -np.cos(thigh_dir)])
    shank_dir = thigh_dir - knee_angle
    ankle = knee + L_SHANK * np.array([np.sin(shank_dir), -np.cos(shank_dir)])
    # foot points roughly forward (+x), tilted up/down by ankle dorsi/plantarflexion
    toe = ankle + L_FOOT * np.array([np.cos(ankle_angle), np.sin(ankle_angle)])
    return hip, knee, ankle, toe


data = {r: load(r) for r in ROBOTS}

fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
fig.suptitle("Same retargeted walk trial, three robots (schematic joint-angle stick figure -- not true scale)",
             fontsize=12, fontweight="bold")

lines = {}
labels_text = {}
COLORS = {"l": "#2e6da4", "r": "#b8292f"}

for ax, robot in zip(axes, ROBOTS):
    ax.set_title(robot)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-2.6, 0.4)
    ax.set_aspect("equal")
    ax.axhline(0.0, color="0.6", linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    lines[robot] = {}
    for side in ("l", "r"):
        (thigh_line,) = ax.plot([], [], "-", color=COLORS[side], linewidth=4, solid_capstyle="round")
        (shank_line,) = ax.plot([], [], "-", color=COLORS[side], linewidth=4, solid_capstyle="round")
        (foot_line,) = ax.plot([], [], "-", color=COLORS[side], linewidth=3, solid_capstyle="round")
        lines[robot][side] = (thigh_line, shank_line, foot_line)
    labels_text[robot] = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9)

time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=10)


def update(frame):
    i = START + frame
    artists = []
    for robot, cols in ROBOTS.items():
        d = data[robot]
        for side, sign in (("l", -1), ("r", 1)):
            hip_a = d[f"{cols['hip']}_{side}"][i]
            knee_a = d[f"{cols['knee']}_{side}"][i]
            ankle_a = d[f"{cols['ankle']}_{side}"][i]
            hip_a = hip_a + (0.0 if side == "r" else 0.0)
            hip, knee, ankle, toe = leg_points(hip_a, knee_a, ankle_a, sign)
            # offset left/right legs slightly in depth (x) purely for visibility
            off = -0.12 if side == "l" else 0.12
            thigh_line, shank_line, foot_line = lines[robot][side]
            thigh_line.set_data([hip[0] + off, knee[0] + off], [hip[1], knee[1]])
            shank_line.set_data([knee[0] + off, ankle[0] + off], [knee[1], ankle[1]])
            foot_line.set_data([ankle[0] + off, toe[0] + off], [ankle[1], toe[1]])
            artists.extend([thigh_line, shank_line, foot_line])
        hip_r_deg = np.degrees(d[cols["hip"] + "_r"][i])
        knee_r_deg = np.degrees(d[cols["knee"] + "_r"][i])
        ankle_r_deg = np.degrees(d[cols["ankle"] + "_r"][i])
        labels_text[robot].set_text(f"hip flex R: {hip_r_deg:5.1f} deg\n"
                                     f"knee R:     {knee_r_deg:5.1f} deg\n"
                                     f"ankle R:    {ankle_r_deg:5.1f} deg")
        artists.append(labels_text[robot])
    time_text.set_text(f"t = {frame / FS:4.2f} s   (blue = left leg, red = right leg)")
    artists.append(time_text)
    return artists


ani = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=1000 / FS, blit=False)
writer = animation.FFMpegWriter(fps=FS, bitrate=3000)
out_path = "../../figures/robot_kinematics_comparison.mp4"
ani.save(out_path, writer=writer, dpi=140)
print("Saved:", out_path)
