"""Plot the flat-terrain velocity training run from metrics.csv.

metrics.csv holds every 10th iteration of the rsl_rl console log of the run
(mjlab 1.2.0, Mjlab-Velocity-Flat-Unitree-G1, 4096 envs, 30k iterations).

    python training/mjlab_velocity_flat/plot_training.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "docs" / "media" / "training_curves.png"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"


def load():
    with open(HERE / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    col = lambda k: [float(r[k]) for r in rows]  # noqa: E731
    return {k: col(k) for k in rows[0]}


def curriculum_steps(m):
    """Iterations where the max forward-speed command was raised."""
    steps, prev = [], m["cmd_lin_vel_x_max"][0]
    for it, v in zip(m["iteration"], m["cmd_lin_vel_x_max"]):
        if v != prev:
            steps.append((it, v))
            prev = v
    return steps


def main():
    m = load()
    it = m["iteration"]
    panels = [
        ("mean_reward", "Mean episode reward", None, 0.97),
        ("mean_episode_length", "Mean episode length (steps, max 1000)", (0, 1050), 0.10),
        ("error_vel_xy", "Linear velocity tracking error (m/s)", None, 0.97),
    ]

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), facecolor=SURFACE)
    for ax, (key, title, ylim, label_y) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.plot(it, m[key], color=SERIES, linewidth=1.6, solid_capstyle="round")
        for x, v in curriculum_steps(m):
            ax.axvline(x, color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)), alpha=0.7)
            ax.annotate(f"{v:g} m/s", (x, label_y), xycoords=("data", "axes fraction"),
                        xytext=(3, 0), textcoords="offset points", fontsize=8, color=INK_2,
                        va="top")
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
        ax.set_xlabel("Training iteration", color=INK_2)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=INK_2, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(0, max(it))
        if ylim:
            ax.set_ylim(*ylim)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:g}k" if x else "0")

    fig.suptitle("G1 flat-terrain velocity policy - PPO in mjlab, 4096 parallel envs",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.005, "Dashed lines: the command curriculum raises the max forward-speed command (to 2 m/s, then 3 m/s). "
             "Source: metrics.csv (every 10th iteration).", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
