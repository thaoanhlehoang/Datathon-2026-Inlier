from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "reports" / "visuals"


def configure_font():
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            return
    plt.rcParams["font.family"] = "DejaVu Sans"


def main():
    configure_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = [
        "Current\nProfit Decline",
        "Promotion\nOptimization",
        "Inventory\nReduction",
        "Conversion\nRecovery",
        "Product Mix\nImprovement",
        "Reduced Markdown\nPressure",
        "Potential\nRecovered Profit",
    ]

    # Illustrative index values: peak-year profitability gap = 0, current gap = -100.
    contributions = [-100, 24, 16, 22, 18, 14]
    final_value = sum(contributions)
    values = contributions + [final_value]

    colors = {
        "start": "#566B5F",
        "lever": "#8FB79C",
        "major": "#2F6B4F",
        "final": "#123D2A",
        "connector": "#B8C5BC",
        "grid": "#E7ECE8",
        "text": "#1E2A23",
        "muted": "#637267",
    }

    fig, ax = plt.subplots(figsize=(15.8, 8.6), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(labels))
    bar_w = 0.62
    cum = 0
    bottoms, heights, end_levels = [], [], []
    for i, val in enumerate(values):
        if i == 0:
            bottom = 0
            height = val
            cum = val
        elif i == len(values) - 1:
            bottom = 0
            height = final_value
        else:
            bottom = cum
            height = val
            cum += val
        bottoms.append(bottom)
        heights.append(height)
        end_levels.append(bottom + height)

    for i, (bottom, height) in enumerate(zip(bottoms, heights)):
        if i == 0:
            color = colors["start"]
        elif i == len(values) - 1:
            color = colors["final"]
        elif labels[i].startswith("Promotion") or labels[i].startswith("Conversion"):
            color = colors["major"]
        else:
            color = colors["lever"]
        ax.bar(
            x[i],
            height,
            bottom=bottom,
            width=bar_w,
            color=color,
            edgecolor="white",
            linewidth=1.4,
            zorder=3,
        )

    for i in range(len(labels) - 2):
        level = end_levels[i]
        ax.plot(
            [x[i] + bar_w / 2, x[i + 1] - bar_w / 2],
            [level, level],
            color=colors["connector"],
            lw=1.2,
            zorder=2,
        )

    ax.axhline(0, color="#B9C2BB", lw=1.1, zorder=1)

    for i, (bottom, height) in enumerate(zip(bottoms, heights)):
        if i == 0:
            ax.text(
                x[i],
                bottom + height / 2,
                "-100",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white",
            )
        elif i == len(values) - 1:
            ax.text(
                x[i],
                final_value - 4,
                f"{final_value:+.0f}",
                ha="center",
                va="top",
                fontsize=12,
                fontweight="bold",
                color="white",
            )
            ax.text(
                x[i],
                final_value + 5,
                "94 pts recovered",
                ha="center",
                va="bottom",
                fontsize=10.5,
                color=colors["final"],
                fontweight="bold",
            )
        else:
            ax.text(
                x[i],
                bottom + height + 3,
                f"+{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=11.5,
                fontweight="bold",
                color=colors["text"],
            )

    ax.text(
        1.05,
        12,
        "COMMERCIAL IMPROVEMENTS",
        fontsize=9.5,
        color=colors["muted"],
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax.plot([0.75, 4.35], [8, 8], color=colors["grid"], lw=1.1)
    ax.text(
        2.0,
        -119,
        "OPERATIONAL IMPROVEMENTS",
        fontsize=9.5,
        color=colors["muted"],
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax.plot([1.75, 5.35], [-114, -114], color=colors["grid"], lw=1.1)

    ax.annotate(
        "Promotion optimization\nreduces margin erosion",
        xy=(1, -76),
        xycoords="data",
        xytext=(0.55, -47),
        textcoords="data",
        arrowprops=dict(arrowstyle="-", color=colors["connector"], lw=1.0),
        fontsize=9.5,
        color=colors["muted"],
        ha="left",
        va="center",
    )
    ax.annotate(
        "Inventory reduction lowers\nmarkdown pressure",
        xy=(2, -60),
        xycoords="data",
        xytext=(2.55, -86),
        textcoords="data",
        arrowprops=dict(arrowstyle="-", color=colors["connector"], lw=1.0),
        fontsize=9.5,
        color=colors["muted"],
        ha="left",
        va="center",
    )
    ax.annotate(
        "Conversion recovery improves\nmonetization efficiency",
        xy=(3, -38),
        xycoords="data",
        xytext=(3.45, -23),
        textcoords="data",
        arrowprops=dict(arrowstyle="-", color=colors["connector"], lw=1.0),
        fontsize=9.5,
        color=colors["muted"],
        ha="left",
        va="center",
    )

    fig.text(
        0.055,
        0.93,
        "Con đường phục hồi lợi nhuận",
        fontsize=23,
        fontweight="bold",
        color=colors["text"],
        ha="left",
    )
    fig.text(
        0.055,
        0.885,
        "Profitability có thể cải thiện đáng kể thông qua các strategic operational và commercial actions.",
        fontsize=12.5,
        color=colors["muted"],
        ha="left",
    )

    fig.text(
        0.78,
        0.925,
        "Executive takeaway",
        fontsize=10.5,
        fontweight="bold",
        color=colors["final"],
        ha="left",
    )
    fig.text(
        0.78,
        0.875,
        "Recovery không đến từ một lever đơn lẻ;\nimpact đến từ phối hợp promotion,\ninventory, conversion và product mix.",
        fontsize=10.0,
        color=colors["muted"],
        ha="left",
        linespacing=1.35,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5, color=colors["text"])
    ax.set_ylabel("Profit impact index\n(Peak-year gap = 0)", fontsize=10.5, color=colors["muted"])
    ax.set_ylim(-125, 20)
    ax.set_yticks([-100, -75, -50, -25, 0])
    ax.tick_params(axis="y", labelsize=9.5, colors=colors["muted"], length=0)
    ax.tick_params(axis="x", length=0, pad=12)
    ax.yaxis.grid(True, color=colors["grid"], linewidth=0.8)
    ax.xaxis.grid(False)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#D8DED9")

    fig.text(
        0.055,
        0.045,
        "Illustrative prescriptive analytics view. Values shown as indexed profit impact to communicate strategic recovery levers, not accounting forecast outputs.",
        fontsize=8.8,
        color="#7A867D",
        ha="left",
    )

    plt.subplots_adjust(left=0.075, right=0.965, top=0.80, bottom=0.18)
    fig.savefig(OUT_DIR / "profit_recovery_waterfall.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "profit_recovery_waterfall.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
