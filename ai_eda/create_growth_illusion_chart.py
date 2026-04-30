from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datathon-2026-round-1"
OUT_DIR = ROOT / "report_visuals"
OUT_DIR.mkdir(exist_ok=True)

BASELINE_YEAR = 2013
YEARS = list(range(2013, 2023))


def indexed(series: pd.Series, baseline_year: int = BASELINE_YEAR) -> pd.Series:
    baseline = series.loc[baseline_year]
    return series / baseline * 100


def smooth_xy(years: pd.Series, values: pd.Series, points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    x = years.to_numpy(dtype=float)
    y = values.to_numpy(dtype=float)
    x_dense = np.linspace(x.min(), x.max(), points)
    return x_dense, PchipInterpolator(x, y)(x_dense)


def load_annual_metrics() -> pd.DataFrame:
    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"])
    orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["order_date"])
    traffic = pd.read_csv(DATA_DIR / "web_traffic.csv", parse_dates=["date"])

    annual_revenue = (
        sales.assign(Year=sales["Date"].dt.year)
        .groupby("Year", as_index=True)["Revenue"]
        .sum()
        .rename("Revenue")
    )
    annual_orders = (
        orders.assign(Year=orders["order_date"].dt.year)
        .groupby("Year", as_index=True)["order_id"]
        .nunique()
        .rename("Orders")
    )
    annual_sessions = (
        traffic.assign(Year=traffic["date"].dt.year)
        .groupby("Year", as_index=True)["sessions"]
        .sum()
        .rename("Sessions")
    )

    annual = pd.concat([annual_revenue, annual_orders, annual_sessions], axis=1).loc[YEARS]
    for metric in ["Revenue", "Orders", "Sessions"]:
        annual[f"{metric}_Index"] = indexed(annual[metric])
    annual["AOV"] = annual["Revenue"] / annual["Orders"]
    annual["AOV_Index"] = indexed(annual["AOV"])
    annual = annual.reset_index()
    return annual


def add_callout(ax, text: str, xy: tuple[float, float], xytext: tuple[float, float], color: str) -> None:
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=10.5,
        color="#202020",
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": color,
            "lw": 1.3,
            "shrinkA": 2,
            "shrinkB": 2,
            "connectionstyle": "angle3,angleA=0,angleB=90",
        },
        bbox={
            "boxstyle": "round,pad=0.35,rounding_size=0.12",
            "fc": "white",
            "ec": "#D9DDE3",
            "lw": 0.9,
            "alpha": 0.96,
        },
    )


def make_chart(annual: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#333333",
            "xtick.color": "#4A4A4A",
            "ytick.color": "#4A4A4A",
        }
    )

    colors = {
        "sessions": "#7EA6C8",
        "revenue": "#8B1E22",
        "orders": "#162D4D",
        "shade": "#EEF2F6",
        "grid": "#E7E9ED",
    }

    fig, ax = plt.subplots(figsize=(15.5, 8.8), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    years = annual["Year"]
    x_sess, y_sess = smooth_xy(years, annual["Sessions_Index"])
    x_rev, y_rev = smooth_xy(years, annual["Revenue_Index"])
    x_ord, y_ord = smooth_xy(years, annual["Orders_Index"])

    ax.axvspan(2018.75, 2022.35, color=colors["shade"], alpha=0.85, zorder=0)
    ax.text(
        2020.55,
        170,
        "Traffic Growth without Revenue Growth",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#4A5663",
    )

    ax.fill_between(x_sess, y_sess, 0, color=colors["sessions"], alpha=0.22, zorder=1)
    ax.plot(x_sess, y_sess, color=colors["sessions"], lw=2.0, alpha=0.7, zorder=2)
    ax.plot(x_rev, y_rev, color=colors["revenue"], lw=4.2, solid_capstyle="round", zorder=4)
    ax.plot(x_ord, y_ord, color=colors["orders"], lw=3.6, solid_capstyle="round", zorder=5)

    ax.scatter(years, annual["Revenue_Index"], s=34, color=colors["revenue"], edgecolor="white", linewidth=1, zorder=6)
    ax.scatter(years, annual["Orders_Index"], s=30, color=colors["orders"], edgecolor="white", linewidth=1, zorder=7)

    peak_rev = annual.loc[annual["Revenue_Index"].idxmax()]
    last = annual.loc[annual["Year"].eq(2022)].iloc[0]
    y2019 = annual.loc[annual["Year"].eq(2019)].iloc[0]

    add_callout(
        ax,
        f"Revenue peaks in {int(peak_rev['Year'])}\nIndex {peak_rev['Revenue_Index']:.0f}",
        (peak_rev["Year"], peak_rev["Revenue_Index"]),
        (2013.35, 147),
        colors["revenue"],
    )
    add_callout(
        ax,
        f"Sessions keep rising\n+{last['Sessions_Index'] - 100:.0f}% vs baseline",
        (2022, last["Sessions_Index"]),
        (2020.1, 194),
        colors["sessions"],
    )
    add_callout(
        ax,
        f"By 2022, traffic index is\n{last['Sessions_Index'] / last['Revenue_Index']:.1f}x revenue and\n{last['Sessions_Index'] / last['Orders_Index']:.1f}x orders",
        (2022, last["Orders_Index"]),
        (2018.7, 74),
        colors["orders"],
    )
    add_callout(
        ax,
        f"AOV index rises to {last['AOV_Index']:.0f}\nwhile order volume falls",
        (2022, last["Revenue_Index"]),
        (2015.9, 43),
        colors["revenue"],
    )
    add_callout(
        ax,
        f"2019-2022: sessions +{(last['Sessions_Index'] / y2019['Sessions_Index'] - 1) * 100:.0f}%,\norders still down {100 - last['Orders_Index']:.0f}%",
        (2020.5, 155),
        (2018.85, 130),
        colors["sessions"],
    )

    ax.text(2022.18, last["Sessions_Index"], "Sessions", color=colors["sessions"], fontsize=11.5, fontweight="bold", va="center")
    ax.text(2022.18, last["Revenue_Index"], "Revenue", color=colors["revenue"], fontsize=11.5, fontweight="bold", va="center")
    ax.text(2022.18, last["Orders_Index"], "Order Count", color=colors["orders"], fontsize=11.5, fontweight="bold", va="center")

    ax.axhline(100, color="#BFC5CD", lw=1.1, linestyle=(0, (4, 4)), zorder=0)
    ax.text(2013.02, 102.5, "Baseline = 100", color="#6C737C", fontsize=9.5, va="bottom")

    ax.set_xlim(2012.85, 2022.85)
    ax.set_ylim(0, 210)
    ax.set_xticks(range(2012, 2023))
    ax.set_yticks(range(0, 211, 25))
    ax.set_ylabel("Index (2013 = 100)", fontsize=11.5, labelpad=12)
    ax.set_xlabel("")

    ax.grid(axis="y", color=colors["grid"], linewidth=0.9)
    ax.grid(axis="x", visible=False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#C8CDD4")
    ax.tick_params(axis="both", length=0, labelsize=10.5)

    fig.suptitle(
        "The Growth Illusion: Traffic Growth No Longer Converts into Revenue",
        x=0.07,
        y=0.965,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color="#111111",
    )
    fig.text(
        0.07,
        0.918,
        "After 2017, acquisition metrics continued rising while monetization performance deteriorated.",
        ha="left",
        fontsize=13,
        color="#555B62",
    )
    fig.text(
        0.07,
        0.055,
        "Source: sales.csv, orders.csv, web_traffic.csv. Common index baseline uses 2013 because web traffic begins in 2013; sales/orders begin mid-2012.",
        ha="left",
        fontsize=8.8,
        color="#747B84",
    )

    fig.subplots_adjust(left=0.07, right=0.90, top=0.86, bottom=0.12)

    png_path = OUT_DIR / "growth_illusion_traffic_revenue_orders.png"
    svg_path = OUT_DIR / "growth_illusion_traffic_revenue_orders.svg"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    annual = load_annual_metrics()
    annual.to_csv(OUT_DIR / "growth_illusion_indexed_metrics.csv", index=False)
    make_chart(annual)


if __name__ == "__main__":
    main()
