from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
VISUALS_DIR = PROJECT_ROOT / "reports" / "visuals"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"


def build_monthly_dataset() -> pd.DataFrame:
    inventory = pd.read_csv(DATA_DIR / "inventory.csv", parse_dates=["snapshot_date"])
    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"])

    inventory["period"] = inventory["snapshot_date"].dt.to_period("M").dt.to_timestamp()
    sales["period"] = sales["Date"].dt.to_period("M").dt.to_timestamp()

    monthly_inventory = (
        inventory.groupby("period", as_index=False)
        .agg(
            days_of_supply=("days_of_supply", "mean"),
            inventory_units=("stock_on_hand", "sum"),
            overstock_rate=("overstock_flag", "mean"),
        )
        .sort_values("period")
    )
    monthly_revenue = (
        sales.groupby("period", as_index=False)
        .agg(revenue=("Revenue", "sum"))
        .sort_values("period")
    )

    dataset = monthly_inventory.merge(monthly_revenue, on="period", how="inner")
    dataset["future_revenue"] = dataset["revenue"].shift(-1)
    dataset = dataset.dropna(subset=["days_of_supply", "future_revenue"]).copy()
    dataset["future_revenue_m"] = dataset["future_revenue"] / 1_000_000
    dataset["period_label"] = dataset["period"].dt.strftime("%b %Y")
    return dataset


def linear_fit_with_ci(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x_grid + intercept

    n = len(x)
    fitted = slope * x + intercept
    residual_std = np.sqrt(np.sum((y - fitted) ** 2) / (n - 2))
    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean) ** 2)
    # 1.98 is close to the 95% t critical value for this monthly sample size.
    ci = 1.98 * residual_std * np.sqrt(1 / n + (x_grid - x_mean) ** 2 / sxx)
    return y_hat, y_hat - ci, y_hat + ci


def annotate_with_arrow(ax, text, xy, xytext, color="#2F5D46"):
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=10,
        color="#244134",
        ha="left",
        va="center",
        arrowprops=dict(
            arrowstyle="-",
            color=color,
            lw=1.1,
            shrinkA=4,
            shrinkB=4,
            connectionstyle="arc3,rad=0.08",
        ),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D7E2D8", lw=0.8, alpha=0.96),
    )


def create_chart() -> None:
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df = build_monthly_dataset()

    x = df["days_of_supply"].to_numpy()
    y = df["future_revenue_m"].to_numpy()
    corr = np.corrcoef(x, y)[0, 1]
    risk_threshold = 1000
    x_grid = np.linspace(max(0, x.min() * 0.85), x.max() * 1.04, 300)
    y_hat, lower_ci, upper_ci = linear_fit_with_ci(x, y, x_grid)

    metrics = pd.DataFrame(
        [
            {
                "periods": len(df),
                "period_start": df["period"].min().date().isoformat(),
                "period_end": df["period"].max().date().isoformat(),
                "correlation_days_supply_future_revenue": corr,
                "risk_threshold_days_supply": risk_threshold,
                "risk_period_count": int((df["days_of_supply"] >= risk_threshold).sum()),
                "avg_future_revenue_all_m": y.mean(),
                "avg_future_revenue_risk_zone_m": df.loc[
                    df["days_of_supply"] >= risk_threshold, "future_revenue_m"
                ].mean(),
            }
        ]
    )
    df.to_csv(METRICS_DIR / "inventory_revenue_leading_indicator_monthly_metrics.csv", index=False)
    metrics.to_csv(METRICS_DIR / "inventory_revenue_leading_indicator_summary.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 21,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(13.6, 8.2), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.axvspan(
        risk_threshold,
        x_grid.max(),
        color="#EAF2EA",
        alpha=0.86,
        zorder=0,
    )
    ax.text(
        risk_threshold + 35,
        y.max() * 0.985,
        "Vùng rủi ro tồn kho dư thừa",
        fontsize=10.5,
        fontweight="bold",
        color="#2F5D46",
        va="top",
    )
    ax.text(
        risk_threshold + 35,
        y.max() * 0.92,
        "High inventory\nWeak future revenue\nPotential cash flow pressure",
        fontsize=9,
        color="#62776B",
        linespacing=1.35,
        va="top",
    )

    size = np.interp(df["inventory_units"], (df["inventory_units"].min(), df["inventory_units"].max()), (55, 170))
    ax.scatter(
        x,
        y,
        s=size,
        color="#8DAF9A",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.66,
        zorder=3,
    )
    ax.fill_between(x_grid, lower_ci, upper_ci, color="#BFD7C4", alpha=0.25, linewidth=0, zorder=1)
    ax.plot(x_grid, y_hat, color="#1F5A3F", linewidth=3.0, zorder=4)

    latest = df.iloc[-1]
    high_risk = df[df["days_of_supply"] >= risk_threshold].sort_values("future_revenue_m").head(1)
    if not high_risk.empty:
        risk_point = high_risk.iloc[0]
        annotate_with_arrow(
            ax,
            "Tồn kho cao thường đi kèm\ndoanh thu tương lai yếu hơn",
            (risk_point["days_of_supply"], risk_point["future_revenue_m"]),
            (risk_threshold + 145, risk_point["future_revenue_m"] + 106),
        )
    annotate_with_arrow(
        ax,
        "Days of Supply tăng mạnh\nở giai đoạn cuối kỳ",
        (latest["days_of_supply"], latest["future_revenue_m"]),
        (latest["days_of_supply"] - 365, latest["future_revenue_m"] + 176),
    )

    ax.text(
        0.028,
        0.92,
        f"Corr ≈ {corr:.2f}",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="#1F5A3F",
        bbox=dict(boxstyle="round,pad=0.42", fc="#F4F8F4", ec="#DCE8DD", lw=0.9),
    )
    ax.text(
        0.028,
        0.865,
        "Mối quan hệ âm đáng kể giữa tồn kho và doanh thu tương lai",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#586B60",
    )

    ax.set_title(
        "Tồn kho dư thừa đang dự báo suy giảm doanh thu tương lai",
        loc="left",
        pad=24,
        fontweight="bold",
        color="#17251D",
    )
    ax.text(
        0,
        1.025,
        "Days of Supply cao có xu hướng đi kèm hiệu suất doanh thu yếu hơn ở kỳ tiếp theo.",
        transform=ax.transAxes,
        fontsize=11.5,
        color="#5F6F66",
    )

    ax.set_xlabel("Days of Supply", labelpad=12, color="#34453B")
    ax.set_ylabel("Doanh thu kỳ tiếp theo (triệu $)", labelpad=12, color="#34453B")
    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.set_ylim(max(0, y.min() * 0.72), y.max() * 1.12)
    ax.grid(True, which="major", color="#E8EEE9", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D7DED8")
    ax.spines["bottom"].set_color("#D7DED8")
    ax.tick_params(colors="#58655E")

    ax.text(
        0,
        -0.16,
        "Nguồn: Inventory snapshots và monthly sales. Mỗi điểm là một tháng; doanh thu được lead sang kỳ tiếp theo.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#78877E",
    )

    plt.tight_layout(rect=[0.035, 0.05, 0.99, 0.95])
    fig.savefig(VISUALS_DIR / "inventory_revenue_leading_indicator_scatter.png", bbox_inches="tight", facecolor="white")
    fig.savefig(VISUALS_DIR / "inventory_revenue_leading_indicator_scatter.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    create_chart()
