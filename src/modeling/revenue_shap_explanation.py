import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This script requires SHAP. Install it with: python -m pip install shap"
    ) from exc

import our_method_forecast as om


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_CHART = PROJECT_ROOT / "outputs" / "model" / "explainability" / "revenue_shap_feature_importance.png"
OUT_TABLE = PROJECT_ROOT / "outputs" / "model" / "explainability" / "revenue_shap_feature_importance.csv"
MODEL_NAMES = ["Revenue", "order_count", "AOV"]
TARGET_MAP = {
    "Revenue": "Revenue",
    "order_count": "order_count",
    "AOV": "AOV",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the revenue models and explain feature importance with SHAP."
    )
    parser.add_argument("--sample-size", type=int, default=1500, help="Rows sampled for SHAP. Default: 1500.")
    parser.add_argument("--top-n", type=int, default=12, help="Top features to plot per model. Default: 12.")
    parser.add_argument("--chart", default=str(OUT_CHART), help="Output PNG path.")
    parser.add_argument("--table", default=str(OUT_TABLE), help="Output CSV path.")
    return parser.parse_args()


def make_training_frame(
    history: pd.DataFrame,
    reference_cache: dict[str, dict[str, object]],
    model_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    x_all = om.make_feature_matrix(
        history["Date"],
        history,
        history["Date"].min(),
        reference_cache,
        model_name,
    )
    features = om.FEATURE_ALLOWLIST[model_name]
    missing = sorted(set(features).difference(x_all.columns))
    if missing:
        raise ValueError(f"Missing selected features for {model_name}: {missing}")
    y = np.log1p(history[TARGET_MAP[model_name]].clip(lower=0))
    return x_all[features], y


def fit_models_and_shap(sample_size: int) -> pd.DataFrame:
    om.DATA_DIR = DATA_DIR
    history = om.load_history().sort_values("Date").reset_index(drop=True)
    history = om.add_derived_columns(history)
    reference_cache = om.make_reference_cache(history)

    rows = []
    rng = np.random.default_rng(om.RANDOM_STATE)
    for model_name in MODEL_NAMES:
        x, y = make_training_frame(history, reference_cache, model_name)
        model = om.make_model()
        model.fit(x, y)

        if len(x) > sample_size:
            row_idx = np.sort(rng.choice(len(x), size=sample_size, replace=False))
            x_explain = x.iloc[row_idx]
        else:
            x_explain = x

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_explain)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        importance = np.abs(shap_values).mean(axis=0)
        for feature, value in zip(x_explain.columns, importance):
            rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "mean_abs_shap": float(value),
                    "explained_rows": len(x_explain),
                }
            )

    return pd.DataFrame(rows).sort_values(["model", "mean_abs_shap"], ascending=[True, False])


def plot_importance(importance: pd.DataFrame, top_n: int, chart_path: Path) -> None:
    chart_rows = (
        importance.sort_values(["model", "mean_abs_shap"], ascending=[True, False])
        .groupby("model", group_keys=False)
        .head(top_n)
        .copy()
    )

    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(18, 7), sharex=False)
    colors = {
        "Revenue": "#2563eb",
        "order_count": "#059669",
        "AOV": "#dc2626",
    }

    for ax, model_name in zip(axes, MODEL_NAMES):
        part = chart_rows[chart_rows["model"] == model_name].sort_values("mean_abs_shap")
        ax.barh(part["feature"], part["mean_abs_shap"], color=colors[model_name], alpha=0.88)
        ax.set_title(model_name)
        ax.set_xlabel("mean |SHAP value|")
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="y", labelsize=12)

    fig.suptitle("Revenue Forecast Feature Importance (SHAP, LightGBM log-target models)", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    chart_path = Path(args.chart)
    table_path = Path(args.table)

    importance = fit_models_and_shap(sample_size=args.sample_size)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(table_path, index=False)
    plot_importance(importance, top_n=args.top_n, chart_path=chart_path)

    print(f"Saved SHAP table to {table_path}")
    print(f"Saved SHAP chart to {chart_path}")
    print(importance.groupby("model").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
