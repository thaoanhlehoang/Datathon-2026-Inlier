import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import our_method_forecast as om

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_FILE = PROJECT_ROOT / "outputs" / "model" / "submissions" / "submission_our_method_revenue_only.csv"
REVENUE_MODEL_NAMES = ["Revenue", "order_count", "AOV"]


@dataclass
class RevenueOnlySystem:
    history: pd.DataFrame
    reference_cache: dict[str, dict[str, object]]
    models: dict[str, om.ModelSpec]

    def forecast(self, forecast_dates: pd.Series, revenue_scale: float = 1.0) -> pd.DataFrame:
        forecast_dates = pd.Series(pd.to_datetime(forecast_dates), name="Date")
        state = self.history.set_index("Date").sort_index().copy()
        rows = []

        for date in forecast_dates:
            pred = {}
            for model_name in REVENUE_MODEL_NAMES:
                x = om.make_single_feature_row(
                    date,
                    state,
                    self.history["Date"].min(),
                    self.reference_cache,
                    model_name,
                )
                pred[model_name] = float(self.models[model_name].predict(x)[0])

            revenue = pred["Revenue"] * revenue_scale
            aov = revenue / pred["order_count"] if pred["order_count"] > 1e-9 else pred["AOV"]

            state.loc[date, ["Revenue", "order_count", "AOV"]] = [
                revenue,
                pred["order_count"],
                aov,
            ]
            rows.append({"Date": date, "Revenue": round(max(revenue, 0.0), 2)})

        return pd.DataFrame(rows)


def fit_revenue_only_system(history: pd.DataFrame) -> RevenueOnlySystem:
    history = om.add_derived_columns(history).sort_values("Date").reset_index(drop=True)
    reference_cache = om.make_reference_cache(history)

    # Revenue-only competition path: no COGS or cogs_ratio models are fit here.
    models = {
        "Revenue": om.fit_model(history, "Revenue", "Revenue", reference_cache),
        "order_count": om.fit_model(history, "order_count", "order_count", reference_cache),
        "AOV": om.fit_model(history, "AOV", "AOV", reference_cache),
    }
    return RevenueOnlySystem(history=history, reference_cache=reference_cache, models=models)


def parse_args():
    parser = argparse.ArgumentParser(description="Revenue-only production forecast.")
    parser.add_argument("--revenue-scale", type=float, default=1.0, help="Optional multiplicative Revenue scale.")
    parser.add_argument("--out-file", default=str(OUT_FILE), help="Output submission path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = om.load_history()
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])

    system = fit_revenue_only_system(history)
    revenue = system.forecast(sample["Date"], revenue_scale=args.revenue_scale)

    submission = sample.copy()
    submission["Revenue"] = revenue["Revenue"].to_numpy()
    if submission["Revenue"].isna().any() or (submission["Revenue"] < 0).any():
        raise ValueError("Invalid Revenue values in generated submission.")

    submission["Date"] = submission["Date"].dt.strftime("%Y-%m-%d")
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out_file, index=False)
    print(f"Saved {len(submission)} rows to {args.out_file}")


if __name__ == "__main__":
    main()
