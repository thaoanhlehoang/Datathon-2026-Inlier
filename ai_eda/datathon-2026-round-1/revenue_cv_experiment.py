import argparse
import itertools
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import our_method_forecast as om
except ModuleNotFoundError:
    om = None

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent
CACHE_DIR = DATA_DIR / "diagnostic_cache_revenue_cv"
REPORT_PATH = DATA_DIR / "revenue_cv_experiment_report.md"

FOLD_SPECS = {
    "C": (pd.Timestamp("2020-12-31"), pd.Timestamp("2021-01-01"), pd.Timestamp("2022-07-01")),
    "D": (pd.Timestamp("2021-12-31"), pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
}

REVENUE_MODEL_NAMES = ["Revenue", "order_count", "AOV"]
TARGET_MAP = {
    "Revenue": "Revenue",
    "order_count": "order_count",
    "AOV": "AOV",
}

ENSEMBLE_WEIGHTS = [round(x, 1) for x in np.linspace(0, 1, 11)]
REVENUE_SCALES = [0.95, 1.00, 1.05, 1.10, 1.12, 1.15, 1.20]

PARAM_GRID = [
    {
        "name": "current",
        "params": {
            "n_estimators": 650,
            "learning_rate": 0.025,
            "num_leaves": 31,
            "min_child_samples": 25,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.05,
            "reg_lambda": 0.4,
        },
    },
    {
        "name": "slower_900",
        "params": {
            "n_estimators": 900,
            "learning_rate": 0.02,
            "num_leaves": 31,
            "min_child_samples": 25,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.05,
            "reg_lambda": 0.4,
        },
    },
    {
        "name": "shallower_leaves",
        "params": {
            "n_estimators": 650,
            "learning_rate": 0.025,
            "num_leaves": 15,
            "min_child_samples": 35,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.10,
            "reg_lambda": 0.6,
        },
    },
    {
        "name": "larger_leaves",
        "params": {
            "n_estimators": 650,
            "learning_rate": 0.025,
            "num_leaves": 63,
            "min_child_samples": 25,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.05,
            "reg_lambda": 0.4,
        },
    },
    {
        "name": "more_regularized",
        "params": {
            "n_estimators": 700,
            "learning_rate": 0.025,
            "num_leaves": 31,
            "min_child_samples": 45,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.20,
            "reg_lambda": 1.0,
        },
    },
]


def require_our_method_forecast():
    global om
    if om is not None:
        return om
    try:
        import our_method_forecast as imported_om
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "revenue_cv_experiment.py requires our_method_forecast.py in the same directory. "
            "The CV script was imported successfully, but training cannot run until that file exists."
        ) from exc
    om = imported_om
    return om


@dataclass
class RevenueCVModelSpec:
    name: str
    model: LGBMRegressor
    feature_names: list[str]
    log_target: bool = True

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        pred = self.model.predict(x.reindex(columns=self.feature_names))
        if self.log_target:
            pred = np.expm1(pred)
        return np.maximum(pred, 0.0)


@dataclass
class RevenueCVSystem:
    history: pd.DataFrame
    reference_cache: dict[str, dict[str, object]]
    models: dict[str, RevenueCVModelSpec]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Revenue-only rolling-origin hyperparameter CV. Does not train COGS/cogs_ratio."
    )
    parser.add_argument("--folds", default="C,D", help="Comma-separated fold names. Default: C,D.")
    parser.add_argument("--refresh-cache", action="store_true", help="Recompute cached fold predictions.")
    parser.add_argument("--param-grid-json", help="Optional JSON file with [{'name': str, 'params': dict}, ...].")
    parser.add_argument("--skip-final-submission", action="store_true", help="Skip full-history submission generation.")
    parser.add_argument("--out-file", default="submission_revenue_cv_best.csv", help="Final candidate filename.")
    return parser.parse_args()


def selected_folds(raw: str) -> list[str]:
    folds = [x.strip().upper() for x in raw.split(",") if x.strip()]
    unknown = sorted(set(folds).difference(FOLD_SPECS))
    if unknown:
        raise ValueError(f"Unknown folds: {unknown}. Supported folds: {sorted(FOLD_SPECS)}")
    return folds


def load_param_grid(path: str | None) -> list[dict[str, object]]:
    if path is None:
        return PARAM_GRID
    with open(path, "r", encoding="utf-8") as f:
        grid = json.load(f)
    if not isinstance(grid, list) or not grid:
        raise ValueError("Parameter grid JSON must be a non-empty list.")
    for item in grid:
        if not isinstance(item, dict) or "name" not in item or "params" not in item:
            raise ValueError("Each grid item must contain 'name' and 'params'.")
        if not isinstance(item["params"], dict):
            raise ValueError("Grid item 'params' must be a dict.")
    return grid


def make_model(params: dict[str, object]) -> LGBMRegressor:
    forecast_module = require_our_method_forecast()
    base = {
        "objective": "regression",
        "random_state": forecast_module.RANDOM_STATE,
        "n_estimators": 650,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "min_child_samples": 25,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 0.4,
        "verbosity": -1,
    }
    base.update(params)
    return LGBMRegressor(**base)


def fit_revenue_cv_model(
    history: pd.DataFrame,
    model_name: str,
    target_col: str,
    reference_cache: dict[str, dict[str, object]],
    params: dict[str, object],
) -> RevenueCVModelSpec:
    forecast_module = require_our_method_forecast()
    x = forecast_module.make_feature_matrix(
        history["Date"],
        history,
        history["Date"].min(),
        reference_cache,
        model_name,
    )
    selected_features = forecast_module.FEATURE_ALLOWLIST[model_name]
    missing = sorted(set(selected_features).difference(x.columns))
    if missing:
        raise ValueError(f"Missing selected features for {model_name}: {missing}")
    y = np.log1p(history[target_col].clip(lower=0))
    model = make_model(params)
    model.fit(x[selected_features], y)
    return RevenueCVModelSpec(model_name, model, list(selected_features), log_target=True)


def fit_revenue_cv_system(history: pd.DataFrame, params: dict[str, object]) -> RevenueCVSystem:
    forecast_module = require_our_method_forecast()
    history = forecast_module.add_derived_columns(history).sort_values("Date").reset_index(drop=True)
    reference_cache = forecast_module.make_reference_cache(history)
    models = {
        model_name: fit_revenue_cv_model(
            history,
            model_name,
            TARGET_MAP[model_name],
            reference_cache,
            params,
        )
        for model_name in REVENUE_MODEL_NAMES
    }
    return RevenueCVSystem(history=history, reference_cache=reference_cache, models=models)


def forecast_revenue_details(system: RevenueCVSystem, dates: pd.Series, revenue_direct_weight: float) -> pd.DataFrame:
    forecast_module = require_our_method_forecast()
    state = system.history.set_index("Date").sort_index().copy()
    rows = []
    for horizon_day, date in enumerate(pd.Series(pd.to_datetime(dates)), start=1):
        pred = {}
        for model_name in REVENUE_MODEL_NAMES:
            x = forecast_module.make_single_feature_row(
                date,
                state,
                system.history["Date"].min(),
                system.reference_cache,
                model_name,
            )
            pred[model_name] = float(system.models[model_name].predict(x)[0])

        component = pred["order_count"] * pred["AOV"]
        revenue = revenue_direct_weight * pred["Revenue"] + (1 - revenue_direct_weight) * component
        aov = revenue / pred["order_count"] if pred["order_count"] > 1e-9 else pred["AOV"]
        state.loc[date, ["Revenue", "order_count", "AOV"]] = [revenue, pred["order_count"], aov]

        rows.append(
            {
                "Date": date,
                "horizon_day": horizon_day,
                "Revenue": round(max(revenue, 0.0), 2),
                "pred_order_count": pred["order_count"],
                "pred_AOV": pred["AOV"],
            }
        )
    return pd.DataFrame(rows)


def metric_dict(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": mean_squared_error(actual, pred) ** 0.5,
        "R2": r2_score(actual, pred),
        "bias": float(np.mean(pred - actual)),
        "pred_actual_mean_ratio": float(np.mean(pred) / np.mean(actual)) if np.mean(actual) else np.nan,
    }


def prediction_cache_path(config_name: str, fold: str) -> Path:
    safe_name = config_name.replace(" ", "_").replace("/", "_")
    return CACHE_DIR / f"{safe_name}_fold_{fold}_revenue_predictions.csv"


def validate_prediction_cache(path: Path, fold: str) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
    except Exception:
        return False
    required = {
        "config_name",
        "fold",
        "Date",
        "horizon_day",
        "actual_Revenue",
        "pred_Revenue_direct",
        "pred_Revenue_component",
    }
    if not required.issubset(df.columns):
        return False
    _, val_start, val_end = FOLD_SPECS[fold]
    expected = pd.date_range(val_start, val_end, freq="D")
    return len(df) == len(expected) and pd.Index(df["Date"]).equals(pd.Index(expected))


def build_or_load_fold_predictions(
    history: pd.DataFrame,
    config_name: str,
    params: dict[str, object],
    fold: str,
    refresh_cache: bool,
) -> tuple[pd.DataFrame, str]:
    path = prediction_cache_path(config_name, fold)
    if validate_prediction_cache(path, fold) and not refresh_cache:
        return pd.read_csv(path, parse_dates=["Date"]), "loaded"

    train_end, val_start, val_end = FOLD_SPECS[fold]
    train = history[history["Date"] <= train_end].copy()
    actual = history[(history["Date"] >= val_start) & (history["Date"] <= val_end)].copy()
    system = fit_revenue_cv_system(train, params)
    direct = forecast_revenue_details(system, actual["Date"], revenue_direct_weight=1.0)
    component = forecast_revenue_details(system, actual["Date"], revenue_direct_weight=0.0)

    out = pd.DataFrame(
        {
            "config_name": config_name,
            "fold": fold,
            "Date": actual["Date"].to_numpy(),
            "horizon_day": np.arange(1, len(actual) + 1),
            "actual_Revenue": actual["Revenue"].to_numpy(),
            "pred_Revenue_direct": direct["Revenue"].to_numpy(),
            "pred_Revenue_component": component["Revenue"].to_numpy(),
            "train_end": train_end.date().isoformat(),
            "validation_start": val_start.date().isoformat(),
            "validation_end": val_end.date().isoformat(),
        }
    )
    CACHE_DIR.mkdir(exist_ok=True)
    out.to_csv(path, index=False)
    return out, "recomputed"


def evaluate_config_predictions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for w, scale in itertools.product(ENSEMBLE_WEIGHTS, REVENUE_SCALES):
        pred = (w * df["pred_Revenue_direct"] + (1 - w) * df["pred_Revenue_component"]) * scale
        temp = df[["config_name", "fold", "actual_Revenue"]].copy()
        temp["pred"] = np.asarray(pred, dtype=float)
        for (config_name, fold), part in temp.groupby(["config_name", "fold"], sort=False):
            rows.append(
                {
                    "config_name": config_name,
                    "fold": fold,
                    "w": w,
                    "scale": scale,
                    **metric_dict(part["actual_Revenue"], part["pred"]),
                }
            )
    details = pd.DataFrame(rows)
    summary = (
        details.groupby(["config_name", "w", "scale"], as_index=False)
        .agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            R2=("R2", "mean"),
            bias=("bias", "mean"),
            pred_actual_mean_ratio=("pred_actual_mean_ratio", "mean"),
        )
        .sort_values(["MAE", "RMSE"])
        .reset_index(drop=True)
    )
    return details, summary


def horizon_bucket(day: int) -> str:
    if day <= 30:
        return "days_1_30"
    if day <= 90:
        return "days_31_90"
    if day <= 180:
        return "days_91_180"
    if day <= 365:
        return "days_181_365"
    return "days_366_plus"


def horizon_bucket_mae(df: pd.DataFrame, best: dict[str, object]) -> pd.DataFrame:
    pred = (
        float(best["w"]) * df["pred_Revenue_direct"]
        + (1 - float(best["w"])) * df["pred_Revenue_component"]
    ) * float(best["scale"])
    temp = df.copy()
    temp["pred"] = np.asarray(pred, dtype=float)
    temp["horizon_bucket"] = temp["horizon_day"].map(horizon_bucket)
    rows = []
    for (fold, bucket), part in temp.groupby(["fold", "horizon_bucket"], sort=False):
        rows.append(
            {
                "config_name": best["config_name"],
                "fold": fold,
                "horizon_bucket": bucket,
                "Revenue_MAE": mean_absolute_error(part["actual_Revenue"], part["pred"]),
                "rows": len(part),
            }
        )
    return pd.DataFrame(rows)


def write_submission(sample: pd.DataFrame, revenue: np.ndarray, filename: str) -> pd.DataFrame:
    out = sample.copy()
    out["Revenue"] = np.maximum(np.asarray(revenue, dtype=float), 0.0).round(2)
    if out["Revenue"].isna().any() or (out["Revenue"] < 0).any():
        raise ValueError(f"{filename} contains invalid Revenue values.")
    write = out.copy()
    write["Date"] = write["Date"].dt.strftime("%Y-%m-%d")
    path = DATA_DIR / filename
    write.to_csv(path, index=False)
    return pd.DataFrame(
        [
            {
                "file": filename,
                "rows": len(write),
                "columns": ", ".join(write.columns),
                "date_min": write["Date"].min(),
                "date_max": write["Date"].max(),
                "Revenue_min": out["Revenue"].min(),
                "Revenue_mean": out["Revenue"].mean(),
                "Revenue_max": out["Revenue"].max(),
                "Revenue_nan_count": int(out["Revenue"].isna().sum()),
                "Revenue_negative_count": int((out["Revenue"] < 0).sum()),
            }
        ]
    )


def generate_final_submission(
    history: pd.DataFrame,
    sample: pd.DataFrame,
    params: dict[str, object],
    best: dict[str, object],
    out_file: str,
) -> pd.DataFrame:
    system = fit_revenue_cv_system(history, params)
    direct = forecast_revenue_details(system, sample["Date"], revenue_direct_weight=1.0)
    component = forecast_revenue_details(system, sample["Date"], revenue_direct_weight=0.0)
    revenue = (
        float(best["w"]) * direct["Revenue"].to_numpy()
        + (1 - float(best["w"])) * component["Revenue"].to_numpy()
    ) * float(best["scale"])
    return write_submission(sample, revenue, out_file)


def fmt_table(df: pd.DataFrame, max_rows: int = 25) -> str:
    if df is None or df.empty:
        return "Not available"
    shown = df.head(max_rows).copy()
    for col in shown.select_dtypes(include=[np.number]).columns:
        shown[col] = shown[col].map(lambda x: f"{x:,.4f}" if pd.notna(x) else "")
    return shown.to_markdown(index=False)


def build_report(context: dict[str, object]) -> str:
    best = context["best"]
    lines = [
        "# Revenue CV Experiment Report",
        "",
        "## A. Scope",
        "Revenue-only rolling-origin hyperparameter CV. The script trains only Revenue, order_count, and AOV models. It does not train or optimize COGS/cogs_ratio.",
        "",
        "## B. Folds",
        fmt_table(context["fold_table"]),
        "",
        "## C. Parameter Grid",
        fmt_table(context["param_table"], max_rows=50),
        "",
        "## D. Best Config",
        fmt_table(pd.DataFrame([best])),
        "",
        "## E. Strategy Summary",
        fmt_table(context["summary"], max_rows=30),
        "",
        "## F. Fold-Level Metrics For Best Config",
        fmt_table(context["best_details"], max_rows=20),
        "",
        "## G. Horizon Bucket MAE",
        fmt_table(context["horizon_mae"], max_rows=20),
        "",
        "## H. Generated Submission",
        fmt_table(context["submission_summary"]),
        "",
        "## I. Runtime",
        f"- Started: `{context['start_time']}`",
        f"- Finished: `{context['end_time']}`",
        f"- Runtime seconds: `{context['runtime_seconds']:.1f}`",
        f"- Cache directory: `{CACHE_DIR.name}`",
        "",
        "Cache status:",
        fmt_table(context["cache_status"], max_rows=50),
    ]
    return "\n".join(lines)


def main() -> None:
    start = time.time()
    start_label = pd.Timestamp.now().isoformat(timespec="seconds")
    args = parse_args()
    folds = selected_folds(args.folds)
    param_grid = load_param_grid(args.param_grid_json)
    CACHE_DIR.mkdir(exist_ok=True)

    forecast_module = require_our_method_forecast()
    history = forecast_module.load_history()
    all_predictions = []
    cache_rows = []
    for item in param_grid:
        config_name = str(item["name"])
        params = dict(item["params"])
        for fold in folds:
            print(f"Config {config_name}, Fold {fold}", flush=True)
            pred, status = build_or_load_fold_predictions(
                history=history,
                config_name=config_name,
                params=params,
                fold=fold,
                refresh_cache=args.refresh_cache,
            )
            all_predictions.append(pred)
            cache_rows.append({"config_name": config_name, "fold": fold, "cache_status": status})

    predictions = pd.concat(all_predictions, ignore_index=True)
    details, summary = evaluate_config_predictions(predictions)
    best = summary.iloc[0].to_dict()
    best_predictions = predictions[predictions["config_name"] == best["config_name"]].copy()
    best_details = details[
        (details["config_name"] == best["config_name"])
        & (details["w"] == best["w"])
        & (details["scale"] == best["scale"])
    ].sort_values("fold")
    horizon_mae = horizon_bucket_mae(best_predictions, best)

    selected_params = next(item["params"] for item in param_grid if item["name"] == best["config_name"])
    if args.skip_final_submission:
        submission_summary = pd.DataFrame([{"file": "skipped", "reason": "--skip-final-submission"}])
    else:
        sample = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])
        submission_summary = generate_final_submission(
            history=history,
            sample=sample,
            params=dict(selected_params),
            best=best,
            out_file=args.out_file,
        )

    fold_table = pd.DataFrame(
        [
            {
                "fold": fold,
                "train_end": FOLD_SPECS[fold][0].date().isoformat(),
                "validation_start": FOLD_SPECS[fold][1].date().isoformat(),
                "validation_end": FOLD_SPECS[fold][2].date().isoformat(),
            }
            for fold in folds
        ]
    )
    param_table = pd.DataFrame(
        [{"config_name": item["name"], **item["params"]} for item in param_grid]
    )

    context = {
        "start_time": start_label,
        "end_time": pd.Timestamp.now().isoformat(timespec="seconds"),
        "runtime_seconds": time.time() - start,
        "fold_table": fold_table,
        "param_table": param_table,
        "summary": summary,
        "best": best,
        "best_details": best_details,
        "horizon_mae": horizon_mae,
        "submission_summary": submission_summary,
        "cache_status": pd.DataFrame(cache_rows),
    }

    predictions.to_csv(CACHE_DIR / "fold_predictions.csv", index=False)
    details.to_csv(CACHE_DIR / "strategy_fold_metrics.csv", index=False)
    summary.to_csv(CACHE_DIR / "strategy_summary.csv", index=False)
    horizon_mae.to_csv(CACHE_DIR / "best_horizon_bucket_mae.csv", index=False)
    submission_summary.to_csv(CACHE_DIR / "submission_summary.csv", index=False)
    REPORT_PATH.write_text(build_report(context), encoding="utf-8")
    print(f"Saved report to {REPORT_PATH.name}", flush=True)
    print(fmt_table(pd.DataFrame([best])), flush=True)


if __name__ == "__main__":
    main()
