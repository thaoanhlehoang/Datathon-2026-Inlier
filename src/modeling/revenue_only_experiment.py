import argparse
import itertools
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import our_method_forecast as om

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
FULL_CACHE_DIR = PROJECT_ROOT / "artifacts" / "cache" / "diagnostic_cache_full"
CACHE_DIR = PROJECT_ROOT / "artifacts" / "cache" / "diagnostic_cache_revenue_only"
REPORT_PATH = OUTPUT_DIR / "revenue_only_experiment_report.md"

TEST_START = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2024-07-01")
FOLD_SPECS = {
    "C": (pd.Timestamp("2020-12-31"), pd.Timestamp("2021-01-01"), pd.Timestamp("2022-07-01")),
    "D": (pd.Timestamp("2021-12-31"), pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
}

REVENUE_MODEL_NAMES = ["Revenue", "order_count", "AOV"]
REVENUE_SCALES = [0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 1.00, 1.025, 1.05]
YEAR1_SCALES = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05]
YEAR2_SCALES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
ENSEMBLE_WEIGHTS = [round(x, 1) for x in np.linspace(0, 1, 11)]
RECENT_WINDOWS = [14, 30, 60, 90]

SUBMISSION_FILES = {
    "current": "submission_revenue_only_current.csv",
    "scaled_best": "submission_revenue_only_scaled_best.csv",
    "horizon_scaled_best": "submission_revenue_only_horizon_scaled_best.csv",
    "recent_level_best": "submission_revenue_only_recent_level_best.csv",
    "ensemble_scaled_best": "submission_revenue_only_ensemble_scaled_best.csv",
}


@dataclass
class RevenueSystem:
    history: pd.DataFrame
    reference_cache: dict
    models: dict


def parse_args():
    parser = argparse.ArgumentParser(description="Revenue-only calibration and submission experiment.")
    parser.add_argument("--folds", default="C,D", help="Comma-separated folds. Default: C,D.")
    parser.add_argument("--refresh-cache", action="store_true", help="Recompute full-history future Revenue predictions.")
    parser.add_argument("--skip-submissions", action="store_true", help="Only write metrics/report, not candidate submissions.")
    return parser.parse_args()


def selected_folds(raw):
    folds = [x.strip().upper() for x in raw.split(",") if x.strip()]
    unknown = sorted(set(folds).difference(FOLD_SPECS))
    if unknown:
        raise ValueError(f"Unknown folds: {unknown}. Supported folds: {sorted(FOLD_SPECS)}")
    return folds


def inspect_submission_format(sample):
    columns = list(sample.columns)
    required = "sample includes COGS; output must preserve Date, Revenue, COGS column layout"
    if columns == ["Date", "Revenue"]:
        required = "sample has Date and Revenue only; output will be Revenue-only"
    elif "Revenue" not in columns or "Date" not in columns:
        raise ValueError(f"sample_submission.csv must contain Date and Revenue. Found: {columns}")
    return {
        "columns": ", ".join(columns),
        "row_count": len(sample),
        "date_min": sample["Date"].min().date().isoformat(),
        "date_max": sample["Date"].max().date().isoformat(),
        "required_format_decision": required,
    }


def validate_sample(sample):
    expected = pd.date_range(TEST_START, TEST_END, freq="D")
    if len(sample) != len(expected):
        raise ValueError(f"sample_submission row count {len(sample)} != expected {len(expected)}")
    if not pd.Index(sample["Date"]).equals(pd.Index(expected)):
        raise ValueError("sample_submission Date column is not the expected daily test sequence.")
    if sample.columns.duplicated().any():
        raise ValueError("sample_submission contains duplicated columns.")


def fold_cache_path(fold):
    return FULL_CACHE_DIR / f"fold_{fold}_full_raw_predictions.csv"


def load_fold_caches(folds):
    frames = []
    status = []
    required_cols = {
        "fold",
        "Date",
        "horizon_day",
        "actual_Revenue",
        "pred_Revenue_direct",
        "pred_Revenue_component",
        "pred_Revenue_current",
    }
    for fold in folds:
        path = fold_cache_path(fold)
        if not path.exists():
            raise FileNotFoundError(f"Missing required fold cache: {path}")
        df = pd.read_csv(path, parse_dates=["Date"])
        missing = sorted(required_cols.difference(df.columns))
        if missing:
            raise ValueError(f"{path.name} missing columns: {missing}")
        _, val_start, val_end = FOLD_SPECS[fold]
        expected = pd.date_range(val_start, val_end, freq="D")
        if len(df) != len(expected) or not pd.Index(df["Date"]).equals(pd.Index(expected)):
            raise ValueError(f"{path.name} does not match Fold {fold} expected dates.")
        frames.append(df)
        status.append({"fold": fold, "cache": path.relative_to(PROJECT_ROOT).as_posix(), "status": "loaded"})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(status)


def horizon_bucket(day):
    if day <= 30:
        return "days_1_30"
    if day <= 90:
        return "days_31_90"
    if day <= 180:
        return "days_91_180"
    if day <= 365:
        return "days_181_365"
    return "days_366_plus"


def metric_dict(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": mean_squared_error(actual, pred) ** 0.5,
        "R2": r2_score(actual, pred),
        "bias": float(np.mean(pred - actual)),
        "pred_actual_mean_ratio": float(np.mean(pred) / np.mean(actual)) if np.mean(actual) else np.nan,
    }


def evaluate_prediction(df, strategy, pred, params=None):
    params = params or {}
    rows = []
    temp = df[["fold", "Date", "horizon_day", "actual_Revenue"]].copy()
    temp["pred"] = np.asarray(pred, dtype=float)
    for fold, part in temp.groupby("fold", sort=False):
        rows.append({"strategy": strategy, "fold": fold, **params, **metric_dict(part["actual_Revenue"], part["pred"])})
    return rows


def summarize_strategy_rows(rows, keys):
    details = pd.DataFrame(rows)
    all_summary = details.groupby(["strategy", *keys]).agg(
        MAE=("MAE", "mean"),
        RMSE=("RMSE", "mean"),
        R2=("R2", "mean"),
        bias=("bias", "mean"),
        pred_actual_mean_ratio=("pred_actual_mean_ratio", "mean"),
    ).reset_index()
    return details, all_summary.sort_values(["MAE", "RMSE"]).reset_index(drop=True)


def current_direct_component_results(df):
    rows = []
    rows.extend(evaluate_prediction(df, "current", df["pred_Revenue_current"]))
    rows.extend(evaluate_prediction(df, "direct", df["pred_Revenue_direct"]))
    rows.extend(evaluate_prediction(df, "component", df["pred_Revenue_component"]))
    return summarize_strategy_rows(rows, [])


def global_scale_results(df):
    rows = []
    for scale in REVENUE_SCALES:
        rows.extend(evaluate_prediction(df, "global_scale", df["pred_Revenue_current"] * scale, {"scale": scale}))
    return summarize_strategy_rows(rows, ["scale"])


def horizon_scale_results(df):
    rows = []
    for year1_scale, year2_scale in itertools.product(YEAR1_SCALES, YEAR2_SCALES):
        factor = np.where(df["horizon_day"] <= 365, year1_scale, year2_scale)
        rows.extend(
            evaluate_prediction(
                df,
                "horizon_scale",
                df["pred_Revenue_current"] * factor,
                {"year1_scale": year1_scale, "year2_scale": year2_scale},
            )
        )
    return summarize_strategy_rows(rows, ["year1_scale", "year2_scale"])


def recent_level_results(df, history):
    rows = []
    for fold, part in df.groupby("fold", sort=False):
        train_end = FOLD_SPECS[fold][0]
        train_hist = history[history["Date"] <= train_end].copy()
        first30 = part.head(30)["pred_Revenue_current"].mean()
        for window in RECENT_WINDOWS:
            multiplier = train_hist.tail(window)["Revenue"].mean() / first30 if first30 > 0 else 1.0
            rows.extend(
                evaluate_prediction(
                    part,
                    "recent_level",
                    part["pred_Revenue_current"] * multiplier,
                    {"window": window, "fold_multiplier": multiplier},
                )
            )
    details = pd.DataFrame(rows)
    summary = details.groupby(["strategy", "window"]).agg(
        MAE=("MAE", "mean"),
        RMSE=("RMSE", "mean"),
        R2=("R2", "mean"),
        bias=("bias", "mean"),
        pred_actual_mean_ratio=("pred_actual_mean_ratio", "mean"),
    ).reset_index().sort_values(["MAE", "RMSE"]).reset_index(drop=True)
    return details, summary


def ensemble_results(df):
    rows = []
    for w in ENSEMBLE_WEIGHTS:
        pred = w * df["pred_Revenue_direct"] + (1 - w) * df["pred_Revenue_component"]
        rows.extend(evaluate_prediction(df, "ensemble", pred, {"w": w}))
    return summarize_strategy_rows(rows, ["w"])


def ensemble_scale_results(df):
    rows = []
    for w, scale in itertools.product(ENSEMBLE_WEIGHTS, REVENUE_SCALES):
        pred = (w * df["pred_Revenue_direct"] + (1 - w) * df["pred_Revenue_component"]) * scale
        rows.extend(evaluate_prediction(df, "ensemble_scale", pred, {"w": w, "scale": scale}))
    return summarize_strategy_rows(rows, ["w", "scale"])


def strategy_metrics_table(df, strategies):
    rows = []
    for name, pred in strategies.items():
        rows.extend(evaluate_prediction(df, name, pred))
    return summarize_strategy_rows(rows, [])[1]


def mae_by_horizon(df, strategy_predictions):
    rows = []
    temp = df[["fold", "horizon_day", "actual_Revenue"]].copy()
    temp["horizon_bucket"] = temp["horizon_day"].map(horizon_bucket)
    for strategy, pred in strategy_predictions.items():
        temp["pred"] = np.asarray(pred, dtype=float)
        for (fold, bucket), part in temp.groupby(["fold", "horizon_bucket"], sort=False):
            rows.append(
                {
                    "strategy": strategy,
                    "fold": fold,
                    "horizon_bucket": bucket,
                    "Revenue_MAE": mean_absolute_error(part["actual_Revenue"], part["pred"]),
                    "rows": len(part),
                }
            )
    return pd.DataFrame(rows)


def select_best(summary):
    return summary.sort_values(["MAE", "RMSE"]).iloc[0].to_dict()


def fit_revenue_system(history):
    history = om.add_derived_columns(history).sort_values("Date").reset_index(drop=True)
    reference_cache = om.make_reference_cache(history)
    models = {
        "Revenue": om.fit_model(history, "Revenue", "Revenue", reference_cache),
        "order_count": om.fit_model(history, "order_count", "order_count", reference_cache),
        "AOV": om.fit_model(history, "AOV", "AOV", reference_cache),
    }
    return RevenueSystem(history=history, reference_cache=reference_cache, models=models)


def forecast_revenue_details(system, dates, revenue_direct_weight):
    state = system.history.set_index("Date").sort_index().copy()
    rows = []
    for i, date in enumerate(pd.Series(pd.to_datetime(dates)), start=1):
        pred = {}
        for model_name in REVENUE_MODEL_NAMES:
            x = om.make_single_feature_row(
                date, state, system.history["Date"].min(), system.reference_cache, model_name
            )
            pred[model_name] = float(system.models[model_name].predict(x)[0])
        component = pred["order_count"] * pred["AOV"]
        revenue = revenue_direct_weight * pred["Revenue"] + (1 - revenue_direct_weight) * component
        aov = revenue / pred["order_count"] if pred["order_count"] > 1e-9 else pred["AOV"]
        state.loc[date, ["Revenue", "order_count", "AOV"]] = [revenue, pred["order_count"], aov]
        rows.append(
            {
                "Date": date,
                "horizon_day": i,
                "Revenue": round(max(revenue, 0.0), 2),
                "pred_order_count": pred["order_count"],
                "pred_AOV": pred["AOV"],
            }
        )
    return pd.DataFrame(rows)


def future_predictions_cache_path():
    return CACHE_DIR / "full_history_revenue_predictions.csv"


def load_or_build_future_predictions(history, sample, refresh=False):
    path = future_predictions_cache_path()
    required = {"Date", "horizon_day", "pred_Revenue_direct", "pred_Revenue_component", "pred_Revenue_current"}
    if path.exists() and not refresh:
        cached = pd.read_csv(path, parse_dates=["Date"])
        if required.issubset(cached.columns) and pd.Index(cached["Date"]).equals(pd.Index(sample["Date"])):
            return cached, "loaded"

    print("Training full-history Revenue/order_count/AOV models for final candidates", flush=True)
    system = fit_revenue_system(history)
    direct = forecast_revenue_details(system, sample["Date"], revenue_direct_weight=1.0)
    component = forecast_revenue_details(system, sample["Date"], revenue_direct_weight=0.0)
    out = pd.DataFrame(
        {
            "Date": sample["Date"],
            "horizon_day": np.arange(1, len(sample) + 1),
            "pred_Revenue_direct": direct["Revenue"],
            "pred_Revenue_component": component["Revenue"],
            "pred_Revenue_current": direct["Revenue"],
            "pred_order_count": direct["pred_order_count"],
            "pred_AOV": direct["pred_AOV"],
        }
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out, "recomputed"


def apply_horizon_strategy(df, year1_scale, year2_scale):
    factor = np.where(df["horizon_day"] <= 365, year1_scale, year2_scale)
    return df["pred_Revenue_current"].to_numpy() * factor


def apply_recent_level_full(df, history, window):
    first30 = df.head(30)["pred_Revenue_current"].mean()
    multiplier = history.tail(int(window))["Revenue"].mean() / first30 if first30 > 0 else 1.0
    return df["pred_Revenue_current"].to_numpy() * multiplier


def apply_ensemble_scale(df, w, scale):
    return (w * df["pred_Revenue_direct"].to_numpy() + (1 - w) * df["pred_Revenue_component"].to_numpy()) * scale


def write_submission(sample, revenue, filename):
    out = sample.copy()
    out["Revenue"] = np.maximum(np.asarray(revenue, dtype=float), 0.0).round(2)
    if out["Revenue"].isna().any():
        raise ValueError(f"{filename} contains NaN Revenue.")
    if (out["Revenue"] < 0).any():
        raise ValueError(f"{filename} contains negative Revenue.")
    if not pd.Index(out["Date"]).equals(pd.Index(sample["Date"])):
        raise ValueError(f"{filename} changed sample row order.")
    if out.isna().any().any():
        raise ValueError(f"{filename} contains NaN in required output columns.")
    write = out.copy()
    write["Date"] = write["Date"].dt.strftime("%Y-%m-%d")
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBMISSION_DIR / filename
    write.to_csv(path, index=False)
    return {
        "file": filename,
        "columns": ", ".join(write.columns),
        "rows": len(write),
        "date_min": write["Date"].min(),
        "date_max": write["Date"].max(),
        "Revenue_min": out["Revenue"].min(),
        "Revenue_mean": out["Revenue"].mean(),
        "Revenue_max": out["Revenue"].max(),
        "Revenue_nan_count": int(out["Revenue"].isna().sum()),
        "Revenue_negative_count": int((out["Revenue"] < 0).sum()),
    }


def generate_submissions(sample, history, full_df, best):
    rows = []
    rows.append(write_submission(sample, full_df["pred_Revenue_current"], SUBMISSION_FILES["current"]))
    rows.append(
        write_submission(
            sample,
            full_df["pred_Revenue_current"].to_numpy() * float(best["global_scale"]["scale"]),
            SUBMISSION_FILES["scaled_best"],
        )
    )
    rows.append(
        write_submission(
            sample,
            apply_horizon_strategy(
                full_df,
                float(best["horizon_scale"]["year1_scale"]),
                float(best["horizon_scale"]["year2_scale"]),
            ),
            SUBMISSION_FILES["horizon_scaled_best"],
        )
    )
    rows.append(
        write_submission(
            sample,
            apply_recent_level_full(full_df, history, int(best["recent_level"]["window"])),
            SUBMISSION_FILES["recent_level_best"],
        )
    )
    rows.append(
        write_submission(
            sample,
            apply_ensemble_scale(
                full_df,
                float(best["ensemble_scale"]["w"]),
                float(best["ensemble_scale"]["scale"]),
            ),
            SUBMISSION_FILES["ensemble_scaled_best"],
        )
    )
    return pd.DataFrame(rows)


def fmt_table(df, max_rows=30):
    if df is None or df.empty:
        return "Not available"
    shown = df.head(max_rows).copy()
    for col in shown.select_dtypes(include=[np.number]).columns:
        shown[col] = shown[col].map(lambda x: f"{x:,.4f}" if pd.notna(x) else "")
    return shown.to_markdown(index=False)


def build_report(context):
    lines = [
        "# Revenue-Only Experiment Report",
        "",
        "## A. Required Submission Format",
        fmt_table(pd.DataFrame([context["submission_format"]])),
        "",
        "The competition objective is Revenue only. Because `sample_submission.csv` includes `COGS`, candidate files preserve that column from the sample template for format compatibility, but no COGS model or COGS calibration is optimized here.",
        "",
        "## B. Baseline Revenue Metrics",
        fmt_table(context["base_summary"], max_rows=10),
        "",
        "## C. Calibration Results",
        "Global scale:",
        fmt_table(context["global_scale_summary"], max_rows=15),
        "",
        "Horizon-specific scale:",
        fmt_table(context["horizon_scale_summary"], max_rows=15),
        "",
        "Recent-level adjustment:",
        fmt_table(context["recent_level_summary"], max_rows=10),
        "",
        "## D. Ensemble Results",
        "Direct/component ensemble:",
        fmt_table(context["ensemble_summary"], max_rows=12),
        "",
        "Ensemble + global scale:",
        fmt_table(context["ensemble_scale_summary"], max_rows=15),
        "",
        "## E. MAE By Horizon Bucket",
        fmt_table(context["horizon_mae"], max_rows=40),
        "",
        "## F. Recommended Submission Order",
        fmt_table(context["recommended"], max_rows=10),
        "",
        "## G. Generated Submissions",
        fmt_table(context["submission_summary"], max_rows=10),
        "",
        "## H. Warning",
        "Do not overfit to the public leaderboard. These candidates are selected from Fold C/D recent validation only; repeated public-LB tuning can degrade hidden performance.",
        "",
        "## I. Runtime",
        f"- Started: `{context['start_time']}`",
        f"- Finished: `{context['end_time']}`",
        f"- Runtime seconds: `{context['runtime_seconds']:.1f}`",
        f"- Fold forecast cache source: `{FULL_CACHE_DIR.name}`",
        f"- Revenue-only cache directory: `{CACHE_DIR.name}`",
        "",
        "Cache status:",
        fmt_table(context["cache_status"], max_rows=10),
    ]
    return "\n".join(lines)


def main():
    start = time.time()
    start_label = pd.Timestamp.now().isoformat(timespec="seconds")
    args = parse_args()
    folds = selected_folds(args.folds)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])
    validate_sample(sample)
    submission_format = inspect_submission_format(sample)

    print("Loading existing Fold C/D full diagnostic caches", flush=True)
    fold_df, fold_cache_status = load_fold_caches(folds)
    history = om.load_history()

    print("Evaluating Revenue-only strategies", flush=True)
    base_details, base_summary = current_direct_component_results(fold_df)
    scale_details, scale_summary = global_scale_results(fold_df)
    horizon_details, horizon_summary = horizon_scale_results(fold_df)
    recent_details, recent_summary = recent_level_results(fold_df, history)
    ensemble_details, ensemble_summary = ensemble_results(fold_df)
    ensemble_scale_details, ensemble_scale_summary = ensemble_scale_results(fold_df)

    best = {
        "global_scale": select_best(scale_summary),
        "horizon_scale": select_best(horizon_summary),
        "recent_level": select_best(recent_summary),
        "ensemble": select_best(ensemble_summary),
        "ensemble_scale": select_best(ensemble_scale_summary),
    }

    strategy_predictions = {
        "current": fold_df["pred_Revenue_current"].to_numpy(),
        "scaled_best": fold_df["pred_Revenue_current"].to_numpy() * float(best["global_scale"]["scale"]),
        "horizon_scaled_best": apply_horizon_strategy(
            fold_df,
            float(best["horizon_scale"]["year1_scale"]),
            float(best["horizon_scale"]["year2_scale"]),
        ),
        "recent_level_best": np.concatenate(
            [
                part["pred_Revenue_current"].to_numpy()
                * (
                    history[history["Date"] <= FOLD_SPECS[fold][0]].tail(int(best["recent_level"]["window"]))["Revenue"].mean()
                    / part.head(30)["pred_Revenue_current"].mean()
                )
                for fold, part in fold_df.groupby("fold", sort=False)
            ]
        ),
        "ensemble_scaled_best": apply_ensemble_scale(
            fold_df,
            float(best["ensemble_scale"]["w"]),
            float(best["ensemble_scale"]["scale"]),
        ),
    }
    horizon_mae = mae_by_horizon(fold_df, strategy_predictions)

    final_comparison = strategy_metrics_table(fold_df, strategy_predictions)
    recommended = final_comparison.sort_values(["MAE", "RMSE"]).copy()
    file_map = {
        "current": SUBMISSION_FILES["current"],
        "scaled_best": SUBMISSION_FILES["scaled_best"],
        "horizon_scaled_best": SUBMISSION_FILES["horizon_scaled_best"],
        "recent_level_best": SUBMISSION_FILES["recent_level_best"],
        "ensemble_scaled_best": SUBMISSION_FILES["ensemble_scaled_best"],
    }
    recommended["file"] = recommended["strategy"].map(file_map)
    recommended.insert(0, "rank", np.arange(1, len(recommended) + 1))

    if args.skip_submissions:
        submission_summary = pd.DataFrame([{"file": "skipped", "reason": "--skip-submissions"}])
        future_cache_status = pd.DataFrame([{"fold": "future", "cache": future_predictions_cache_path().relative_to(PROJECT_ROOT).as_posix(), "status": "skipped"}])
    else:
        full_df, future_status = load_or_build_future_predictions(history, sample, refresh=args.refresh_cache)
        submission_summary = generate_submissions(sample, history, full_df, best)
        future_cache_status = pd.DataFrame([{"fold": "future", "cache": future_predictions_cache_path().relative_to(PROJECT_ROOT).as_posix(), "status": future_status}])

    cache_status = pd.concat([fold_cache_status, future_cache_status], ignore_index=True)

    context = {
        "submission_format": submission_format,
        "base_summary": base_summary,
        "global_scale_summary": scale_summary,
        "horizon_scale_summary": horizon_summary,
        "recent_level_summary": recent_summary,
        "ensemble_summary": ensemble_summary,
        "ensemble_scale_summary": ensemble_scale_summary,
        "horizon_mae": horizon_mae,
        "recommended": recommended,
        "submission_summary": submission_summary,
        "cache_status": cache_status,
        "start_time": start_label,
        "end_time": pd.Timestamp.now().isoformat(timespec="seconds"),
        "runtime_seconds": time.time() - start,
    }

    pd.concat([base_details, scale_details, horizon_details, recent_details, ensemble_details, ensemble_scale_details], ignore_index=True).to_csv(CACHE_DIR / "strategy_fold_metrics.csv", index=False)
    final_comparison.to_csv(CACHE_DIR / "recommended_strategy_metrics.csv", index=False)
    horizon_mae.to_csv(CACHE_DIR / "horizon_bucket_mae.csv", index=False)
    submission_summary.to_csv(CACHE_DIR / "submission_summary.csv", index=False)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(context), encoding="utf-8")
    print(f"Saved report to {REPORT_PATH.name}", flush=True)
    print(fmt_table(recommended, max_rows=10), flush=True)


if __name__ == "__main__":
    main()
