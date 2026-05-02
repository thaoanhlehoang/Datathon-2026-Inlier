import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

import our_method_forecast as om

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
CACHE_DIR = PROJECT_ROOT / "artifacts" / "cache" / "diagnostic_cache_fast"
REPORT_FILE = OUTPUT_DIR / "fast_revenue_diagnostic_report.md"

FOLDS = [
    ("C", pd.Timestamp("2020-12-31"), pd.Timestamp("2021-01-01"), pd.Timestamp("2022-07-01")),
    ("D", pd.Timestamp("2021-12-31"), pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
]
OPTIONAL_FOLD_B = (
    "B",
    pd.Timestamp("2019-12-31"),
    pd.Timestamp("2020-01-01"),
    pd.Timestamp("2021-07-01"),
)

COARSE_SCALES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05]
RECENT_WINDOWS = [14, 30, 60]
GROWTH_RATES = [-0.20, -0.15, -0.10, -0.05, 0.00, 0.05]
ENSEMBLE_WEIGHTS = [0.7, 0.8, 0.9, 1.0]
CURRENT_REVENUE_WEIGHT = 1.0
CURRENT_COGS_WEIGHT = 0.6


def parse_args():
    parser = argparse.ArgumentParser(description="Fast Revenue-focused calibration diagnostics.")
    parser.add_argument("--refresh-cache", action="store_true", help="Retrain folds and overwrite cached fold forecasts.")
    parser.add_argument("--fast-model", action="store_true", help="Use reduced LightGBM complexity for diagnostics and output generation.")
    parser.add_argument("--full-model", action="store_true", help="Use original model settings.")
    parser.add_argument("--include-fold-b", action="store_true", help="Also evaluate Fold B.")
    return parser.parse_args()


def patch_fast_model():
    def make_fast_model():
        return om.LGBMRegressor(
            objective="regression",
            random_state=om.RANDOM_STATE,
            n_estimators=300,
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=25,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.4,
            n_jobs=-1,
            verbosity=-1,
        )

    om.make_model = make_fast_model


def rmse(actual, pred):
    return mean_squared_error(actual, pred) ** 0.5


def smape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.abs(actual) + np.abs(pred)
    denom = np.where(denom == 0, np.nan, denom)
    return float(np.nanmean(2 * np.abs(actual - pred) / denom) * 100)


def revenue_metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": rmse(actual, pred),
        "R2": r2_score(actual, pred),
        "MAPE": mean_absolute_percentage_error(actual, pred) * 100,
        "sMAPE": smape(actual, pred),
        "bias": float(np.mean(pred - actual)),
        "prediction_mean": float(np.mean(pred)),
        "actual_mean": float(np.mean(actual)),
        "prediction_actual_ratio": float(np.mean(pred) / np.mean(actual)),
    }


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


def cache_path(fold_name, mode):
    return CACHE_DIR / f"fold_{fold_name}_{mode}_raw_predictions.csv"


def build_fold_cache(history, fold, mode, refresh_cache):
    fold_name, train_end, val_start, val_end = fold
    path = cache_path(fold_name, mode)
    if path.exists() and not refresh_cache:
        print(f"Loading cached Fold {fold_name}: {path.relative_to(PROJECT_ROOT)}", flush=True)
        return pd.read_csv(path, parse_dates=["Date"])

    print(f"Training Fold {fold_name}: train <= {train_end.date()}, validate {val_start.date()} to {val_end.date()}", flush=True)
    train = history[history["Date"] <= train_end].copy()
    actual = history[(history["Date"] >= val_start) & (history["Date"] <= val_end)].copy()
    system = om.fit_base_system(train)

    direct = om.ForecastSystem(system.history, system.reference_cache, system.models, 1.0, 1.0).forecast(actual["Date"])
    component = om.ForecastSystem(system.history, system.reference_cache, system.models, 0.0, 0.0).forecast(actual["Date"])

    out = pd.DataFrame(
        {
            "Date": actual["Date"].to_numpy(),
            "actual_Revenue": actual["Revenue"].to_numpy(),
            "pred_Revenue_direct": direct["Revenue"].to_numpy(),
            "pred_Revenue_component": component["Revenue"].to_numpy(),
            "pred_Revenue_current": direct["Revenue"].to_numpy(),
            "actual_COGS": actual["COGS"].to_numpy(),
            "pred_COGS_current": direct["COGS"].to_numpy(),
            "horizon_day": np.arange(1, len(actual) + 1),
            "fold": fold_name,
            "train_end": train_end.date().isoformat(),
            "validation_start": val_start.date().isoformat(),
            "validation_end": val_end.date().isoformat(),
            "model_mode": mode,
        }
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Saved Fold {fold_name} cache: {path.relative_to(PROJECT_ROOT)}", flush=True)
    return out


def load_or_build_folds(history, folds, mode, refresh_cache):
    return [build_fold_cache(history, fold, mode, refresh_cache) for fold in folds]


def baseline_tables(fold_frames):
    metric_rows = []
    bias_rows = []
    bucket_rows = []
    for df in fold_frames:
        fold = df["fold"].iloc[0]
        metrics = revenue_metrics(df["actual_Revenue"], df["pred_Revenue_current"])
        metric_rows.append({"fold": fold, **{k: metrics[k] for k in ["MAE", "RMSE", "R2", "MAPE", "sMAPE"]}})
        bias_rows.append(
            {
                "fold": fold,
                "actual_mean": metrics["actual_mean"],
                "prediction_mean": metrics["prediction_mean"],
                "bias": metrics["bias"],
                "prediction_actual_ratio": metrics["prediction_actual_ratio"],
            }
        )
        tmp = df.copy()
        tmp["horizon_bucket"] = tmp["horizon_day"].map(horizon_bucket)
        for bucket, part in tmp.groupby("horizon_bucket", sort=False):
            bucket_rows.append(
                {
                    "fold": fold,
                    "horizon_bucket": bucket,
                    "Revenue_MAE": mean_absolute_error(part["actual_Revenue"], part["pred_Revenue_current"]),
                    "rows": len(part),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(bias_rows), pd.DataFrame(bucket_rows)


def summarize_experiment(rows, config_cols):
    details = pd.DataFrame(rows)
    summary = (
        details.groupby(config_cols)
        .agg(
            avg_MAE=("MAE", "mean"),
            std_MAE=("MAE", "std"),
            avg_RMSE=("RMSE", "mean"),
            avg_R2=("R2", "mean"),
            avg_bias=("bias", "mean"),
            avg_ratio=("prediction_actual_ratio", "mean"),
        )
        .reset_index()
        .sort_values(["avg_MAE", "std_MAE"], na_position="last")
    )
    return details, summary


def eval_prediction_set(fold_frames, name, pred_fn, config):
    rows = []
    for df in fold_frames:
        pred = pred_fn(df)
        metrics = revenue_metrics(df["actual_Revenue"], pred)
        rows.append({"fold": df["fold"].iloc[0], "experiment": name, **config, **metrics})
    return rows


def evaluate_scaling(fold_frames):
    rows = []
    for scale in COARSE_SCALES:
        rows.extend(eval_prediction_set(fold_frames, "scale", lambda df, s=scale: df["pred_Revenue_current"] * s, {"scale": scale}))
    _, coarse = summarize_experiment(rows, ["scale"])
    best = float(coarse.iloc[0]["scale"])
    refined = sorted({round(x, 2) for x in np.arange(best - 0.05, best + 0.0501, 0.01) if x > 0})
    for scale in refined:
        if scale not in COARSE_SCALES:
            rows.extend(eval_prediction_set(fold_frames, "scale", lambda df, s=scale: df["pred_Revenue_current"] * s, {"scale": scale}))
    details, summary = summarize_experiment(rows, ["scale"])
    return details, summary


def recent_level_prediction(df, window):
    train_end = pd.Timestamp(df["train_end"].iloc[0])
    history = om.load_history()
    recent = history[(history["Date"] <= train_end)].tail(window)["Revenue"].mean()
    first30 = df.head(30)["pred_Revenue_current"].mean()
    factor = recent / first30 if first30 > 0 else 1.0
    return df["pred_Revenue_current"] * factor


def evaluate_recent_level(fold_frames):
    rows = []
    for window in RECENT_WINDOWS:
        rows.extend(eval_prediction_set(fold_frames, "recent_level", lambda df, w=window: recent_level_prediction(df, w), {"window": window}))
    return summarize_experiment(rows, ["window"])


def trend_prediction(df, growth_rate):
    multiplier = np.power(1 + growth_rate, (df["horizon_day"] - 1) / 365.0)
    return df["pred_Revenue_current"] * multiplier


def evaluate_trends(fold_frames):
    rows = []
    for growth_rate in GROWTH_RATES:
        rows.extend(eval_prediction_set(fold_frames, "trend", lambda df, g=growth_rate: trend_prediction(df, g), {"growth_rate": growth_rate}))
    return summarize_experiment(rows, ["growth_rate"])


def ensemble_prediction(df, weight):
    return weight * df["pred_Revenue_direct"] + (1 - weight) * df["pred_Revenue_component"]


def evaluate_ensemble(fold_frames, scale_summary):
    rows = []
    for weight in ENSEMBLE_WEIGHTS:
        rows.extend(eval_prediction_set(fold_frames, "ensemble", lambda df, w=weight: ensemble_prediction(df, w), {"weight": weight}))
    ensemble_details, ensemble_summary = summarize_experiment(rows, ["weight"])

    best_weights = ensemble_summary.head(2)["weight"].astype(float).tolist()
    best_scales = scale_summary.head(3)["scale"].astype(float).tolist()
    combo_rows = []
    for weight, scale in itertools.product(best_weights, best_scales):
        combo_rows.extend(
            eval_prediction_set(
                fold_frames,
                "ensemble_scale",
                lambda df, w=weight, s=scale: ensemble_prediction(df, w) * s,
                {"weight": weight, "scale": scale},
            )
        )
    combo_details, combo_summary = summarize_experiment(combo_rows, ["weight", "scale"])
    return ensemble_details, ensemble_summary, combo_details, combo_summary


def choose_row(summary):
    best_mae = summary["avg_MAE"].min()
    near_best = summary[summary["avg_MAE"] <= best_mae * 1.01].copy()
    return near_best.sort_values(["std_MAE", "avg_MAE"], na_position="last").iloc[0]


def fit_full_outputs(mode):
    history = om.load_history()
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])
    print(f"Training full-history {mode} model for fast candidate submissions", flush=True)
    system = om.fit_base_system(history)
    direct = om.ForecastSystem(system.history, system.reference_cache, system.models, 1.0, 1.0).forecast(sample["Date"])
    component = om.ForecastSystem(system.history, system.reference_cache, system.models, 0.0, 0.0).forecast(sample["Date"])
    current_cogs = om.ForecastSystem(system.history, system.reference_cache, system.models, CURRENT_REVENUE_WEIGHT, CURRENT_COGS_WEIGHT).forecast(sample["Date"])["COGS"]
    frame = pd.DataFrame(
        {
            "Date": sample["Date"],
            "Revenue_current": direct["Revenue"].to_numpy(),
            "Revenue_direct": direct["Revenue"].to_numpy(),
            "Revenue_component": component["Revenue"].to_numpy(),
            "COGS_current": current_cogs.to_numpy(),
            "horizon_day": np.arange(1, len(sample) + 1),
        }
    )
    return history, sample, frame


def apply_recent_full(history, frame, window):
    recent = history.tail(window)["Revenue"].mean()
    first30 = frame.head(30)["Revenue_current"].mean()
    factor = recent / first30 if first30 > 0 else 1.0
    return frame["Revenue_current"] * factor


def write_submission(sample, revenue, cogs, filename):
    out = sample[["Date"]].copy()
    out["Revenue"] = np.maximum(np.asarray(revenue, dtype=float), 0.0).round(2)
    out["COGS"] = np.maximum(np.asarray(cogs, dtype=float), 0.0).round(2)
    if out[["Revenue", "COGS"]].isna().any().any():
        raise ValueError(f"{filename} contains NaN values.")
    if not out["Date"].equals(sample["Date"]):
        raise ValueError(f"{filename} date order does not match sample_submission.csv.")
    write = out.copy()
    write["Date"] = write["Date"].dt.strftime("%Y-%m-%d")
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBMISSION_DIR / filename
    write.to_csv(path, index=False)
    return {
        "file": filename,
        "rows": len(write),
        "date_min": write["Date"].min(),
        "date_max": write["Date"].max(),
        "revenue_min": float(out["Revenue"].min()),
        "cogs_min": float(out["COGS"].min()),
        "nan_count": int(out[["Revenue", "COGS"]].isna().sum().sum()),
        "negative_count": int((out[["Revenue", "COGS"]] < 0).sum().sum()),
    }


def generate_candidates(history, sample, frame, selected):
    cogs = frame["COGS_current"].to_numpy()
    submissions = []
    submissions.append(write_submission(sample, frame["Revenue_current"], cogs, "submission_fast_current.csv"))
    submissions.append(write_submission(sample, frame["Revenue_current"] * selected["scale"], cogs, "submission_fast_scaled_best.csv"))
    submissions.append(write_submission(sample, apply_recent_full(history, frame, int(selected["recent_window"])), cogs, "submission_fast_recent_level_best.csv"))
    submissions.append(write_submission(sample, trend_prediction(frame.rename(columns={"Revenue_current": "pred_Revenue_current"}), selected["growth_rate"]), cogs, "submission_fast_trend_best.csv"))

    blended_revenue = (selected["blend_weight"] * frame["Revenue_direct"] + (1 - selected["blend_weight"]) * frame["Revenue_component"]) * selected["blend_scale"]
    submissions.append(write_submission(sample, blended_revenue, cogs, "submission_fast_blended_best.csv"))

    best_revenue = selected["best_revenue_fn"](history, frame)
    capped_cogs = np.minimum(cogs, best_revenue * 1.10)
    submissions.append(write_submission(sample, best_revenue, capped_cogs, "submission_fast_best_cogs_cap_110.csv"))
    return pd.DataFrame(submissions)


def fmt_table(df, max_rows=20):
    shown = df.head(max_rows).copy()
    for col in shown.select_dtypes(include=[np.number]).columns:
        shown[col] = shown[col].map(lambda x: f"{x:,.4f}" if pd.notna(x) else "")
    return shown.to_markdown(index=False)


def build_report(mode, cache_used, folds, baseline, bias, buckets, scale_summary, recent_summary, trend_summary, ensemble_summary, combo_summary, submissions, selected):
    fold_names = ", ".join([f[0] for f in folds])
    report_lines = [
        "# Fast Revenue Diagnostic Report",
        "",
        "## A. Runtime mode",
        f"- Model mode: `{mode}`.",
        f"- Cache used for all requested folds: `{cache_used}`.",
        f"- Evaluated folds: `{fold_names}`.",
        "- Future data usage: none. Fold features come from historical data, recursive predictions, and deterministic dates only.",
        "",
        "## B. Baseline fold metrics",
        fmt_table(baseline),
        "",
        "Revenue MAE by horizon bucket:",
        fmt_table(buckets, max_rows=30),
        "",
        "## C. Bias diagnosis",
        fmt_table(bias),
        "",
        "## D. Scaling results",
        fmt_table(scale_summary, max_rows=10),
        f"Selected scale: `{selected['scale']}`.",
        "",
        "## E. Recent-level adjustment results",
        fmt_table(recent_summary, max_rows=10),
        f"Selected recent-level window: `last_{selected['recent_window']}`.",
        "",
        "## F. Trend multiplier results",
        fmt_table(trend_summary, max_rows=10),
        f"Selected annual growth rate: `{selected['growth_rate']}`.",
        "",
        "## G. Ensemble results",
        "Direct/component ensemble:",
        fmt_table(ensemble_summary, max_rows=10),
        "",
        "Best small ensemble+scale combinations:",
        fmt_table(combo_summary, max_rows=10),
        f"Selected blended candidate: weight=`{selected['blend_weight']}`, scale=`{selected['blend_scale']}`.",
        "",
        "## H. Recommended Kaggle trial order",
        fmt_table(submissions[["trial_rank", "file", "rows", "date_min", "date_max", "nan_count", "negative_count"]], max_rows=10),
        "",
        "## I. Warnings",
        "- Do not tune only from Kaggle public leaderboard feedback; it can overfit the public split.",
        "- Use few submissions and treat public Revenue MAE as noisy feedback.",
        "- Recent folds disagree on scale direction, so prefer stable improvements over extreme factors.",
        "- Hidden scoring may include COGS, RMSE, or R2 even if the public board emphasizes Revenue MAE.",
    ]
    return "\n".join(report_lines)


def main():
    args = parse_args()
    if args.fast_model and args.full_model:
        raise ValueError("Use only one of --fast-model or --full-model.")

    mode = "full" if args.full_model else "fast"
    if mode == "fast":
        patch_fast_model()

    folds = FOLDS.copy()
    if args.include_fold_b:
        folds = [OPTIONAL_FOLD_B] + folds

    history = om.load_history()
    expected_paths = [cache_path(f[0], mode) for f in folds]
    cache_used = all(p.exists() for p in expected_paths) and not args.refresh_cache
    fold_frames = load_or_build_folds(history, folds, mode, args.refresh_cache)

    baseline, bias, buckets = baseline_tables(fold_frames)
    _, scale_summary = evaluate_scaling(fold_frames)
    _, recent_summary = evaluate_recent_level(fold_frames)
    _, trend_summary = evaluate_trends(fold_frames)
    _, ensemble_summary, _, combo_summary = evaluate_ensemble(fold_frames, scale_summary)

    selected_scale = choose_row(scale_summary)
    selected_recent = choose_row(recent_summary)
    selected_trend = choose_row(trend_summary)
    selected_combo = choose_row(combo_summary)

    candidate_scores = pd.DataFrame(
        [
            {"name": "scaled", "avg_MAE": selected_scale["avg_MAE"], "std_MAE": selected_scale["std_MAE"]},
            {"name": "recent", "avg_MAE": selected_recent["avg_MAE"], "std_MAE": selected_recent["std_MAE"]},
            {"name": "trend", "avg_MAE": selected_trend["avg_MAE"], "std_MAE": selected_trend["std_MAE"]},
            {"name": "blended", "avg_MAE": selected_combo["avg_MAE"], "std_MAE": selected_combo["std_MAE"]},
        ]
    ).sort_values(["avg_MAE", "std_MAE"])

    selected = {
        "scale": float(selected_scale["scale"]),
        "recent_window": int(selected_recent["window"]),
        "growth_rate": float(selected_trend["growth_rate"]),
        "blend_weight": float(selected_combo["weight"]),
        "blend_scale": float(selected_combo["scale"]),
    }

    best_name = candidate_scores.iloc[0]["name"]
    if best_name == "scaled":
        selected["best_revenue_fn"] = lambda _history, frame: frame["Revenue_current"] * selected["scale"]
    elif best_name == "recent":
        selected["best_revenue_fn"] = lambda hist, frame: apply_recent_full(hist, frame, selected["recent_window"])
    elif best_name == "trend":
        selected["best_revenue_fn"] = lambda _history, frame: trend_prediction(frame.rename(columns={"Revenue_current": "pred_Revenue_current"}), selected["growth_rate"])
    else:
        selected["best_revenue_fn"] = lambda _history, frame: (selected["blend_weight"] * frame["Revenue_direct"] + (1 - selected["blend_weight"]) * frame["Revenue_component"]) * selected["blend_scale"]

    full_history, sample, full_frame = fit_full_outputs(mode)
    submissions = generate_candidates(full_history, sample, full_frame, selected)

    order = [
        "submission_fast_blended_best.csv",
        "submission_fast_recent_level_best.csv",
        "submission_fast_scaled_best.csv",
        "submission_fast_trend_best.csv",
        "submission_fast_best_cogs_cap_110.csv",
        "submission_fast_current.csv",
    ]
    rank_table = pd.DataFrame({"file": order, "trial_rank": range(1, len(order) + 1)})
    submissions = rank_table.merge(submissions, on="file", how="left")

    report = build_report(
        mode,
        cache_used,
        folds,
        baseline,
        bias,
        buckets,
        scale_summary,
        recent_summary,
        trend_summary,
        ensemble_summary,
        combo_summary,
        submissions,
        selected,
    )
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)
    print(f"\nSaved report to {REPORT_FILE.name}", flush=True)


if __name__ == "__main__":
    main()
