import argparse
import itertools
import time
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
CACHE_DIR = PROJECT_ROOT / "artifacts" / "cache" / "diagnostic_cache_full"
REPORT_PATH = OUTPUT_DIR / "calibration_diagnostic_report_full.md"
CUTOFF_DATE = pd.Timestamp("2022-12-31")
TEST_START = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2024-07-01")

FOLD_SPECS = {
    "A": (pd.Timestamp("2018-12-31"), pd.Timestamp("2019-01-01"), pd.Timestamp("2020-07-01")),
    "B": (pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-01"), pd.Timestamp("2021-07-01")),
    "C": (pd.Timestamp("2020-12-31"), pd.Timestamp("2021-01-01"), pd.Timestamp("2022-07-01")),
    "D": (pd.Timestamp("2021-12-31"), pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
}
RECENT_FOLDS = {"C", "D"}

SOURCE_TABLES = {
    "sales.csv",
    "orders.csv",
    "order_items.csv",
    "products.csv",
    "promotions.csv",
    "inventory.csv",
    "web_traffic.csv",
    "returns.csv",
    "reviews.csv",
    "payments.csv",
    "shipments.csv",
    "geography.csv",
    "customers.csv",
}

REVENUE_SCALES = [0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 1.00, 1.025, 1.05]
YEAR1_SCALES = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05]
YEAR2_SCALES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
W_VALUES = [round(x, 1) for x in np.linspace(0, 1, 11)]
RECENT_WINDOWS = [14, 30, 60, 90]
GROWTH_RATES = [-0.25, -0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15]
ALPHA_VALUES = W_VALUES
COGS_SCALES = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
COGS_CAPS = [None, 1.05, 1.10, 1.15, 1.20]

CURRENT_REVENUE_WEIGHT = 1.0
CURRENT_COGS_WEIGHT = 0.6

REQUIRED_CACHE_COLUMNS = [
    "fold",
    "Date",
    "horizon_day",
    "actual_Revenue",
    "actual_COGS",
    "pred_Revenue_direct",
    "pred_Revenue_component",
    "pred_Revenue_current",
    "pred_COGS_direct",
    "pred_COGS_component",
    "pred_COGS_current",
    "pred_order_count",
    "pred_AOV",
    "pred_cogs_ratio",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Full resumable Revenue/COGS diagnostic workflow.")
    parser.add_argument("--refresh-cache", action="store_true", help="Recompute requested fold caches.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed fold caches and compute missing folds. This is the default.")
    parser.add_argument("--folds", default="A,B,C,D", help="Comma-separated fold names, e.g. C,D.")
    parser.add_argument("--skip-cogs-grid", action="store_true", help="Skip COGS scale/cap grid.")
    parser.add_argument("--full", action="store_true", help="Run the complete workflow. Present for explicitness; default also runs complete workflow.")
    return parser.parse_args()


def smape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.abs(actual) + np.abs(pred)
    denom = np.where(denom == 0, np.nan, denom)
    return float(np.nanmean(2 * np.abs(actual - pred) / denom) * 100)


def metric_dict(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": mean_squared_error(actual, pred) ** 0.5,
        "R2": r2_score(actual, pred),
        "MAPE": mean_absolute_percentage_error(actual, pred) * 100,
        "sMAPE": smape(actual, pred),
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


def selected_folds(raw):
    names = [x.strip().upper() for x in raw.split(",") if x.strip()]
    unknown = sorted(set(names).difference(FOLD_SPECS))
    if unknown:
        raise ValueError(f"Unknown fold names: {unknown}")
    return names


def detect_date_columns(path):
    sample = pd.read_csv(path, nrows=200, low_memory=False)
    cols = [c for c in sample.columns if "date" in c.lower()]
    found = []
    for col in cols:
        parsed = pd.to_datetime(sample[col], errors="coerce")
        if parsed.notna().any():
            found.append(col)
    return found


def inspect_data_availability():
    rows = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        date_cols = detect_date_columns(path)
        row_count = sum(len(chunk) for chunk in pd.read_csv(path, chunksize=200000, low_memory=False))
        min_dates, max_dates = [], []
        future_records = 0
        for col in date_cols:
            dates = pd.to_datetime(pd.read_csv(path, usecols=[col], low_memory=False)[col], errors="coerce")
            if dates.notna().any():
                min_dates.append(dates.min())
                max_dates.append(dates.max())
                future_records += int((dates > CUTOFF_DATE).sum())
        rows.append(
            {
                "file": path.name,
                "source_table": path.name in SOURCE_TABLES,
                "rows": row_count,
                "date_columns": ", ".join(date_cols) if date_cols else "Not found",
                "min_date": min(min_dates).date().isoformat() if min_dates else "Not found",
                "max_date": max(max_dates).date().isoformat() if max_dates else "Not found",
                "records_after_2022_12_31": future_records,
            }
        )
    return pd.DataFrame(rows)


def cache_path(fold_name):
    return CACHE_DIR / f"fold_{fold_name}_full_raw_predictions.csv"


def validate_cache(path, fold_name):
    if not path.exists():
        return False, "missing"
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
    except Exception as exc:
        return False, f"read failed: {exc}"
    missing = sorted(set(REQUIRED_CACHE_COLUMNS).difference(df.columns))
    if missing:
        return False, f"missing columns: {missing}"
    _, val_start, val_end = FOLD_SPECS[fold_name]
    expected_dates = pd.date_range(val_start, val_end, freq="D")
    if len(df) != len(expected_dates):
        return False, f"row count {len(df)} != expected {len(expected_dates)}"
    if df["Date"].min() != val_start or df["Date"].max() != val_end:
        return False, "date range mismatch"
    if not pd.Index(df["Date"]).equals(pd.Index(expected_dates)):
        return False, "date sequence mismatch"
    pred_cols = [c for c in REQUIRED_CACHE_COLUMNS if c.startswith("pred_")]
    if df[pred_cols].isna().any().any():
        return False, "NaN in prediction columns"
    if df[["actual_Revenue", "actual_COGS"]].isna().any().any():
        return False, "NaN in actual columns"
    return True, "ok"


def forecast_with_details(system, dates, revenue_weight, cogs_weight):
    state = system.history.set_index("Date").sort_index().copy()
    rows = []
    for date in pd.Series(pd.to_datetime(dates)):
        pred = {}
        for model_name in ["Revenue", "COGS", "order_count", "AOV", "cogs_ratio"]:
            x = om.make_single_feature_row(date, state, system.history["Date"].min(), system.reference_cache, model_name)
            pred[model_name] = float(system.models[model_name].predict(x)[0])
        revenue_component = pred["order_count"] * pred["AOV"]
        revenue = revenue_weight * pred["Revenue"] + (1 - revenue_weight) * revenue_component
        cogs_component = revenue * pred["cogs_ratio"]
        cogs = cogs_weight * pred["COGS"] + (1 - cogs_weight) * cogs_component
        aov = revenue / pred["order_count"] if pred["order_count"] > 1e-9 else pred["AOV"]
        cogs_ratio = cogs / revenue if revenue > 1e-9 else pred["cogs_ratio"]
        state.loc[date, ["Revenue", "COGS", "order_count", "AOV", "cogs_ratio"]] = [
            revenue,
            cogs,
            pred["order_count"],
            aov,
            cogs_ratio,
        ]
        rows.append(
            {
                "Date": date,
                "Revenue": round(revenue, 2),
                "COGS": round(cogs, 2),
                "pred_order_count": pred["order_count"],
                "pred_AOV": pred["AOV"],
                "pred_cogs_ratio": pred["cogs_ratio"],
            }
        )
    return pd.DataFrame(rows)


def build_fold_cache(history, fold_name, refresh):
    path = cache_path(fold_name)
    ok, reason = validate_cache(path, fold_name)
    if ok and not refresh:
        print(f"Loaded Fold {fold_name} from cache: {path.relative_to(PROJECT_ROOT)}", flush=True)
        return pd.read_csv(path, parse_dates=["Date"]), "loaded"
    if path.exists() and not refresh:
        print(f"Recomputing Fold {fold_name}; cache invalid: {reason}", flush=True)

    train_end, val_start, val_end = FOLD_SPECS[fold_name]
    print(f"Training Fold {fold_name}: train <= {train_end.date()}, validate {val_start.date()} to {val_end.date()}", flush=True)
    train = history[history["Date"] <= train_end].copy()
    actual = history[(history["Date"] >= val_start) & (history["Date"] <= val_end)].copy()
    system = om.fit_base_system(train)
    direct = forecast_with_details(system, actual["Date"], 1.0, 1.0)
    component = forecast_with_details(system, actual["Date"], 0.0, 0.0)
    current = forecast_with_details(system, actual["Date"], CURRENT_REVENUE_WEIGHT, CURRENT_COGS_WEIGHT)

    out = pd.DataFrame(
        {
            "fold": fold_name,
            "Date": actual["Date"].to_numpy(),
            "horizon_day": np.arange(1, len(actual) + 1),
            "actual_Revenue": actual["Revenue"].to_numpy(),
            "actual_COGS": actual["COGS"].to_numpy(),
            "pred_Revenue_direct": direct["Revenue"].to_numpy(),
            "pred_Revenue_component": component["Revenue"].to_numpy(),
            "pred_Revenue_current": current["Revenue"].to_numpy(),
            "pred_COGS_direct": direct["COGS"].to_numpy(),
            "pred_COGS_component": component["COGS"].to_numpy(),
            "pred_COGS_current": current["COGS"].to_numpy(),
            "pred_order_count": current["pred_order_count"].to_numpy(),
            "pred_AOV": current["pred_AOV"].to_numpy(),
            "pred_cogs_ratio": current["pred_cogs_ratio"].to_numpy(),
            "train_end": train_end.date().isoformat(),
            "validation_start": val_start.date().isoformat(),
            "validation_end": val_end.date().isoformat(),
        }
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    ok, reason = validate_cache(path, fold_name)
    if not ok:
        raise ValueError(f"New cache for Fold {fold_name} failed validation: {reason}")
    print(f"Saved Fold {fold_name} cache: {path.relative_to(PROJECT_ROOT)}", flush=True)
    return out, "recomputed"


def load_or_build_caches(history, folds, refresh):
    frames, status = [], {}
    for fold_name in folds:
        df, state = build_fold_cache(history, fold_name, refresh)
        frames.append(df)
        status[fold_name] = state
    return pd.concat(frames, ignore_index=True), status


def add_bucket(df):
    out = df.copy()
    out["horizon_bucket"] = out["horizon_day"].map(horizon_bucket)
    out["month"] = out["Date"].dt.to_period("M").astype(str)
    return out


def fold_metric_rows(df, revenue_col="pred_Revenue_current", cogs_col="pred_COGS_current", label="current"):
    rows = []
    for fold, part in df.groupby("fold", sort=False):
        for target, pred_col in [("Revenue", revenue_col), ("COGS", cogs_col)]:
            rows.append({"strategy": label, "fold": fold, "target": target, **metric_dict(part[f"actual_{target}"], part[pred_col])})
    return pd.DataFrame(rows)


def aggregate_metrics(metric_rows):
    all_avg = metric_rows.groupby(["strategy", "target"])[["MAE", "RMSE", "R2", "MAPE", "sMAPE"]].mean().reset_index()
    recent = metric_rows[metric_rows["fold"].isin(RECENT_FOLDS)]
    recent_avg = recent.groupby(["strategy", "target"])[["MAE", "RMSE", "R2", "MAPE", "sMAPE"]].mean().reset_index()
    return all_avg, recent_avg


def baseline_diagnostics(df):
    df = add_bucket(df)
    metrics = fold_metric_rows(df)
    all_avg, recent_avg = aggregate_metrics(metrics)
    bucket_rows = []
    for (fold, bucket), part in df.groupby(["fold", "horizon_bucket"], sort=False):
        bucket_rows.append(
            {
                "fold": fold,
                "horizon_bucket": bucket,
                "Revenue_MAE": mean_absolute_error(part["actual_Revenue"], part["pred_Revenue_current"]),
                "COGS_MAE": mean_absolute_error(part["actual_COGS"], part["pred_COGS_current"]),
                "rows": len(part),
            }
        )
    bias_rows = []
    for fold, part in df.groupby("fold", sort=False):
        for target in ["Revenue", "COGS"]:
            actual = part[f"actual_{target}"]
            pred = part[f"pred_{target}_current"]
            bias_rows.append(
                {
                    "fold": fold,
                    "target": target,
                    "actual_mean": actual.mean(),
                    "prediction_mean": pred.mean(),
                    "actual_median": actual.median(),
                    "prediction_median": pred.median(),
                    "bias": (pred - actual).mean(),
                    "ratio": pred.mean() / actual.mean(),
                    "actual_std": actual.std(),
                    "prediction_std": pred.std(),
                    "volatility_ratio": pred.std() / actual.std(),
                }
            )
    monthly_rows = []
    for (fold, month), part in df.groupby(["fold", "month"], sort=False):
        monthly_rows.append(
            {
                "fold": fold,
                "month": month,
                "Revenue_MAE": mean_absolute_error(part["actual_Revenue"], part["pred_Revenue_current"]),
                "COGS_MAE": mean_absolute_error(part["actual_COGS"], part["pred_COGS_current"]),
            }
        )
    return metrics, all_avg, recent_avg, pd.DataFrame(bucket_rows), pd.DataFrame(bias_rows), pd.DataFrame(monthly_rows)


def eval_revenue_strategy(df, strategy_name, pred, params):
    rows = []
    temp = df[["fold", "actual_Revenue"]].copy()
    temp["pred"] = np.asarray(pred, dtype=float)
    for fold, part in temp.groupby("fold", sort=False):
        m = metric_dict(part["actual_Revenue"], part["pred"])
        m.update(
            {
                "bias": float((part["pred"] - part["actual_Revenue"]).mean()),
                "ratio": float(part["pred"].mean() / part["actual_Revenue"].mean()),
            }
        )
        rows.append({"strategy": strategy_name, "fold": fold, **params, **m})
    return rows


def summarize_revenue(rows, keys):
    details = pd.DataFrame(rows)
    all_summary = (
        details.groupby(keys)
        .agg(
            all_MAE=("MAE", "mean"),
            all_RMSE=("RMSE", "mean"),
            all_R2=("R2", "mean"),
            all_std_MAE=("MAE", "std"),
            all_bias=("bias", "mean"),
            all_ratio=("ratio", "mean"),
        )
        .reset_index()
    )
    recent = details[details["fold"].isin(RECENT_FOLDS)]
    recent_summary = (
        recent.groupby(keys)
        .agg(
            recent_MAE=("MAE", "mean"),
            recent_RMSE=("RMSE", "mean"),
            recent_R2=("R2", "mean"),
            recent_std_MAE=("MAE", "std"),
            recent_bias=("bias", "mean"),
            recent_ratio=("ratio", "mean"),
        )
        .reset_index()
    )
    summary = all_summary.merge(recent_summary, on=keys, how="left").sort_values(["all_MAE", "all_std_MAE"], na_position="last")
    return details, summary


def select_best(summary, metric_col, stability_col=None):
    min_val = summary[metric_col].min()
    near = summary[summary[metric_col] <= min_val * 1.01].copy()
    sort_cols = [stability_col, metric_col] if stability_col and stability_col in near.columns else [metric_col]
    return near.sort_values(sort_cols, na_position="last").iloc[0]


def revenue_global_scale(df):
    rows = []
    for scale in REVENUE_SCALES:
        rows.extend(eval_revenue_strategy(df, "global_scale", df["pred_Revenue_current"] * scale, {"scale": scale}))
    return summarize_revenue(rows, ["scale"])


def revenue_horizon_scale(df):
    rows = []
    for y1, y2 in itertools.product(YEAR1_SCALES, YEAR2_SCALES):
        factor = np.where(df["horizon_day"] <= 365, y1, y2)
        rows.extend(eval_revenue_strategy(df, "horizon_scale", df["pred_Revenue_current"] * factor, {"year1_scale": y1, "year2_scale": y2}))
    return summarize_revenue(rows, ["year1_scale", "year2_scale"])


def revenue_ensemble(df):
    rows = []
    for w in W_VALUES:
        pred = w * df["pred_Revenue_direct"] + (1 - w) * df["pred_Revenue_component"]
        rows.extend(eval_revenue_strategy(df, "ensemble", pred, {"w": w}))
    return summarize_revenue(rows, ["w"])


def revenue_ensemble_scale(df):
    rows = []
    for w, scale in itertools.product(W_VALUES, REVENUE_SCALES):
        pred = (w * df["pred_Revenue_direct"] + (1 - w) * df["pred_Revenue_component"]) * scale
        rows.extend(eval_revenue_strategy(df, "ensemble_scale", pred, {"w": w, "scale": scale}))
    return summarize_revenue(rows, ["w", "scale"])


def recent_level(df, history):
    rows = []
    for fold, part in df.groupby("fold", sort=False):
        train_end = FOLD_SPECS[fold][0]
        train_hist = history[history["Date"] <= train_end]
        first30 = part.head(30)["pred_Revenue_current"].mean()
        for window in RECENT_WINDOWS:
            recent_avg = train_hist.tail(window)["Revenue"].mean()
            multiplier = recent_avg / first30 if first30 > 0 else 1.0
            rows.extend(eval_revenue_strategy(part, "recent_level", part["pred_Revenue_current"] * multiplier, {"window": window, "multiplier": multiplier}))
    return summarize_revenue(rows, ["window"])


def annual_growth(df):
    rows = []
    for rate in GROWTH_RATES:
        multiplier = np.power(1 + rate, (df["horizon_day"] - 1) / 365.0)
        rows.extend(eval_revenue_strategy(df, "annual_growth", df["pred_Revenue_current"] * multiplier, {"growth_rate": rate}))
    return summarize_revenue(rows, ["growth_rate"])


def seasonal_naive_for_fold(part, history):
    fold = part["fold"].iloc[0]
    train_end = FOLD_SPECS[fold][0]
    train_hist = history[history["Date"] <= train_end].set_index("Date")["Revenue"].astype(float)
    monthly = train_hist.groupby(train_hist.index.month).mean()
    state = train_hist.copy()
    preds = []
    for date in part["Date"]:
        vals = [state.get(date - pd.Timedelta(days=lag), np.nan) for lag in [364, 365, 366]]
        if np.isfinite(vals).any():
            val = float(np.nanmean(vals))
        else:
            val = float(monthly.get(date.month, train_hist.tail(30).mean()))
        val = max(val, 0.0)
        state.loc[date] = val
        preds.append(val)
    return np.asarray(preds)


def seasonal_blend(df, history):
    rows = []
    seasonal = []
    for _, part in df.groupby("fold", sort=False):
        s = seasonal_naive_for_fold(part, history)
        seasonal.append(pd.Series(s, index=part.index))
    seasonal = pd.concat(seasonal).sort_index()
    for alpha in ALPHA_VALUES:
        pred = alpha * df["pred_Revenue_current"] + (1 - alpha) * seasonal
        rows.extend(eval_revenue_strategy(df, "seasonal_blend", pred, {"alpha": alpha}))
    return summarize_revenue(rows, ["alpha"])


def cogs_experiments(df, revenue_pred_for_caps, skip=False):
    if skip:
        empty = pd.DataFrame()
        return empty, empty
    rows = []
    for scale in COGS_SCALES:
        scaled = df["pred_COGS_current"] * scale
        for cap in COGS_CAPS:
            if cap is None:
                pred = scaled
                cap_name = "no_cap"
            else:
                pred = np.minimum(scaled, revenue_pred_for_caps * cap)
                cap_name = f"cap_{cap:.2f}"
            temp = df[["fold", "actual_COGS"]].copy()
            temp["pred"] = np.asarray(pred, dtype=float)
            temp["Revenue_for_cap"] = np.asarray(revenue_pred_for_caps, dtype=float)
            for fold, part in temp.groupby("fold", sort=False):
                m = metric_dict(part["actual_COGS"], part["pred"])
                ratio = np.divide(part["pred"], part["Revenue_for_cap"], out=np.full(len(part), np.nan), where=part["Revenue_for_cap"] > 0)
                rows.append(
                    {
                        "fold": fold,
                        "scale": scale,
                        "cap": cap_name,
                        **m,
                        "cogs_gt_revenue_days": int((part["pred"] > part["Revenue_for_cap"]).sum()),
                        "max_cogs_revenue_ratio": float(np.nanmax(ratio)),
                        "avg_cogs_revenue_ratio": float(np.nanmean(ratio)),
                    }
                )
    details = pd.DataFrame(rows)
    all_summary = details.groupby(["scale", "cap"]).agg(
        all_COGS_MAE=("MAE", "mean"),
        all_COGS_RMSE=("RMSE", "mean"),
        all_COGS_R2=("R2", "mean"),
        all_cogs_gt_revenue_days=("cogs_gt_revenue_days", "mean"),
        all_max_cogs_revenue_ratio=("max_cogs_revenue_ratio", "max"),
        all_avg_cogs_revenue_ratio=("avg_cogs_revenue_ratio", "mean"),
    ).reset_index()
    recent = details[details["fold"].isin(RECENT_FOLDS)]
    recent_summary = recent.groupby(["scale", "cap"]).agg(
        recent_COGS_MAE=("MAE", "mean"),
        recent_COGS_RMSE=("RMSE", "mean"),
        recent_COGS_R2=("R2", "mean"),
        recent_cogs_gt_revenue_days=("cogs_gt_revenue_days", "mean"),
        recent_max_cogs_revenue_ratio=("max_cogs_revenue_ratio", "max"),
        recent_avg_cogs_revenue_ratio=("avg_cogs_revenue_ratio", "mean"),
    ).reset_index()
    summary = all_summary.merge(recent_summary, on=["scale", "cap"], how="left").sort_values(["all_COGS_MAE"])
    return details, summary


def apply_revenue_strategy(df, strategy):
    kind = strategy["kind"]
    if kind == "current":
        return df["pred_Revenue_current"].to_numpy()
    if kind == "scale":
        return (df["pred_Revenue_current"] * strategy["scale"]).to_numpy()
    if kind == "horizon":
        factor = np.where(df["horizon_day"] <= 365, strategy["year1_scale"], strategy["year2_scale"])
        return (df["pred_Revenue_current"] * factor).to_numpy()
    if kind == "ensemble":
        return (strategy["w"] * df["pred_Revenue_direct"] + (1 - strategy["w"]) * df["pred_Revenue_component"]).to_numpy()
    if kind == "ensemble_scale":
        return ((strategy["w"] * df["pred_Revenue_direct"] + (1 - strategy["w"]) * df["pred_Revenue_component"]) * strategy["scale"]).to_numpy()
    if kind == "recent_level":
        preds = pd.Series(index=df.index, dtype=float)
        history = om.load_history()
        for fold, part in df.groupby("fold", sort=False):
            train_end = FOLD_SPECS[fold][0]
            train_hist = history[history["Date"] <= train_end]
            first30 = part.head(30)["pred_Revenue_current"].mean()
            multiplier = train_hist.tail(strategy["window"])["Revenue"].mean() / first30 if first30 > 0 else 1.0
            preds.loc[part.index] = part["pred_Revenue_current"] * multiplier
        return preds.sort_index().to_numpy()
    if kind == "growth":
        mult = np.power(1 + strategy["growth_rate"], (df["horizon_day"] - 1) / 365.0)
        return (df["pred_Revenue_current"] * mult).to_numpy()
    if kind == "seasonal_blend":
        history = om.load_history()
        seasonal = []
        for _, part in df.groupby("fold", sort=False):
            seasonal.append(pd.Series(seasonal_naive_for_fold(part, history), index=part.index))
        seasonal = pd.concat(seasonal).sort_index()
        return (strategy["alpha"] * df["pred_Revenue_current"] + (1 - strategy["alpha"]) * seasonal).to_numpy()
    raise ValueError(strategy)


def candidate_comparison(df, revenue_strategies, cogs_strategies):
    rows = []
    for r_name, r_strategy in revenue_strategies.items():
        revenue_pred = apply_revenue_strategy(df, r_strategy)
        for c_name, c_strategy in cogs_strategies.items():
            cogs_pred = df["pred_COGS_current"].to_numpy() * c_strategy.get("scale", 1.0)
            cap = c_strategy.get("cap")
            if cap is not None:
                cogs_pred = np.minimum(cogs_pred, revenue_pred * cap)
            rev_details = []
            cogs_details = []
            for fold, part in df.groupby("fold", sort=False):
                idx = part.index
                rev_details.append({"fold": fold, **metric_dict(part["actual_Revenue"], revenue_pred[idx])})
                cogs_details.append({"fold": fold, **metric_dict(part["actual_COGS"], cogs_pred[idx])})
            rev = pd.DataFrame(rev_details)
            cg = pd.DataFrame(cogs_details)
            rows.append(
                {
                    "revenue_strategy": r_name,
                    "cogs_strategy": c_name,
                    "Revenue_MAE_all": rev["MAE"].mean(),
                    "Revenue_RMSE_all": rev["RMSE"].mean(),
                    "Revenue_R2_all": rev["R2"].mean(),
                    "Revenue_MAE_recent": rev[rev["fold"].isin(RECENT_FOLDS)]["MAE"].mean(),
                    "Revenue_RMSE_recent": rev[rev["fold"].isin(RECENT_FOLDS)]["RMSE"].mean(),
                    "Revenue_R2_recent": rev[rev["fold"].isin(RECENT_FOLDS)]["R2"].mean(),
                    "COGS_MAE_all": cg["MAE"].mean(),
                    "COGS_RMSE_all": cg["RMSE"].mean(),
                    "COGS_R2_all": cg["R2"].mean(),
                    "COGS_MAE_recent": cg[cg["fold"].isin(RECENT_FOLDS)]["MAE"].mean(),
                    "COGS_RMSE_recent": cg[cg["fold"].isin(RECENT_FOLDS)]["RMSE"].mean(),
                    "COGS_R2_recent": cg[cg["fold"].isin(RECENT_FOLDS)]["R2"].mean(),
                }
            )
    return pd.DataFrame(rows).sort_values(["Revenue_MAE_all", "Revenue_MAE_recent"])


def train_full_predictions():
    history = om.load_history()
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])
    print("Training full-history production model for final candidates", flush=True)
    system = om.fit_base_system(history)
    direct = forecast_with_details(system, sample["Date"], 1.0, 1.0)
    component = forecast_with_details(system, sample["Date"], 0.0, 0.0)
    current = forecast_with_details(system, sample["Date"], CURRENT_REVENUE_WEIGHT, CURRENT_COGS_WEIGHT)
    df = pd.DataFrame(
        {
            "Date": sample["Date"],
            "horizon_day": np.arange(1, len(sample) + 1),
            "pred_Revenue_direct": direct["Revenue"],
            "pred_Revenue_component": component["Revenue"],
            "pred_Revenue_current": current["Revenue"],
            "pred_COGS_current": current["COGS"],
        }
    )
    return history, sample, df


def apply_full_revenue(full_df, history, strategy):
    kind = strategy["kind"]
    if kind == "current":
        return full_df["pred_Revenue_current"].to_numpy()
    if kind == "scale":
        return (full_df["pred_Revenue_current"] * strategy["scale"]).to_numpy()
    if kind == "horizon":
        factor = np.where(full_df["horizon_day"] <= 365, strategy["year1_scale"], strategy["year2_scale"])
        return (full_df["pred_Revenue_current"] * factor).to_numpy()
    if kind == "ensemble":
        return (strategy["w"] * full_df["pred_Revenue_direct"] + (1 - strategy["w"]) * full_df["pred_Revenue_component"]).to_numpy()
    if kind == "ensemble_scale":
        return ((strategy["w"] * full_df["pred_Revenue_direct"] + (1 - strategy["w"]) * full_df["pred_Revenue_component"]) * strategy["scale"]).to_numpy()
    if kind == "recent_level":
        first30 = full_df.head(30)["pred_Revenue_current"].mean()
        multiplier = history.tail(strategy["window"])["Revenue"].mean() / first30 if first30 > 0 else 1.0
        return (full_df["pred_Revenue_current"] * multiplier).to_numpy()
    if kind == "growth":
        mult = np.power(1 + strategy["growth_rate"], (full_df["horizon_day"] - 1) / 365.0)
        return (full_df["pred_Revenue_current"] * mult).to_numpy()
    if kind == "seasonal_blend":
        state = history.set_index("Date")["Revenue"].astype(float).copy()
        monthly = state.groupby(state.index.month).mean()
        seasonal = []
        for date in full_df["Date"]:
            vals = [state.get(date - pd.Timedelta(days=lag), np.nan) for lag in [364, 365, 366]]
            val = float(np.nanmean(vals)) if np.isfinite(vals).any() else float(monthly.get(date.month, state.tail(30).mean()))
            state.loc[date] = max(val, 0.0)
            seasonal.append(max(val, 0.0))
        return (strategy["alpha"] * full_df["pred_Revenue_current"] + (1 - strategy["alpha"]) * np.asarray(seasonal)).to_numpy()
    raise ValueError(strategy)


def write_submission(sample, revenue, cogs, filename):
    out = sample[["Date"]].copy()
    out["Revenue"] = np.maximum(np.asarray(revenue, dtype=float), 0.0).round(2)
    out["COGS"] = np.maximum(np.asarray(cogs, dtype=float), 0.0).round(2)
    if out[["Revenue", "COGS"]].isna().any().any():
        raise ValueError(f"{filename} contains NaN.")
    if not out["Date"].equals(sample["Date"]):
        raise ValueError(f"{filename} date order mismatch.")
    write = out.copy()
    write["Date"] = write["Date"].dt.strftime("%Y-%m-%d")
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBMISSION_DIR / filename
    write.to_csv(path, index=False)
    ratio = np.divide(out["COGS"], out["Revenue"], out=np.full(len(out), np.nan), where=out["Revenue"] > 0)
    row = {
        "file": filename,
        "rows": len(out),
        "date_min": write["Date"].min(),
        "date_max": write["Date"].max(),
        "Revenue_min": out["Revenue"].min(),
        "Revenue_mean": out["Revenue"].mean(),
        "Revenue_median": out["Revenue"].median(),
        "Revenue_max": out["Revenue"].max(),
        "Revenue_std": out["Revenue"].std(),
        "COGS_min": out["COGS"].min(),
        "COGS_mean": out["COGS"].mean(),
        "COGS_median": out["COGS"].median(),
        "COGS_max": out["COGS"].max(),
        "COGS_std": out["COGS"].std(),
        "nan_count": int(out[["Revenue", "COGS"]].isna().sum().sum()),
        "negative_count": int((out[["Revenue", "COGS"]] < 0).sum().sum()),
        "COGS_gt_Revenue": int((out["COGS"] > out["Revenue"]).sum()),
        "max_COGS_Revenue_ratio": float(np.nanmax(ratio)),
    }
    print(f"Saved {filename}", flush=True)
    return row


def generate_submissions(sample, full_history, full_df, revenue_strategies, cogs_strategy):
    files = {
        "current": "submission_full_current_reproduced.csv",
        "best_global_scale_allfold": "submission_full_revenue_scaled_allfold_best.csv",
        "best_global_scale_recentfold": "submission_full_revenue_scaled_recentfold_best.csv",
        "best_horizon_scale_allfold": "submission_full_revenue_horizon_scaled_allfold_best.csv",
        "best_horizon_scale_recentfold": "submission_full_revenue_horizon_scaled_recentfold_best.csv",
        "best_ensemble": "submission_full_revenue_ensemble_best.csv",
        "best_ensemble_scale_allfold": "submission_full_revenue_ensemble_scaled_allfold_best.csv",
        "best_ensemble_scale_recentfold": "submission_full_revenue_ensemble_scaled_recentfold_best.csv",
        "best_recent_level": "submission_full_revenue_recent_level_best.csv",
        "best_annual_growth": "submission_full_revenue_growth_best.csv",
        "best_seasonal_blend": "submission_full_revenue_seasonal_blend_best.csv",
    }
    rows = []
    for name, filename in files.items():
        rev = apply_full_revenue(full_df, full_history, revenue_strategies[name])
        cogs = full_df["pred_COGS_current"].to_numpy()
        rows.append(write_submission(sample, rev, cogs, filename))
    best_rev = apply_full_revenue(full_df, full_history, revenue_strategies["best_combined"])
    cogs = full_df["pred_COGS_current"].to_numpy() * cogs_strategy.get("scale", 1.0)
    if cogs_strategy.get("cap") is not None:
        cogs = np.minimum(cogs, best_rev * cogs_strategy["cap"])
    rows.append(write_submission(sample, best_rev, cogs, "submission_full_best_combined_revenue_cogs.csv"))
    return pd.DataFrame(rows)


def fmt_table(df, max_rows=20):
    if df is None or df.empty:
        return "Not computed"
    shown = df.head(max_rows).copy()
    for col in shown.select_dtypes(include=[np.number]).columns:
        shown[col] = shown[col].map(lambda x: f"{x:,.4f}" if pd.notna(x) else "")
    return shown.to_markdown(index=False)


def build_report(context):
    data_check = context["data_check"]
    source_future = data_check[(data_check["source_table"]) & (data_check["records_after_2022_12_31"] > 0)]
    fold_status = pd.DataFrame([{"fold": k, "cache_status": v} for k, v in context["cache_status"].items()])
    lines = [
        "# Full Calibration Diagnostic Report",
        "",
        "## A. Runtime and cache status",
        f"- Script start: `{context['start_time']}`",
        f"- Script end: `{context['end_time']}`",
        f"- Cache directory: `{CACHE_DIR.name}`",
        f"- Folds requested: `{', '.join(context['folds'])}`",
        f"- Total runtime seconds: `{context['runtime_seconds']:.1f}`",
        "",
        fmt_table(fold_status),
        "",
        "## B. Data availability check",
        fmt_table(data_check[data_check["source_table"]], max_rows=30),
        "",
        f"Source data files with records after 2022-12-31: `{len(source_future)}`. Generated submissions/sample files may contain future dates but are not used as future covariates.",
        "",
        "## C. Baseline rolling-origin backtest",
        "Fold-level metrics:",
        fmt_table(context["baseline_metrics"].sort_values(["fold", "target"]), max_rows=30),
        "",
        "All-fold averages:",
        fmt_table(context["baseline_all_avg"], max_rows=10),
        "",
        "Recent Fold C/D averages:",
        fmt_table(context["baseline_recent_avg"], max_rows=10),
        "",
        "Horizon bucket MAE:",
        fmt_table(context["bucket_mae"], max_rows=40),
        "",
        "## D. Bias and recursive drift",
        fmt_table(context["bias"], max_rows=20),
        "",
        "Monthly error sample:",
        fmt_table(context["monthly_mae"], max_rows=24),
        "",
        "Fold A is treated as a possible older-regime outlier if its bias and MAE are much larger than Fold C/D. Compare the all-fold and recent-fold columns below before choosing a candidate.",
        "",
        "## E. Revenue global scaling",
        fmt_table(context["scale_summary"], max_rows=20),
        "",
        f"Best all-fold scale: `{context['best_scale_all']['scale']}`. Best recent-fold scale: `{context['best_scale_recent']['scale']}`. Most stable scale: `{context['stable_scale']['scale']}`.",
        "",
        "## F. Revenue horizon-specific scaling",
        fmt_table(context["horizon_summary"], max_rows=20),
        "",
        f"Best all-fold horizon config: year1=`{context['best_horizon_all']['year1_scale']}`, year2=`{context['best_horizon_all']['year2_scale']}`. Best recent-fold config: year1=`{context['best_horizon_recent']['year1_scale']}`, year2=`{context['best_horizon_recent']['year2_scale']}`.",
        "",
        "## G. Direct/component Revenue ensemble",
        "Ensemble weights:",
        fmt_table(context["ensemble_summary"], max_rows=15),
        "",
        "Top ensemble + scale by all-fold MAE:",
        fmt_table(context["ensemble_scale_summary"].sort_values("all_MAE"), max_rows=20),
        "",
        "Top ensemble + scale by recent-fold MAE:",
        fmt_table(context["ensemble_scale_summary"].sort_values("recent_MAE"), max_rows=20),
        "",
        "## H. Trend and level adjustments",
        "Recent-level results:",
        fmt_table(context["recent_summary"], max_rows=20),
        "",
        "Annual growth/decline results:",
        fmt_table(context["growth_summary"], max_rows=20),
        "",
        "Seasonal naive blend results:",
        fmt_table(context["seasonal_summary"], max_rows=20),
        "",
        "## I. COGS calibration and constraints",
        fmt_table(context["cogs_summary"], max_rows=25),
        "",
        "COGS constraints are evaluated against the selected Revenue candidate for cap ratios. Hidden metrics may penalize aggressive Revenue-only tuning if COGS degrades.",
        "",
        "## J. Combined strategy comparison",
        "Rank by Revenue MAE:",
        fmt_table(context["candidate_comparison"].sort_values("Revenue_MAE_all"), max_rows=25),
        "",
        "Balanced Revenue+COGS ranking:",
        fmt_table(context["candidate_comparison"].assign(balance_score=context["candidate_comparison"]["Revenue_MAE_all"] + context["candidate_comparison"]["COGS_MAE_all"]).sort_values("balance_score"), max_rows=25),
        "",
        "## K. Generated submission files",
        fmt_table(context["submission_summary"], max_rows=30),
        "",
        "## L. Recommended Kaggle submission order",
        fmt_table(context["recommended"], max_rows=12),
        "",
        "## M. Warnings",
        "- Do not overfit to the public leaderboard; public Revenue MAE may be only a subset of the hidden scoring behavior.",
        "- Fold A may represent an older regime. Recent Fold C/D recommendations can be more relevant to the 2023-2024 horizon but less robust across history.",
        "- Hidden scoring may include COGS, RMSE, or R2, so Revenue-only scaling can be risky.",
        "- Avoid using too many Kaggle submissions as a manual tuning loop.",
        "- No future covariates are used. Do not add future promotions, traffic, inventory, returns, reviews, orders, or shipments.",
    ]
    return "\n".join(lines)


def main():
    start = time.time()
    start_label = pd.Timestamp.now().isoformat(timespec="seconds")
    args = parse_args()
    folds = selected_folds(args.folds)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Inspecting data availability", flush=True)
    data_check = inspect_data_availability()

    print("Loading history", flush=True)
    history = om.load_history()

    print("Loading/recomputing fold caches", flush=True)
    fold_df, cache_status = load_or_build_caches(history, folds, args.refresh_cache)
    fold_df = add_bucket(fold_df)

    print("Running baseline diagnostics", flush=True)
    baseline_metrics_df, baseline_all_avg, baseline_recent_avg, bucket_mae, bias, monthly_mae = baseline_diagnostics(fold_df)

    print("Running Revenue global scaling", flush=True)
    _, scale_summary = revenue_global_scale(fold_df)
    best_scale_all = select_best(scale_summary, "all_MAE")
    best_scale_recent = select_best(scale_summary, "recent_MAE")
    stable_scale = select_best(scale_summary, "all_MAE", "all_std_MAE")

    print("Running Revenue horizon scaling", flush=True)
    _, horizon_summary = revenue_horizon_scale(fold_df)
    best_horizon_all = select_best(horizon_summary, "all_MAE")
    best_horizon_recent = select_best(horizon_summary, "recent_MAE")

    print("Running Revenue direct/component ensemble", flush=True)
    _, ensemble_summary = revenue_ensemble(fold_df)
    best_ensemble_all = select_best(ensemble_summary, "all_MAE")
    best_ensemble_recent = select_best(ensemble_summary, "recent_MAE")

    print("Running Revenue ensemble + scale", flush=True)
    _, ensemble_scale_summary = revenue_ensemble_scale(fold_df)
    best_ensemble_scale_all = select_best(ensemble_scale_summary, "all_MAE")
    best_ensemble_scale_recent = select_best(ensemble_scale_summary, "recent_MAE")

    print("Running trend and level adjustments", flush=True)
    _, recent_summary = recent_level(fold_df, history)
    _, growth_summary = annual_growth(fold_df)
    _, seasonal_summary = seasonal_blend(fold_df, history)
    best_recent = select_best(recent_summary, "all_MAE")
    best_growth = select_best(growth_summary, "all_MAE")
    best_seasonal = select_best(seasonal_summary, "all_MAE")

    revenue_strategies = {
        "current": {"kind": "current"},
        "best_global_scale_allfold": {"kind": "scale", "scale": float(best_scale_all["scale"])},
        "best_global_scale_recentfold": {"kind": "scale", "scale": float(best_scale_recent["scale"])},
        "best_horizon_scale_allfold": {"kind": "horizon", "year1_scale": float(best_horizon_all["year1_scale"]), "year2_scale": float(best_horizon_all["year2_scale"])},
        "best_horizon_scale_recentfold": {"kind": "horizon", "year1_scale": float(best_horizon_recent["year1_scale"]), "year2_scale": float(best_horizon_recent["year2_scale"])},
        "best_ensemble": {"kind": "ensemble", "w": float(best_ensemble_all["w"])},
        "best_ensemble_scale_allfold": {"kind": "ensemble_scale", "w": float(best_ensemble_scale_all["w"]), "scale": float(best_ensemble_scale_all["scale"])},
        "best_ensemble_scale_recentfold": {"kind": "ensemble_scale", "w": float(best_ensemble_scale_recent["w"]), "scale": float(best_ensemble_scale_recent["scale"])},
        "best_recent_level": {"kind": "recent_level", "window": int(best_recent["window"])},
        "best_annual_growth": {"kind": "growth", "growth_rate": float(best_growth["growth_rate"])},
        "best_seasonal_blend": {"kind": "seasonal_blend", "alpha": float(best_seasonal["alpha"])},
    }

    cap_revenue_strategy = revenue_strategies["best_ensemble_scale_recentfold"]
    cap_revenue_pred = apply_revenue_strategy(fold_df, cap_revenue_strategy)

    print("Running COGS calibration/constraints", flush=True)
    _, cogs_summary = cogs_experiments(fold_df, cap_revenue_pred, skip=args.skip_cogs_grid)
    if cogs_summary.empty:
        best_cogs = {"scale": 1.0, "cap": None}
        cogs_strategies = {"current": {"scale": 1.0, "cap": None}}
    else:
        best_cogs_row = select_best(cogs_summary, "all_COGS_MAE")
        cap_value = None if best_cogs_row["cap"] == "no_cap" else float(str(best_cogs_row["cap"]).replace("cap_", ""))
        best_cogs = {"scale": float(best_cogs_row["scale"]), "cap": cap_value}
        cogs_strategies = {
            "current": {"scale": 1.0, "cap": None},
            "best_cogs_scale_cap": best_cogs,
        }

    print("Running combined strategy comparison", flush=True)
    candidate_table = candidate_comparison(fold_df, revenue_strategies, cogs_strategies)
    revenue_strategies["best_combined"] = revenue_strategies[candidate_table.sort_values("Revenue_MAE_recent").iloc[0]["revenue_strategy"]]

    print("Generating final full-history submissions", flush=True)
    full_history, sample, full_df = train_full_predictions()
    submission_summary = generate_submissions(sample, full_history, full_df, revenue_strategies, best_cogs)

    recommended = pd.DataFrame(
        [
            {"rank": 1, "purpose": "safest all-fold generalization", "file": "submission_full_revenue_ensemble_scaled_allfold_best.csv"},
            {"rank": 2, "purpose": "best recent-fold Revenue MAE", "file": "submission_full_revenue_ensemble_scaled_recentfold_best.csv"},
            {"rank": 3, "purpose": "public-score-oriented Revenue candidate", "file": "submission_full_revenue_scaled_recentfold_best.csv"},
            {"rank": 4, "purpose": "balanced Revenue+COGS candidate", "file": "submission_full_best_combined_revenue_cogs.csv"},
            {"rank": 5, "purpose": "exploratory/riskier candidate", "file": "submission_full_revenue_recent_level_best.csv"},
        ]
    )

    context = {
        "start_time": start_label,
        "end_time": pd.Timestamp.now().isoformat(timespec="seconds"),
        "runtime_seconds": time.time() - start,
        "folds": folds,
        "cache_status": cache_status,
        "data_check": data_check,
        "baseline_metrics": baseline_metrics_df,
        "baseline_all_avg": baseline_all_avg,
        "baseline_recent_avg": baseline_recent_avg,
        "bucket_mae": bucket_mae,
        "bias": bias,
        "monthly_mae": monthly_mae,
        "scale_summary": scale_summary,
        "best_scale_all": best_scale_all,
        "best_scale_recent": best_scale_recent,
        "stable_scale": stable_scale,
        "horizon_summary": horizon_summary,
        "best_horizon_all": best_horizon_all,
        "best_horizon_recent": best_horizon_recent,
        "ensemble_summary": ensemble_summary,
        "ensemble_scale_summary": ensemble_scale_summary,
        "recent_summary": recent_summary,
        "growth_summary": growth_summary,
        "seasonal_summary": seasonal_summary,
        "cogs_summary": cogs_summary,
        "candidate_comparison": candidate_table,
        "submission_summary": submission_summary,
        "recommended": recommended,
    }
    report = build_report(context)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved report to {REPORT_PATH.name}", flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main()
