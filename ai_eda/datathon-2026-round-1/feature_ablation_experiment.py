import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import our_method_forecast as om

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent
CACHE_DIR = DATA_DIR / "diagnostic_cache_feature_ablation"
REPORT_PATH = DATA_DIR / "feature_ablation_report.md"

FOLD_SPECS = {
    "C": (pd.Timestamp("2020-12-31"), pd.Timestamp("2021-01-01"), pd.Timestamp("2022-07-01")),
    "D": (pd.Timestamp("2021-12-31"), pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
}
MODEL_NAMES = ["Revenue", "COGS", "order_count", "AOV", "cogs_ratio"]
TARGET_MAP = {
    "Revenue": "Revenue",
    "COGS": "COGS",
    "order_count": "order_count",
    "AOV": "AOV",
    "cogs_ratio": "cogs_ratio",
}
TARGETS = ["Revenue", "COGS"]
CURRENT_REVENUE_WEIGHT = 1.0
CURRENT_COGS_WEIGHT = 0.6


@dataclass
class AblationModelSpec:
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
class AblationSystem:
    history: pd.DataFrame
    reference_cache: dict
    models: dict[str, AblationModelSpec]
    feature_map: dict[str, list[str]]


def parse_args():
    parser = argparse.ArgumentParser(description="Feature ablation experiment for our_method_forecast.py")
    parser.add_argument("--folds", default="C,D", help="Comma-separated folds. Default: C,D")
    parser.add_argument("--resume", action="store_true", default=True, help="Reuse cached predictions. Default behavior.")
    parser.add_argument("--refresh-cache", action="store_true", help="Recompute predictions and reports.")
    parser.add_argument("--fast", action="store_true", help="Use fewer LightGBM estimators for a quicker directional run.")
    return parser.parse_args()


def selected_folds(raw):
    folds = [x.strip().upper() for x in raw.split(",") if x.strip()]
    unknown = sorted(set(folds).difference(FOLD_SPECS))
    if unknown:
        raise ValueError(f"Unknown fold names: {unknown}. Available: {sorted(FOLD_SPECS)}")
    return folds


def make_model(fast=False):
    if not fast:
        return om.make_model()
    return LGBMRegressor(
        objective="regression",
        random_state=om.RANDOM_STATE,
        n_estimators=250,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=25,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.4,
        verbosity=-1,
    )


def fit_model_with_features(history, model_name, target_col, reference_cache, features, fast=False):
    x = om.make_feature_matrix(
        history["Date"], history, history["Date"].min(), reference_cache, model_name
    )
    missing = sorted(set(features).difference(x.columns))
    if missing:
        raise ValueError(f"Missing features for {model_name}: {missing}")
    y = np.log1p(history[target_col].clip(lower=0))
    model = make_model(fast=fast)
    model.fit(x[features], y)
    return AblationModelSpec(model_name, model, list(features), log_target=True)


def fit_system(history, feature_map, fast=False):
    history = om.add_derived_columns(history).sort_values("Date").reset_index(drop=True)
    reference_cache = om.make_reference_cache(history)
    models = {
        name: fit_model_with_features(
            history,
            name,
            TARGET_MAP[name],
            reference_cache,
            feature_map[name],
            fast=fast,
        )
        for name in MODEL_NAMES
    }
    return AblationSystem(history, reference_cache, models, feature_map)


def forecast_with_details(system, dates, revenue_weight=CURRENT_REVENUE_WEIGHT, cogs_weight=CURRENT_COGS_WEIGHT):
    state = system.history.set_index("Date").sort_index().copy()
    rows = []
    for i, date in enumerate(pd.Series(pd.to_datetime(dates)), start=1):
        pred = {}
        for model_name in MODEL_NAMES:
            x = om.make_single_feature_row(
                date, state, system.history["Date"].min(), system.reference_cache, model_name
            )
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
                "horizon_day": i,
                "Revenue": round(max(revenue, 0.0), 2),
                "COGS": round(max(cogs, 0.0), 2),
                "pred_order_count": pred["order_count"],
                "pred_AOV": pred["AOV"],
                "pred_cogs_ratio": pred["cogs_ratio"],
            }
        )
    return pd.DataFrame(rows)


def feature_importance_table(system, experiment, fold):
    rows = []
    for model_name, spec in system.models.items():
        imp = pd.Series(spec.model.feature_importances_, index=spec.feature_names)
        total = float(imp.sum())
        for rank, (feature, importance) in enumerate(imp.sort_values(ascending=False).items(), start=1):
            rows.append(
                {
                    "experiment": experiment,
                    "fold": fold,
                    "model": model_name,
                    "rank": rank,
                    "feature": feature,
                    "importance": float(importance),
                    "importance_share": float(importance / total) if total > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def preserve_allowlist_order(model_name, selected):
    selected = set(selected)
    ordered = [f for f in om.FEATURE_ALLOWLIST[model_name] if f in selected]
    if not ordered:
        ordered = [om.FEATURE_ALLOWLIST[model_name][0]]
    return ordered


def top_n_feature_map(baseline_importance, n):
    feature_map = {}
    for model_name in MODEL_NAMES:
        part = baseline_importance[baseline_importance["model"] == model_name].sort_values(
            ["importance", "feature"], ascending=[False, True]
        )
        feature_map[model_name] = preserve_allowlist_order(model_name, part.head(n)["feature"])
    return feature_map


def low_importance_removed_map(baseline_importance):
    feature_map = {}
    for model_name in MODEL_NAMES:
        part = baseline_importance[baseline_importance["model"] == model_name].sort_values(
            "importance", ascending=False
        )
        kept = part[part["importance_share"] >= 0.075]["feature"].tolist()
        if len(kept) < 8:
            kept = part.head(8)["feature"].tolist()
        if len(kept) == len(om.FEATURE_ALLOWLIST[model_name]):
            kept = part.head(len(kept) - 1)["feature"].tolist()
        feature_map[model_name] = preserve_allowlist_order(model_name, kept)
    return feature_map


def redundant_removed_map(baseline_importance):
    feature_map = {}
    for model_name in MODEL_NAMES:
        features = list(om.FEATURE_ALLOWLIST[model_name])
        part = baseline_importance[baseline_importance["model"] == model_name].set_index("feature")["importance"]
        remove = set()
        prefixes = sorted({f.rsplit("_lag_", 1)[0] for f in features if "_lag_" in f})
        for prefix in prefixes:
            yearly = [f"{prefix}_lag_{lag}" for lag in [364, 365, 366] if f"{prefix}_lag_{lag}" in features]
            if len(yearly) > 1:
                keep = max(yearly, key=lambda f: float(part.get(f, 0.0)))
                remove.update(f for f in yearly if f != keep)
            yoy = f"{prefix}_yoy_lag_mean"
            if yoy in features and yearly:
                best_yearly_importance = max(float(part.get(f, 0.0)) for f in yearly)
                if float(part.get(yoy, 0.0)) < best_yearly_importance:
                    remove.add(yoy)
        for calendar_feature in ["time_index", "year_cos_3"]:
            if calendar_feature in features:
                importances = part.reindex(features).fillna(0.0)
                if float(part.get(calendar_feature, 0.0)) <= float(importances.quantile(0.25)):
                    remove.add(calendar_feature)
        kept = [f for f in features if f not in remove]
        if len(kept) < 6:
            kept = [f for f in features if f in set(part.sort_values(ascending=False).head(6).index)]
        feature_map[model_name] = kept
    return feature_map


def hand_picked_importance_map():
    return {
        "Revenue": list(om.FEATURE_ALLOWLIST["Revenue"]),
        "COGS": [
            "COGS_seasonal_base",
            "COGS_lag_1",
            "COGS_lag_364",
            "COGS_lag_7",
            "COGS_lag_365",
            "COGS_lag_90",
            "COGS_lag_730",
            "COGS_rolling_std_7",
            "COGS_lag_30",
            "COGS_lag_366",
            "time_index",
        ],
        "order_count": [
            "order_count_seasonal_base",
            "order_count_lag_1",
            "order_count_lag_7",
            "order_count_lag_365",
            "order_count_lag_364",
            "order_count_lag_730",
            "order_count_lag_366",
            "order_count_lag_90",
            "year_cos_3",
            "order_count_lag_30",
            "order_count_rolling_std_7",
        ],
        "AOV": [
            "AOV_seasonal_base",
            "AOV_lag_1",
            "AOV_lag_30",
            "AOV_lag_730",
            "AOV_lag_90",
            "AOV_lag_366",
            "AOV_rolling_std_7",
            "AOV_lag_365",
            "AOV_lag_364",
            "AOV_lag_14",
            "AOV_lag_7",
        ],
        "cogs_ratio": [
            "cogs_ratio_seasonal_base",
            "cogs_ratio_lag_1",
            "cogs_ratio_lag_730",
            "cogs_ratio_rolling_std_7",
            "cogs_ratio_lag_364",
            "cogs_ratio_lag_30",
            "cogs_ratio_lag_90",
            "cogs_ratio_lag_365",
            "cogs_ratio_rolling_mean_7",
            "cogs_ratio_lag_366",
            "time_index",
        ],
    }


def experiment_feature_maps(baseline_importance):
    return {
        "baseline_current": {k: list(v) for k, v in om.FEATURE_ALLOWLIST.items()},
        "hand_picked_importance": hand_picked_importance_map(),
        "top_10": top_n_feature_map(baseline_importance, 10),
        "top_8": top_n_feature_map(baseline_importance, 8),
        "top_6": top_n_feature_map(baseline_importance, 6),
        "remove_low_importance": low_importance_removed_map(baseline_importance),
        "remove_redundant_obvious": redundant_removed_map(baseline_importance),
    }


def cache_path(experiment, fold, fast):
    suffix = "fast" if fast else "full"
    return CACHE_DIR / f"{experiment}_fold_{fold}_{suffix}_predictions.csv"


def metadata_path(experiment, fold, fast):
    suffix = "fast" if fast else "full"
    return CACHE_DIR / f"{experiment}_fold_{fold}_{suffix}_features.json"


def validate_cache(path, fold):
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
    except Exception:
        return False
    _, val_start, val_end = FOLD_SPECS[fold]
    expected = pd.date_range(val_start, val_end, freq="D")
    required = {"experiment", "fold", "Date", "horizon_day", "actual_Revenue", "actual_COGS", "pred_Revenue", "pred_COGS"}
    return (
        required.issubset(df.columns)
        and len(df) == len(expected)
        and pd.Index(df["Date"]).equals(pd.Index(expected))
        and not df[["pred_Revenue", "pred_COGS"]].isna().any().any()
    )


def build_or_load_predictions(history, fold, experiment, feature_map, refresh=False, fast=False):
    path = cache_path(experiment, fold, fast)
    if validate_cache(path, fold) and not refresh:
        return pd.read_csv(path, parse_dates=["Date"]), "loaded"

    train_end, val_start, val_end = FOLD_SPECS[fold]
    train = history[history["Date"] <= train_end].copy()
    actual = history[(history["Date"] >= val_start) & (history["Date"] <= val_end)].copy()
    print(f"Training {experiment} fold {fold}: train <= {train_end.date()}, features {[len(feature_map[m]) for m in MODEL_NAMES]}", flush=True)
    system = fit_system(train, feature_map, fast=fast)
    pred = forecast_with_details(system, actual["Date"])
    out = pd.DataFrame(
        {
            "experiment": experiment,
            "fold": fold,
            "Date": actual["Date"].to_numpy(),
            "horizon_day": np.arange(1, len(actual) + 1),
            "actual_Revenue": actual["Revenue"].to_numpy(),
            "actual_COGS": actual["COGS"].to_numpy(),
            "pred_Revenue": pred["Revenue"].to_numpy(),
            "pred_COGS": pred["COGS"].to_numpy(),
            "pred_order_count": pred["pred_order_count"].to_numpy(),
            "pred_AOV": pred["pred_AOV"].to_numpy(),
            "pred_cogs_ratio": pred["pred_cogs_ratio"].to_numpy(),
        }
    )
    CACHE_DIR.mkdir(exist_ok=True)
    out.to_csv(path, index=False)
    metadata_path(experiment, fold, fast).write_text(json.dumps(feature_map, indent=2), encoding="utf-8")
    return out, "recomputed"


def metric_dict(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": mean_squared_error(actual, predicted) ** 0.5,
        "R2": r2_score(actual, predicted),
    }


def compute_metrics(predictions):
    rows = []
    for (experiment, fold), part in predictions.groupby(["experiment", "fold"], sort=False):
        for target in TARGETS:
            rows.append(
                {
                    "experiment": experiment,
                    "fold": fold,
                    "target": target,
                    **metric_dict(part[f"actual_{target}"], part[f"pred_{target}"]),
                }
            )
    fold_metrics = pd.DataFrame(rows)
    summary = fold_metrics.groupby(["experiment", "target"]).agg(
        MAE=("MAE", "mean"),
        RMSE=("RMSE", "mean"),
        R2=("R2", "mean"),
    ).reset_index()
    baseline = summary[summary["experiment"] == "baseline_current"].set_index("target")
    comparison_rows = []
    for _, row in summary.iterrows():
        base_mae = float(baseline.loc[row["target"], "MAE"])
        delta = float(row["MAE"] - base_mae)
        pct = delta / base_mae if base_mae else np.nan
        if row["experiment"] == "baseline_current":
            status = "baseline"
        elif pct <= -0.01:
            status = "improves"
        elif pct <= 0.01:
            status = "stays similar"
        else:
            status = "gets worse"
        comparison_rows.append(
            {
                **row.to_dict(),
                "MAE_delta_vs_baseline": delta,
                "MAE_pct_delta_vs_baseline": pct,
                "status_vs_baseline": status,
            }
        )
    return fold_metrics, pd.DataFrame(comparison_rows)


def aggregate_importance(importance):
    if importance.empty:
        return importance
    return (
        importance[importance["experiment"] == "baseline_current"]
        .groupby(["model", "feature"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            mean_importance_share=("importance_share", "mean"),
            folds_seen=("fold", "nunique"),
        )
        .sort_values(["model", "mean_importance"], ascending=[True, False])
    )


def features_used_table(feature_map):
    rows = []
    for model_name in MODEL_NAMES:
        for rank, feature in enumerate(feature_map[model_name], start=1):
            rows.append({"model": model_name, "position": rank, "feature": feature})
    return pd.DataFrame(rows)


def feature_counts_table(feature_maps_by_fold):
    rows = []
    for fold, maps in feature_maps_by_fold.items():
        for experiment, fmap in maps.items():
            row = {"fold": fold, "experiment": experiment}
            for model_name in MODEL_NAMES:
                row[f"{model_name}_feature_count"] = len(fmap[model_name])
            rows.append(row)
    return pd.DataFrame(rows)


def probably_removed_features(feature_maps_by_fold):
    rows = []
    baseline = set()
    for features in om.FEATURE_ALLOWLIST.values():
        baseline.update(features)
    for experiment in [
        "hand_picked_importance",
        "top_10",
        "top_8",
        "top_6",
        "remove_low_importance",
        "remove_redundant_obvious",
    ]:
        for model_name in MODEL_NAMES:
            removed_sets = []
            for maps in feature_maps_by_fold.values():
                removed_sets.append(set(om.FEATURE_ALLOWLIST[model_name]) - set(maps[experiment][model_name]))
            common_removed = sorted(set.intersection(*removed_sets)) if removed_sets else []
            for feature in common_removed:
                rows.append({"experiment": experiment, "model": model_name, "feature_removed_in_all_folds": feature})
    return pd.DataFrame(rows)


def fmt_table(df, max_rows=40):
    if df is None or df.empty:
        return "Not available"
    shown = df.head(max_rows).copy()
    for col in shown.select_dtypes(include=[np.number]).columns:
        shown[col] = shown[col].map(lambda x: f"{x:,.4f}" if pd.notna(x) else "")
    return shown.to_markdown(index=False)


def choose_recommendation(comparison):
    pivot = comparison.pivot(index="experiment", columns="target", values="MAE")
    baseline_rev = float(pivot.loc["baseline_current", "Revenue"])
    baseline_cogs = float(pivot.loc["baseline_current", "COGS"])
    candidates = []
    for experiment, row in pivot.iterrows():
        rev_pct = (float(row["Revenue"]) - baseline_rev) / baseline_rev
        cogs_pct = (float(row["COGS"]) - baseline_cogs) / baseline_cogs
        total = float(row["Revenue"]) + float(row["COGS"])
        max_regression = max(rev_pct, cogs_pct)
        candidates.append(
            {
                "experiment": experiment,
                "Revenue_MAE_pct_delta": rev_pct,
                "COGS_MAE_pct_delta": cogs_pct,
                "combined_MAE": total,
                "max_target_regression": max_regression,
            }
        )
    candidates = pd.DataFrame(candidates)
    acceptable = candidates[candidates["max_target_regression"] <= 0.01].copy()
    if acceptable.empty:
        return "baseline_current", candidates
    rank = acceptable.sort_values(["combined_MAE", "max_target_regression"]).iloc[0]
    return str(rank["experiment"]), candidates.sort_values(["combined_MAE", "max_target_regression"])


def build_report(context):
    recommended = context["recommended"]
    warning = ""
    if recommended == "baseline_current":
        warning = "Reducing features did not produce a clearly safer validation result; keep the current selected features."
    else:
        warning = f"{recommended} stayed within the 1% MAE tolerance on both targets and had the best combined MAE among acceptable reduced sets. Re-test before changing production because recursive errors can compound across the 2023-2024 horizon."
    lines = [
        "# Feature Ablation Report",
        "",
        "## A. Features Actually Used",
        fmt_table(context["features_used"], max_rows=80),
        "",
        "## B. Top LightGBM Feature Importance",
        "Averaged across the baseline models trained on Fold C and Fold D training windows.",
        fmt_table(context["top_importance"], max_rows=80),
        "",
        "## C. Ablation Feature Counts",
        fmt_table(context["feature_counts"], max_rows=30),
        "",
        "## D. Validation Metrics",
        "Fold-level metrics:",
        fmt_table(context["fold_metrics"].sort_values(["experiment", "fold", "target"]), max_rows=80),
        "",
        "Average across Fold C and Fold D:",
        fmt_table(context["comparison"].sort_values(["target", "MAE"]), max_rows=40),
        "",
        "## E. Recommended Feature Set",
        f"Recommended set: `{recommended}`.",
        "",
        warning,
        "",
        "## F. Features That Can Probably Be Removed",
        "These are features removed in all Fold C/D versions of a reduced experiment. Treat them as candidates, not automatic production deletions.",
        fmt_table(context["removed_features"], max_rows=80),
        "",
        "The `hand_picked_importance` set is intentionally conservative: it keeps all Revenue features because top-N pruning worsened recursive Revenue MAE, and removes only the consistently weaker auxiliary features `COGS_lag_14`, `order_count_lag_14`, `AOV_rolling_mean_7`, and `cogs_ratio_yoy_lag_mean`.",
        "",
        "## G. Recursive Forecasting Warning",
        "Feature reductions are evaluated recursively: Revenue, COGS, order_count, AOV, and cogs_ratio predictions are fed back into later validation days. If a reduced feature set looks fine on direct training fit but worsens Fold C/D recursive MAE or R2, it should not replace the current allowlist.",
        "",
        "## H. Runtime",
        f"- Started: `{context['start_time']}`",
        f"- Finished: `{context['end_time']}`",
        f"- Runtime seconds: `{context['runtime_seconds']:.1f}`",
        f"- Folds: `{', '.join(context['folds'])}`",
        f"- Model mode: `{'fast' if context['fast'] else 'full'}`",
        f"- Cache directory: `{CACHE_DIR.name}`",
        "",
        "Cache status:",
        fmt_table(context["cache_status"], max_rows=40),
    ]
    return "\n".join(lines)


def main():
    start = time.time()
    start_label = pd.Timestamp.now().isoformat(timespec="seconds")
    args = parse_args()
    folds = selected_folds(args.folds)
    CACHE_DIR.mkdir(exist_ok=True)

    print("Loading history", flush=True)
    history = om.load_history()

    all_predictions = []
    all_importance = []
    cache_rows = []
    feature_maps_by_fold = {}

    for fold in folds:
        train_end, _, _ = FOLD_SPECS[fold]
        train = history[history["Date"] <= train_end].copy()
        print(f"Fitting baseline for fold {fold} to derive importances", flush=True)
        baseline_system = fit_system(train, {k: list(v) for k, v in om.FEATURE_ALLOWLIST.items()}, fast=args.fast)
        baseline_importance = feature_importance_table(baseline_system, "baseline_current", fold)
        all_importance.append(baseline_importance)
        feature_maps = experiment_feature_maps(baseline_importance)
        feature_maps_by_fold[fold] = feature_maps

        for experiment, feature_map in feature_maps.items():
            df, status = build_or_load_predictions(
                history,
                fold,
                experiment,
                feature_map,
                refresh=args.refresh_cache,
                fast=args.fast,
            )
            all_predictions.append(df)
            cache_rows.append({"fold": fold, "experiment": experiment, "cache_status": status})

    predictions = pd.concat(all_predictions, ignore_index=True)
    importance = pd.concat(all_importance, ignore_index=True)
    fold_metrics, comparison = compute_metrics(predictions)
    top_importance = (
        aggregate_importance(importance)
        .groupby("model", group_keys=False)
        .head(12)
        .reset_index(drop=True)
    )
    recommended, recommendation_candidates = choose_recommendation(comparison)

    context = {
        "features_used": features_used_table({k: list(v) for k, v in om.FEATURE_ALLOWLIST.items()}),
        "top_importance": top_importance,
        "feature_counts": feature_counts_table(feature_maps_by_fold),
        "removed_features": probably_removed_features(feature_maps_by_fold),
        "fold_metrics": fold_metrics,
        "comparison": comparison,
        "recommendation_candidates": recommendation_candidates,
        "cache_status": pd.DataFrame(cache_rows),
        "recommended": recommended,
        "start_time": start_label,
        "end_time": pd.Timestamp.now().isoformat(timespec="seconds"),
        "runtime_seconds": time.time() - start,
        "folds": folds,
        "fast": args.fast,
    }

    predictions.to_csv(CACHE_DIR / "all_predictions.csv", index=False)
    importance.to_csv(CACHE_DIR / "baseline_feature_importance_by_fold.csv", index=False)
    context["top_importance"].to_csv(CACHE_DIR / "baseline_feature_importance_summary.csv", index=False)
    context["fold_metrics"].to_csv(CACHE_DIR / "fold_metrics.csv", index=False)
    context["comparison"].to_csv(CACHE_DIR / "metric_comparison.csv", index=False)
    context["feature_counts"].to_csv(CACHE_DIR / "feature_counts.csv", index=False)
    context["removed_features"].to_csv(CACHE_DIR / "removed_feature_candidates.csv", index=False)
    REPORT_PATH.write_text(build_report(context), encoding="utf-8")

    print(f"Saved report to {REPORT_PATH.name}", flush=True)
    print(fmt_table(context["comparison"].sort_values(["target", "MAE"]), max_rows=40), flush=True)


if __name__ == "__main__":
    main()
