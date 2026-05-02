import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_FILE = PROJECT_ROOT / "outputs" / "model" / "submissions" / "submission_our_method.csv"
RANDOM_STATE = 42
RECOVERY_CARRY_FORWARD = 1.00
GROWTH_CLIP_LOWER = 0.95
GROWTH_CLIP_UPPER = 1.25
RECOVERY_DRIVER_BOOST = {
    "Revenue": 1.00,
    "COGS": 1.00,
    "order_count": 1.25,
    "AOV": 1.35,
}

TARGETS = ["Revenue", "COGS"]
MODEL_SERIES = ["Revenue", "COGS", "order_count", "AOV", "cogs_ratio"]
LAGS = [1, 7, 14, 30, 90, 364, 365, 366, 730]
ROLL_WINDOWS = [7, 30, 90, 365]

FEATURE_GROUPS = {
    "Revenue": ["Revenue", "order_count", "AOV"],
    "COGS": ["COGS", "cogs_ratio"],
    "order_count": ["order_count"],
    "AOV": ["AOV"],
    "cogs_ratio": ["cogs_ratio"],
}

# Pruned feature sets based on the previous full LightGBM importance run.
# This keeps the hybrid architecture explainable while removing weak candidate
# features that added complexity without materially contributing.
FEATURE_ALLOWLIST = {
    "Revenue": [
        "Revenue_seasonal_base",
        "AOV_lag_1",
        "Revenue_lag_1",
        "Revenue_lag_7",
        "order_count_lag_90",
        "order_count_lag_7",
        "order_count_rolling_std_7",
        "AOV_lag_364",
        "Revenue_lag_365",
        "order_count_seasonal_base",
        "AOV_lag_730",
        "Revenue_lag_364",
    ],
    "COGS": [
        "COGS_seasonal_base",
        "COGS_lag_1",
        "COGS_lag_364",
        "COGS_lag_7",
        "COGS_lag_365",
        "COGS_lag_90",
        "COGS_lag_730",
        "COGS_lag_14",
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
        "order_count_lag_14",
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
        "AOV_rolling_mean_7",
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
        "cogs_ratio_yoy_lag_mean",
        "cogs_ratio_lag_366",
        "time_index",
    ],
}


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.abs(actual) + np.abs(predicted)
    denominator = np.where(denominator == 0, np.nan, denominator)
    return float(np.nanmean(2 * np.abs(actual - predicted) / denominator) * 100)


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=np.abs(denominator) > 1e-9,
    )


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["AOV"] = safe_divide(out["Revenue"], out["order_count"])
    out["cogs_ratio"] = safe_divide(out["COGS"], out["Revenue"])
    out[["AOV", "cogs_ratio"]] = out[["AOV", "cogs_ratio"]].replace(
        [np.inf, -np.inf], np.nan
    )
    out[["AOV", "cogs_ratio"]] = out[["AOV", "cogs_ratio"]].ffill().bfill()
    return out


def load_history() -> pd.DataFrame:
    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"])
    orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["order_date"])

    daily_sales = (
        sales[["Date", "Revenue", "COGS"]]
        .dropna(subset=["Date"])
        .groupby("Date", as_index=False)[["Revenue", "COGS"]]
        .sum()
        .sort_values("Date")
    )
    order_count = (
        orders.dropna(subset=["order_date"])
        .groupby("order_date")["order_id"]
        .nunique()
        .rename("order_count")
        .reset_index()
        .rename(columns={"order_date": "Date"})
    )

    full_dates = pd.DataFrame(
        {"Date": pd.date_range(daily_sales["Date"].min(), daily_sales["Date"].max(), freq="D")}
    )
    history = full_dates.merge(daily_sales, on="Date", how="left").merge(
        order_count, on="Date", how="left"
    )
    history[["Revenue", "COGS", "order_count"]] = history[
        ["Revenue", "COGS", "order_count"]
    ].fillna(0.0)
    return add_derived_columns(history)


def annual_growth(history: pd.DataFrame, series_name: str) -> float:
    annual = history.groupby(history["Date"].dt.year)[series_name].mean().replace(0, np.nan)
    annual = annual.dropna()
    yoy = annual.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if yoy.empty:
        return 1.0

    recent_yoy = float(yoy.iloc[-1])
    if series_name in {"Revenue", "COGS", "order_count", "AOV"} and recent_yoy > 0:
        driver_boost = RECOVERY_DRIVER_BOOST.get(series_name, 1.0)
        growth = 1 + RECOVERY_CARRY_FORWARD * driver_boost * recent_yoy
    else:
        stable_yoy = yoy.loc[yoy.index >= 2013]
        growth = float((1 + stable_yoy).prod() ** (1 / len(stable_yoy))) if len(stable_yoy) else 1.0

    if not np.isfinite(growth) or growth <= 0:
        return 1.0
    return float(np.clip(growth, GROWTH_CLIP_LOWER, GROWTH_CLIP_UPPER))


def make_reference_cache(history: pd.DataFrame) -> dict[str, dict[str, object]]:
    cache = {}
    for series_name in MODEL_SERIES:
        hist = history[["Date", series_name]].copy()
        hist["year"] = hist["Date"].dt.year
        hist["month"] = hist["Date"].dt.month
        hist["day"] = hist["Date"].dt.day
        hist["quarter"] = hist["Date"].dt.quarter
        hist["dow"] = hist["Date"].dt.dayofweek

        annual_mean = hist.groupby("year")[series_name].mean().replace(0, np.nan)
        latest_year = int(annual_mean.dropna().index.max())
        latest_annual_mean = float(annual_mean.loc[latest_year])

        hist = hist.merge(
            annual_mean.rename("annual_mean"),
            left_on="year",
            right_index=True,
            how="left",
        )
        hist["seasonal_norm"] = hist[series_name] / hist["annual_mean"].replace(0, np.nan)
        cache[series_name] = {
            "latest_year": latest_year,
            "latest_annual_mean": latest_annual_mean,
            "growth": annual_growth(history, series_name),
            "month_day_norm": hist.groupby(["month", "day"])["seasonal_norm"].mean(),
            "month_mean": hist.groupby("month")[series_name].mean(),
            "quarter_mean": hist.groupby("quarter")[series_name].mean(),
            "dow_mean": hist.groupby("dow")[series_name].mean(),
        }
    return cache


def make_calendar_features(dates: pd.Series, history_start: pd.Timestamp) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(dates), name="Date").reset_index(drop=True)
    x = pd.DataFrame(index=np.arange(len(dates)))
    x["time_index"] = (dates - history_start).dt.days.astype(float)
    x["year"] = dates.dt.year
    x["year_offset"] = dates.dt.year - history_start.year
    x["quarter"] = dates.dt.quarter
    x["month"] = dates.dt.month
    x["weekofyear"] = dates.dt.isocalendar().week.astype(int)
    x["dayofyear"] = dates.dt.dayofyear
    x["day_of_month"] = dates.dt.day
    x["day_of_week"] = dates.dt.dayofweek
    x["month_index"] = (dates.dt.year - history_start.year) * 12 + dates.dt.month
    x["is_weekend"] = (x["day_of_week"] >= 5).astype(int)
    x["is_month_start"] = dates.dt.is_month_start.astype(int)
    x["is_month_end"] = dates.dt.is_month_end.astype(int)
    x["is_peak_month_eda"] = dates.dt.month.isin([4, 5, 6]).astype(int)
    x["is_low_month_eda"] = dates.dt.month.isin([11, 12, 1]).astype(int)
    x["is_peak_quarter_eda"] = (dates.dt.quarter == 2).astype(int)
    x["is_low_quarter_eda"] = (dates.dt.quarter == 4).astype(int)

    x["month_sin"] = np.sin(2 * np.pi * x["month"] / 12)
    x["month_cos"] = np.cos(2 * np.pi * x["month"] / 12)
    x["quarter_sin"] = np.sin(2 * np.pi * x["quarter"] / 4)
    x["quarter_cos"] = np.cos(2 * np.pi * x["quarter"] / 4)
    x["dow_sin"] = np.sin(2 * np.pi * x["day_of_week"] / 7)
    x["dow_cos"] = np.cos(2 * np.pi * x["day_of_week"] / 7)
    for harmonic in range(1, 5):
        angle = 2 * np.pi * harmonic * x["dayofyear"] / 365.25
        x[f"year_sin_{harmonic}"] = np.sin(angle)
        x[f"year_cos_{harmonic}"] = np.cos(angle)
    return x


def add_reference_features(
    x: pd.DataFrame,
    dates: pd.Series,
    reference_cache: dict[str, dict[str, object]],
    series_names: list[str],
) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(dates), name="Date").reset_index(drop=True)
    out = x.copy()
    for series_name in series_names:
        ref = reference_cache[series_name]
        md = pd.MultiIndex.from_arrays([dates.dt.month, dates.dt.day])
        seasonal_norm = pd.Series(md.map(ref["month_day_norm"]), index=out.index)
        seasonal_norm = seasonal_norm.astype(float).fillna(1.0)
        years_ahead = dates.dt.year - int(ref["latest_year"])
        base = (
            float(ref["latest_annual_mean"])
            * np.power(float(ref["growth"]), years_ahead)
            * seasonal_norm
        )
        out[f"{series_name}_seasonal_base"] = base
        out[f"{series_name}_seasonal_base_log"] = np.log1p(np.clip(base, 0, None))
        out[f"{series_name}_month_avg"] = dates.dt.month.map(ref["month_mean"]).astype(float)
        out[f"{series_name}_quarter_avg"] = dates.dt.quarter.map(ref["quarter_mean"]).astype(float)
        out[f"{series_name}_dow_avg"] = dates.dt.dayofweek.map(ref["dow_mean"]).astype(float)
    return out


def add_vectorized_history_features(
    x: pd.DataFrame,
    dates: pd.Series,
    state: pd.DataFrame,
    series_names: list[str],
) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(dates), name="Date").reset_index(drop=True)
    out = x.copy()
    indexed = state.set_index("Date").sort_index()
    full_index = pd.date_range(
        min(indexed.index.min(), dates.min()), max(indexed.index.max(), dates.max()), freq="D"
    )
    for series_name in series_names:
        series = indexed[series_name].reindex(full_index).astype(float)
        for lag in LAGS:
            out[f"{series_name}_lag_{lag}"] = dates.map(series.shift(lag, freq="D")).astype(float)
        out[f"{series_name}_yoy_lag_mean"] = out[
            [f"{series_name}_lag_364", f"{series_name}_lag_365", f"{series_name}_lag_366"]
        ].mean(axis=1)
        shifted = series.shift(1)
        for window in ROLL_WINDOWS:
            min_periods = max(2, min(window // 3, 30))
            rolling = shifted.rolling(window, min_periods=min_periods)
            out[f"{series_name}_rolling_mean_{window}"] = dates.map(rolling.mean()).astype(float)
            out[f"{series_name}_rolling_std_{window}"] = dates.map(rolling.std()).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def make_feature_matrix(
    dates: pd.Series,
    state: pd.DataFrame,
    history_start: pd.Timestamp,
    reference_cache: dict[str, dict[str, object]],
    model_name: str,
) -> pd.DataFrame:
    series_names = FEATURE_GROUPS[model_name]
    x = make_calendar_features(dates, history_start)
    x = add_reference_features(x, dates, reference_cache, series_names)
    x = add_vectorized_history_features(x, dates, state, series_names)
    return x


def make_single_feature_row(
    date: pd.Timestamp,
    state_indexed: pd.DataFrame,
    history_start: pd.Timestamp,
    reference_cache: dict[str, dict[str, object]],
    model_name: str,
) -> pd.DataFrame:
    date = pd.Timestamp(date)
    series_names = FEATURE_GROUPS[model_name]
    x = make_calendar_features(pd.Series([date]), history_start)
    x = add_reference_features(x, pd.Series([date]), reference_cache, series_names)
    for series_name in series_names:
        series = state_indexed[series_name].astype(float)
        past = series.loc[: date - pd.Timedelta(days=1)]
        for lag in LAGS:
            x.loc[0, f"{series_name}_lag_{lag}"] = series.get(date - pd.Timedelta(days=lag), np.nan)
        x.loc[0, f"{series_name}_yoy_lag_mean"] = x.loc[
            0, [f"{series_name}_lag_364", f"{series_name}_lag_365", f"{series_name}_lag_366"]
        ].mean()
        for window in ROLL_WINDOWS:
            min_periods = max(2, min(window // 3, 30))
            values = past.tail(window).dropna()
            x.loc[0, f"{series_name}_rolling_mean_{window}"] = values.mean() if len(values) >= min_periods else np.nan
            x.loc[0, f"{series_name}_rolling_std_{window}"] = values.std() if len(values) >= min_periods else np.nan
    return x.replace([np.inf, -np.inf], np.nan)


def make_model() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        random_state=RANDOM_STATE,
        n_estimators=650,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=25,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.4,
        verbosity=-1,
    )


@dataclass
class ModelSpec:
    name: str
    model: LGBMRegressor
    feature_names: list[str]
    log_target: bool = True

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        pred = self.model.predict(x.reindex(columns=self.feature_names))
        if self.log_target:
            pred = np.expm1(pred)
        return np.maximum(pred, 0.0)

    def top_features(self, n: int = 15) -> pd.DataFrame:
        imp = pd.Series(self.model.feature_importances_, index=self.feature_names)
        return (
            imp.sort_values(ascending=False)
            .head(n)
            .rename("importance")
            .reset_index()
            .rename(columns={"index": "feature"})
            .assign(model=self.name)
        )


def fit_model(history: pd.DataFrame, model_name: str, target_col: str, reference_cache) -> ModelSpec:
    x = make_feature_matrix(
        history["Date"], history, history["Date"].min(), reference_cache, model_name
    )
    selected_features = FEATURE_ALLOWLIST[model_name]
    missing = sorted(set(selected_features).difference(x.columns))
    if missing:
        raise ValueError(f"Missing pruned features for {model_name}: {missing}")
    x = x[selected_features]
    y_raw = history[target_col].clip(lower=0)
    y = np.log1p(y_raw)
    model = make_model()
    model.fit(x, y)
    return ModelSpec(model_name, model, list(x.columns), log_target=True)


@dataclass
class ForecastSystem:
    history: pd.DataFrame
    reference_cache: dict[str, dict[str, object]]
    models: dict[str, ModelSpec]
    revenue_direct_weight: float
    cogs_direct_weight: float

    def forecast(self, forecast_dates: pd.Series) -> pd.DataFrame:
        forecast_dates = pd.Series(pd.to_datetime(forecast_dates), name="Date")
        state = self.history.set_index("Date").sort_index().copy()
        rows = []
        for date in forecast_dates:
            pred = {}
            for model_name in ["Revenue", "COGS", "order_count", "AOV", "cogs_ratio"]:
                x = make_single_feature_row(
                    date, state, self.history["Date"].min(), self.reference_cache, model_name
                )
                pred[model_name] = float(self.models[model_name].predict(x)[0])

            revenue_component = pred["order_count"] * pred["AOV"]
            revenue = (
                self.revenue_direct_weight * pred["Revenue"]
                + (1 - self.revenue_direct_weight) * revenue_component
            )
            cogs_component = revenue * pred["cogs_ratio"]
            cogs = (
                self.cogs_direct_weight * pred["COGS"]
                + (1 - self.cogs_direct_weight) * cogs_component
            )

            aov = revenue / pred["order_count"] if pred["order_count"] > 1e-9 else pred["AOV"]
            cogs_ratio = cogs / revenue if revenue > 1e-9 else pred["cogs_ratio"]
            state.loc[date, ["Revenue", "COGS", "order_count", "AOV", "cogs_ratio"]] = [
                revenue,
                cogs,
                pred["order_count"],
                aov,
                cogs_ratio,
            ]
            rows.append({"Date": date, "Revenue": revenue, "COGS": cogs})
        out = pd.DataFrame(rows)
        out["Revenue"] = out["Revenue"].round(2)
        out["COGS"] = out["COGS"].round(2)
        return out

    def feature_importance_table(self) -> pd.DataFrame:
        return pd.concat(
            [model.top_features(12) for model in self.models.values()], ignore_index=True
        )[["model", "feature", "importance"]]


def fit_base_system(history: pd.DataFrame) -> ForecastSystem:
    history = add_derived_columns(history).sort_values("Date").reset_index(drop=True)
    reference_cache = make_reference_cache(history)
    target_map = {
        "Revenue": "Revenue",
        "COGS": "COGS",
        "order_count": "order_count",
        "AOV": "AOV",
        "cogs_ratio": "cogs_ratio",
    }
    models = {
        name: fit_model(history, name, target, reference_cache)
        for name, target in target_map.items()
    }
    return ForecastSystem(history, reference_cache, models, 1.0, 1.0)


def metric_frame(actual: pd.DataFrame, predicted: pd.DataFrame, label: str) -> pd.DataFrame:
    merged = actual.merge(predicted, on="Date", suffixes=("_actual", "_pred"))
    rows = []
    for target in TARGETS:
        y_true = merged[f"{target}_actual"].to_numpy()
        y_pred = merged[f"{target}_pred"].to_numpy()
        rows.append(
            {
                "model": label,
                "target": target,
                "MAE": mean_absolute_error(y_true, y_pred),
                "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
                "R2": r2_score(y_true, y_pred),
                "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
                "sMAPE": smape(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)


def choose_blend_weights(
    system: ForecastSystem, validation: pd.DataFrame
) -> tuple[float, float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    direct_system = ForecastSystem(
        system.history, system.reference_cache, system.models, 1.0, 1.0
    )
    component_system = ForecastSystem(
        system.history, system.reference_cache, system.models, 0.0, 0.0
    )
    direct = direct_system.forecast(validation["Date"])
    component = component_system.forecast(validation["Date"])

    merged = validation[["Date", "Revenue", "COGS"]].merge(
        direct, on="Date", suffixes=("_actual", "_direct")
    ).merge(component, on="Date")
    merged = merged.rename(columns={"Revenue": "Revenue_component", "COGS": "COGS_component"})

    weight_grid = np.linspace(0, 1, 11)
    best = {}
    rows = []
    for target in TARGETS:
        y_true = merged[f"{target}_actual"].to_numpy()
        best_rmse = np.inf
        best_weight = 1.0
        for weight in weight_grid:
            y_pred = (
                weight * merged[f"{target}_direct"].to_numpy()
                + (1 - weight) * merged[f"{target}_component"].to_numpy()
            )
            rmse = mean_squared_error(y_true, y_pred) ** 0.5
            rows.append({"target": target, "direct_weight": weight, "RMSE": rmse})
            if rmse < best_rmse:
                best_rmse = rmse
                best_weight = float(weight)
        best[target] = best_weight
    return best["Revenue"], best["COGS"], pd.DataFrame(rows), direct, component


def evaluate_validation(history: pd.DataFrame) -> tuple[float, float]:
    train = history[history["Date"].dt.year < 2022].copy()
    validation = history[history["Date"].dt.year == 2022].copy()
    base_system = fit_base_system(train)

    revenue_weight, cogs_weight, blend_table, direct_pred, component_pred = choose_blend_weights(
        base_system, validation
    )
    hybrid_system = ForecastSystem(
        base_system.history,
        base_system.reference_cache,
        base_system.models,
        revenue_weight,
        cogs_weight,
    )

    metrics = pd.concat(
        [
            metric_frame(validation, direct_pred, "direct"),
            metric_frame(validation, component_pred, "component"),
            metric_frame(validation, hybrid_system.forecast(validation["Date"]), "hybrid"),
        ],
        ignore_index=True,
    )
    print("Validation on 2022")
    with pd.option_context("display.float_format", "{:,.4f}".format):
        print(metrics.to_string(index=False))
    print(
        f"\nChosen blend weights: Revenue direct={revenue_weight:.1f}, "
        f"COGS direct={cogs_weight:.1f}"
    )
    print("\nBest blend grid rows")
    print(
        blend_table.sort_values(["target", "RMSE"])
        .groupby("target")
        .head(3)
        .to_string(index=False)
    )
    print("\nTop LightGBM features")
    print(base_system.feature_importance_table().to_string(index=False))

    best_rows = metrics.sort_values("RMSE").groupby("target").head(1)
    best_model_by_target = dict(zip(best_rows["target"], best_rows["model"]))
    revenue_weight = {"direct": 1.0, "component": 0.0, "hybrid": revenue_weight}[
        best_model_by_target["Revenue"]
    ]
    cogs_weight = {"direct": 1.0, "component": 0.0, "hybrid": cogs_weight}[
        best_model_by_target["COGS"]
    ]
    print(
        f"\nFinal recursive-validation weights: Revenue direct={revenue_weight:.1f}, "
        f"COGS direct={cogs_weight:.1f}"
    )
    return revenue_weight, cogs_weight


def main() -> None:
    history = load_history()
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])

    revenue_weight, cogs_weight = evaluate_validation(history)

    base_system = fit_base_system(history)
    final_system = ForecastSystem(
        base_system.history,
        base_system.reference_cache,
        base_system.models,
        revenue_weight,
        cogs_weight,
    )
    submission = final_system.forecast(sample["Date"])[["Date", "Revenue", "COGS"]]
    submission["Date"] = submission["Date"].dt.strftime("%Y-%m-%d")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUT_FILE, index=False)

    print(f"\nSaved {len(submission)} rows to {OUT_FILE}")
    print(submission.head(10))


if __name__ == "__main__":
    main()
