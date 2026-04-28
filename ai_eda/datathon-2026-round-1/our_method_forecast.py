import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

DATA_DIR = Path(".")
OUT_FILE = DATA_DIR / "submission_our_method.csv"
RANDOM_STATE = 42
TARGETS = ["Revenue", "COGS"]
SERIES_FOR_HISTORY = ["Revenue", "COGS", "gross_profit", "cogs_ratio", "gross_margin"]
LAGS = [1, 7, 30, 90, 364, 365, 366, 730, 1095]
ROLL_WINDOWS = [7, 30, 90, 365]
TOP_N_FEATURES = 1
MANUAL_FEATURES = {
    "Revenue": [
        "Revenue_seasonal_base_log",
        "Revenue_dow_avg",
        "is_low_quarter_eda",
    ],
    "COGS": [
        "COGS_seasonal_base_log",
        "COGS_dow_avg",
        "is_low_quarter_eda",
    ],
}
TARGET_PREFIXES = ("Revenue", "COGS", "gross_profit", "cogs_ratio", "gross_margin")
CROSS_TARGET_FEATURES = {
    "lag1_cogs_to_revenue",
    "lag7_cogs_to_revenue",
    "rolling30_cogs_to_revenue",
    "rolling90_cogs_to_revenue",
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


def add_relationship_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gross_profit"] = out["Revenue"] - out["COGS"]
    out["cogs_ratio"] = safe_divide(out["COGS"], out["Revenue"])
    out["gross_margin"] = safe_divide(out["gross_profit"], out["Revenue"])
    out[["cogs_ratio", "gross_margin"]] = out[
        ["cogs_ratio", "gross_margin"]
    ].replace([np.inf, -np.inf], np.nan)
    out[["cogs_ratio", "gross_margin"]] = out[
        ["cogs_ratio", "gross_margin"]
    ].ffill().bfill()
    return out


def load_sales_history() -> pd.DataFrame:
    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"])
    required = {"Date", "Revenue", "COGS"}
    missing = required.difference(sales.columns)
    if missing:
        raise ValueError(f"sales.csv is missing required columns: {sorted(missing)}")

    daily = (
        sales[["Date", "Revenue", "COGS"]]
        .dropna(subset=["Date"])
        .groupby("Date", as_index=False)[["Revenue", "COGS"]]
        .sum()
        .sort_values("Date")
    )
    full_dates = pd.DataFrame(
        {"Date": pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")}
    )
    daily = full_dates.merge(daily, on="Date", how="left")
    daily[["Revenue", "COGS"]] = daily[["Revenue", "COGS"]].interpolate(
        limit_direction="both"
    )
    return add_relationship_columns(daily)


def annual_growth(history: pd.DataFrame, target: str) -> float:
    annual = history.groupby(history["Date"].dt.year)[target].mean().replace(0, np.nan)
    annual = annual.dropna()
    annual = annual.loc[annual.index >= 2013]
    yoy = annual.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if yoy.empty:
        return 1.0
    growth = float((1 + yoy).prod() ** (1 / len(yoy)))
    if not np.isfinite(growth) or growth <= 0:
        return 1.0
    return float(np.clip(growth, 0.88, 1.08))


def make_reference_cache(history: pd.DataFrame) -> dict[str, dict[str, object]]:
    cache = {}
    for target in TARGETS:
        hist = history[["Date", target]].copy()
        hist["year"] = hist["Date"].dt.year
        hist["month"] = hist["Date"].dt.month
        hist["day"] = hist["Date"].dt.day
        hist["quarter"] = hist["Date"].dt.quarter
        hist["dow"] = hist["Date"].dt.dayofweek

        annual_mean = hist.groupby("year")[target].mean().replace(0, np.nan)
        latest_year = int(annual_mean.dropna().index.max())
        latest_annual_mean = float(annual_mean.loc[latest_year])

        hist = hist.merge(
            annual_mean.rename("annual_mean"),
            left_on="year",
            right_index=True,
            how="left",
        )
        hist["seasonal_norm"] = hist[target] / hist["annual_mean"].replace(0, np.nan)
        cache[target] = {
            "latest_year": latest_year,
            "latest_annual_mean": latest_annual_mean,
            "growth": annual_growth(history, target),
            "month_day_norm": hist.groupby(["month", "day"])["seasonal_norm"].mean(),
            "month_mean": hist.groupby("month")[target].mean(),
            "quarter_mean": hist.groupby("quarter")[target].mean(),
            "dow_mean": hist.groupby("dow")[target].mean(),
        }
    return cache


def make_calendar_features(
    dates: pd.Series, history_start: pd.Timestamp
) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(dates), name="Date").reset_index(drop=True)
    x = pd.DataFrame(index=np.arange(len(dates)))
    x["time_index"] = (dates - history_start).dt.days.astype(float)
    x["trend_feature"] = x["time_index"]
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


def add_seasonal_features(
    x: pd.DataFrame, dates: pd.Series, reference_cache: dict[str, dict[str, object]]
) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(dates), name="Date").reset_index(drop=True)
    out = x.copy()
    for target, ref in reference_cache.items():
        md = pd.MultiIndex.from_arrays([dates.dt.month, dates.dt.day])
        seasonal_norm = pd.Series(md.map(ref["month_day_norm"]), index=out.index)
        seasonal_norm = seasonal_norm.astype(float).fillna(1.0)
        years_ahead = dates.dt.year - int(ref["latest_year"])
        base = (
            float(ref["latest_annual_mean"])
            * np.power(float(ref["growth"]), years_ahead)
            * seasonal_norm
        )
        out[f"{target}_seasonal_base"] = base
        out[f"{target}_seasonal_base_log"] = np.log1p(np.clip(base, 0, None))
        out[f"{target}_month_avg"] = dates.dt.month.map(ref["month_mean"]).astype(float)
        out[f"{target}_quarter_avg"] = dates.dt.quarter.map(ref["quarter_mean"]).astype(
            float
        )
        out[f"{target}_dow_avg"] = dates.dt.dayofweek.map(ref["dow_mean"]).astype(float)
    return out


def add_vectorized_history_features(
    x: pd.DataFrame, dates: pd.Series, state: pd.DataFrame
) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(dates), name="Date").reset_index(drop=True)
    out = x.copy()
    indexed = state.set_index("Date").sort_index()
    full_index = pd.date_range(
        min(indexed.index.min(), dates.min()), max(indexed.index.max(), dates.max()), freq="D"
    )

    for series_name in SERIES_FOR_HISTORY:
        series = indexed[series_name].reindex(full_index).astype(float)
        for lag in LAGS:
            out[f"{series_name}_lag_{lag}"] = dates.map(
                series.shift(lag, freq="D")
            ).astype(float)

        out[f"{series_name}_yoy_lag_mean"] = out[
            [f"{series_name}_lag_364", f"{series_name}_lag_365", f"{series_name}_lag_366"]
        ].mean(axis=1)
        out[f"{series_name}_two_year_lag"] = out[f"{series_name}_lag_730"]

        shifted = series.shift(1)
        for window in ROLL_WINDOWS:
            min_periods = max(2, min(window // 3, 30))
            rolling = shifted.rolling(window, min_periods=min_periods)
            out[f"{series_name}_rolling_mean_{window}"] = dates.map(
                rolling.mean()
            ).astype(float)
            out[f"{series_name}_rolling_std_{window}"] = dates.map(
                rolling.std()
            ).astype(float)

    out["lag1_cogs_to_revenue"] = safe_divide(out["COGS_lag_1"], out["Revenue_lag_1"])
    out["lag7_cogs_to_revenue"] = safe_divide(out["COGS_lag_7"], out["Revenue_lag_7"])
    out["rolling30_cogs_to_revenue"] = safe_divide(
        out["COGS_rolling_mean_30"], out["Revenue_rolling_mean_30"]
    )
    out["rolling90_cogs_to_revenue"] = safe_divide(
        out["COGS_rolling_mean_90"], out["Revenue_rolling_mean_90"]
    )
    return out.replace([np.inf, -np.inf], np.nan)


def make_candidate_features(
    dates: pd.Series,
    state: pd.DataFrame,
    history_start: pd.Timestamp,
    reference_cache: dict[str, dict[str, object]],
) -> pd.DataFrame:
    x = make_calendar_features(dates, history_start)
    x = add_seasonal_features(x, dates, reference_cache)
    x = add_vectorized_history_features(x, dates, state)
    return x


def allowed_features_for_target(columns: pd.Index, target: str) -> list[str]:
    """Keep target-native history/seasonality plus pure calendar features only."""
    allowed = []
    for col in columns:
        if col in CROSS_TARGET_FEATURES:
            continue
        matching_prefix = next(
            (prefix for prefix in TARGET_PREFIXES if col.startswith(f"{prefix}_")),
            None,
        )
        if matching_prefix is None:
            allowed.append(col)
        elif matching_prefix == target:
            allowed.append(col)
    return allowed


def make_single_candidate_row(
    date: pd.Timestamp,
    state_indexed: pd.DataFrame,
    history_start: pd.Timestamp,
    reference_cache: dict[str, dict[str, object]],
) -> pd.DataFrame:
    date = pd.Timestamp(date)
    x = make_calendar_features(pd.Series([date]), history_start)
    x = add_seasonal_features(x, pd.Series([date]), reference_cache)

    for series_name in SERIES_FOR_HISTORY:
        series = state_indexed[series_name].astype(float)
        past = series.loc[: date - pd.Timedelta(days=1)]

        for lag in LAGS:
            lag_date = date - pd.Timedelta(days=lag)
            x.loc[0, f"{series_name}_lag_{lag}"] = series.get(lag_date, np.nan)

        x.loc[0, f"{series_name}_yoy_lag_mean"] = x.loc[
            0,
            [f"{series_name}_lag_364", f"{series_name}_lag_365", f"{series_name}_lag_366"],
        ].mean()
        x.loc[0, f"{series_name}_two_year_lag"] = x.loc[
            0, f"{series_name}_lag_730"
        ]

        for window in ROLL_WINDOWS:
            min_periods = max(2, min(window // 3, 30))
            values = past.tail(window).dropna()
            if len(values) >= min_periods:
                x.loc[0, f"{series_name}_rolling_mean_{window}"] = values.mean()
                x.loc[0, f"{series_name}_rolling_std_{window}"] = values.std()
            else:
                x.loc[0, f"{series_name}_rolling_mean_{window}"] = np.nan
                x.loc[0, f"{series_name}_rolling_std_{window}"] = np.nan

    x["lag1_cogs_to_revenue"] = safe_divide(x["COGS_lag_1"], x["Revenue_lag_1"])
    x["lag7_cogs_to_revenue"] = safe_divide(x["COGS_lag_7"], x["Revenue_lag_7"])
    x["rolling30_cogs_to_revenue"] = safe_divide(
        x["COGS_rolling_mean_30"], x["Revenue_rolling_mean_30"]
    )
    x["rolling90_cogs_to_revenue"] = safe_divide(
        x["COGS_rolling_mean_90"], x["Revenue_rolling_mean_90"]
    )
    return x.replace([np.inf, -np.inf], np.nan)


@dataclass
class DirectTargetModel:
    target: str
    features: list[str]
    model: Pipeline
    importances: pd.Series

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        pred = self.model.predict(x.reindex(columns=self.features))
        return np.maximum(np.expm1(pred), 0.0)


def fit_direct_model(
    history: pd.DataFrame,
    target: str,
    reference_cache: dict[str, dict[str, object]],
) -> DirectTargetModel:
    x_all = make_candidate_features(
        history["Date"], history, history["Date"].min(), reference_cache
    )
    allowed_columns = allowed_features_for_target(x_all.columns, target)
    x_target = x_all[allowed_columns]
    y = np.log1p(history[target].clip(lower=0))

    selector = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    random_state=RANDOM_STATE,
                    n_estimators=350,
                    min_samples_leaf=2,
                    max_features=0.85,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    selector.fit(x_target, y)
    importances = pd.Series(
        selector.named_steps["model"].feature_importances_, index=x_target.columns
    ).sort_values(ascending=False)
    selected_features = MANUAL_FEATURES.get(target)
    if selected_features is None:
        selected_features = importances.head(TOP_N_FEATURES).index.tolist()
    missing_manual_features = sorted(set(selected_features).difference(x_target.columns))
    if missing_manual_features:
        raise ValueError(
            f"Manual features for {target} are not available: {missing_manual_features}"
        )

    final_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    random_state=RANDOM_STATE,
                    max_iter=350,
                    learning_rate=0.035,
                    max_leaf_nodes=15,
                    l2_regularization=0.4,
                ),
            ),
        ]
    )
    final_model.fit(x_target[selected_features], y)
    return DirectTargetModel(target, selected_features, final_model, importances)


@dataclass
class DirectForecastSystem:
    history: pd.DataFrame
    reference_cache: dict[str, dict[str, object]]
    models: dict[str, DirectTargetModel]

    def forecast(self, forecast_dates: pd.Series) -> pd.DataFrame:
        forecast_dates = pd.Series(pd.to_datetime(forecast_dates), name="Date")
        state = self.history.set_index("Date").sort_index().copy()
        rows = []

        for date in forecast_dates:
            x = make_single_candidate_row(
                date, state, self.history["Date"].min(), self.reference_cache
            )
            revenue_pred = float(self.models["Revenue"].predict(x)[0])
            cogs_pred = float(self.models["COGS"].predict(x)[0])
            gross_profit = revenue_pred - cogs_pred
            cogs_ratio = cogs_pred / revenue_pred if revenue_pred > 1e-9 else np.nan
            gross_margin = gross_profit / revenue_pred if revenue_pred > 1e-9 else np.nan

            state.loc[
                date, ["Revenue", "COGS", "gross_profit", "cogs_ratio", "gross_margin"]
            ] = [revenue_pred, cogs_pred, gross_profit, cogs_ratio, gross_margin]
            rows.append({"Date": date, "Revenue": revenue_pred, "COGS": cogs_pred})

        out = pd.DataFrame(rows)
        out["Revenue"] = out["Revenue"].round(2)
        out["COGS"] = out["COGS"].round(2)
        return out

    def selected_feature_table(self) -> pd.DataFrame:
        rows = []
        for target, model in self.models.items():
            for rank, feature in enumerate(model.features, start=1):
                rows.append(
                    {
                        "target": target,
                        "rank": rank,
                        "feature": feature,
                        "importance": float(model.importances.loc[feature]),
                    }
                )
        return pd.DataFrame(rows)


def fit_forecast_system(history: pd.DataFrame) -> DirectForecastSystem:
    history = add_relationship_columns(history).sort_values("Date").reset_index(drop=True)
    reference_cache = make_reference_cache(history)
    models = {
        target: fit_direct_model(history, target, reference_cache) for target in TARGETS
    }
    return DirectForecastSystem(history, reference_cache, models)


def evaluate_validation(history: pd.DataFrame) -> None:
    train = history[history["Date"].dt.year < 2021].copy()
    validation = history[history["Date"].dt.year >= 2021].copy()
    system = fit_forecast_system(train)
    predicted = system.forecast(validation["Date"])
    actual = validation.merge(
        predicted[["Date", "Revenue", "COGS"]],
        on="Date",
        suffixes=("_actual", "_pred"),
    )

    print("Validation on 2021-2022")
    for target in TARGETS:
        y_true = actual[f"{target}_actual"].to_numpy()
        y_pred = actual[f"{target}_pred"].to_numpy()
        print(
            f"{target}: "
            f"MAE={mean_absolute_error(y_true, y_pred):,.2f} "
            f"RMSE={mean_squared_error(y_true, y_pred) ** 0.5:,.2f} "
            f"R2={r2_score(y_true, y_pred):.4f} "
            f"MAPE={mean_absolute_percentage_error(y_true, y_pred) * 100:.2f}% "
            f"sMAPE={smape(y_true, y_pred):.2f}%"
        )

    print("\nValidation-selected manual features")
    print(system.selected_feature_table().to_string(index=False))


def main() -> None:
    history = load_sales_history()
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])

    evaluate_validation(history)

    system = fit_forecast_system(history)
    submission = system.forecast(sample["Date"])[["Date", "Revenue", "COGS"]]
    submission["Date"] = submission["Date"].dt.strftime("%Y-%m-%d")
    submission.to_csv(OUT_FILE, index=False)

    print(f"\nSaved {len(submission)} rows to {OUT_FILE}")
    print(submission.head(10))
    print("\nFinal manual features used by each model")
    print(system.selected_feature_table().to_string(index=False))


if __name__ == "__main__":
    main()
