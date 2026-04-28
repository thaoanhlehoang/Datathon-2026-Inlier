import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

import our_method_forecast as om

warnings.filterwarnings("ignore")

DATA_DIR = Path(".")
OUT_FILE = DATA_DIR / "submission_our_method_overfit.csv"


def make_calibration_features(prediction: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame({"Date": prediction["Date"]})
    d = features["Date"]

    features["year"] = d.dt.year
    features["month"] = d.dt.month
    features["day"] = d.dt.day
    features["dow"] = d.dt.dayofweek
    features["day_of_year"] = d.dt.dayofyear.clip(upper=365)
    features["week_of_year"] = d.dt.isocalendar().week.astype(int)
    features["is_weekend"] = (features["dow"] >= 5).astype(int)
    features["is_month_start"] = d.dt.is_month_start.astype(int)
    features["is_month_end"] = d.dt.is_month_end.astype(int)

    for harmonic in range(1, 5):
        angle = 2 * np.pi * harmonic * features["day_of_year"] / 365.25
        features[f"sin_year_{harmonic}"] = np.sin(angle)
        features[f"cos_year_{harmonic}"] = np.cos(angle)

    features["Revenue_base"] = prediction["Revenue"].to_numpy()
    features["COGS_base"] = prediction["COGS"].to_numpy()
    features["base_cogs_ratio"] = features["COGS_base"] / features["Revenue_base"]
    features["log_Revenue_base"] = np.log1p(features["Revenue_base"])
    features["log_COGS_base"] = np.log1p(features["COGS_base"])
    return features.drop(columns=["Date"])


def fit_overfit_calibrators(
    validation_actual: pd.DataFrame, validation_base_prediction: pd.DataFrame
) -> dict[str, object]:
    train = validation_actual.merge(
        validation_base_prediction, on="Date", suffixes=("_actual", "_base")
    )
    x_train = make_calibration_features(
        train.rename(columns={"Revenue_base": "Revenue", "COGS_base": "COGS"})[
            ["Date", "Revenue", "COGS"]
        ]
    )

    calibrators = {}
    for target in ["Revenue", "COGS"]:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                random_state=42,
                n_estimators=800,
                min_samples_leaf=1,
                bootstrap=False,
                n_jobs=-1,
            ),
        )
        model.fit(x_train, train[f"{target}_actual"])
        calibrators[target] = model
    return calibrators


def apply_calibrators(
    base_prediction: pd.DataFrame, calibrators: dict[str, object]
) -> pd.DataFrame:
    x_pred = make_calibration_features(base_prediction)
    calibrated = base_prediction[["Date"]].copy()
    for target in ["Revenue", "COGS"]:
        calibrated[target] = np.maximum(calibrators[target].predict(x_pred), 0.0)
    calibrated["Revenue"] = calibrated["Revenue"].round(2)
    calibrated["COGS"] = calibrated["COGS"].round(2)
    return calibrated


def print_metrics(label: str, actual: pd.DataFrame, predicted: pd.DataFrame) -> None:
    merged = actual.merge(predicted, on="Date", suffixes=("_actual", "_pred"))
    print(label)
    for target in ["Revenue", "COGS"]:
        y_true = merged[f"{target}_actual"].to_numpy()
        y_pred = merged[f"{target}_pred"].to_numpy()
        print(
            f"{target}: "
            f"MAE={mean_absolute_error(y_true, y_pred):,.2f} "
            f"RMSE={mean_squared_error(y_true, y_pred) ** 0.5:,.2f} "
            f"R2={r2_score(y_true, y_pred):.4f}"
        )


def main() -> None:
    components = om.build_daily_business_components()
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])

    validation_train = components[components["Date"].dt.year < 2021].copy()
    validation_actual = components.loc[
        components["Date"].dt.year >= 2021, ["Date", "Revenue", "COGS"]
    ].copy()
    validation_base = om.forecast_bottom_up(validation_train, validation_actual["Date"])[
        ["Date", "Revenue", "COGS"]
    ]

    calibrators = fit_overfit_calibrators(validation_actual, validation_base)
    validation_overfit = apply_calibrators(validation_base, calibrators)

    print_metrics("Adapted bottom-up validation on 2021-2022", validation_actual, validation_base)
    print_metrics(
        "Overfit-calibrated validation on 2021-2022",
        validation_actual,
        validation_overfit,
    )

    full_base_forecast = om.forecast_bottom_up(components, sample["Date"])[
        ["Date", "Revenue", "COGS"]
    ]
    submission = apply_calibrators(full_base_forecast, calibrators)
    submission["Date"] = submission["Date"].dt.strftime("%Y-%m-%d")
    submission.to_csv(OUT_FILE, index=False)

    print(f"Saved {len(submission)} rows to {OUT_FILE}")
    print(submission.head(10))


if __name__ == "__main__":
    main()
