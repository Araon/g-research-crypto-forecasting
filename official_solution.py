"""Train a stronger, reproducible baseline on the official G-Research data.

The original competition predicts a short-horizon residual return for fourteen
crypto assets.  This script deliberately keeps the experiment small enough to
run locally while preserving the parts that make a quant backtest credible:

* the source is the original multi-asset competition archive;
* all splits move forward in time, never randomly;
* a 16-minute embargo separates each train/validation boundary because the
  supplied target looks forward in time; and
* validation uses the competition's asset weights with a strict Pearson metric.

Run ``python prepare_official_data.py`` first (with the JAY environment), then
run this script in the main Python environment.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path("data/official/train_last_180_days.parquet")
MANIFEST_PATH = Path("data/official/train_last_180_days.manifest.json")
OUTPUT_DIR = Path("outputs")
EMBARGO = pd.Timedelta(minutes=16)
HOLDOUT_FRACTION = 0.15
RANDOM_SEED = 42


@dataclass(frozen=True)
class ModelConfig:
    """A deliberately short candidate list keeps selection affordable locally."""

    learning_rate: float
    num_leaves: int
    min_child_samples: int
    feature_fraction: float
    reg_lambda: float
    n_estimators: int = 350


CANDIDATES = [
    ModelConfig(learning_rate=0.04, num_leaves=31, min_child_samples=600, feature_fraction=0.8, reg_lambda=2.0),
    ModelConfig(learning_rate=0.03, num_leaves=63, min_child_samples=900, feature_fraction=0.9, reg_lambda=5.0),
]

FEATURES = [
    "asset_id", "return_1m", "return_5m", "return_15m", "return_60m",
    "volatility_15_rows", "volatility_60_rows", "relative_volume_15_rows",
    "relative_volume_60_rows", "open_to_close", "high_low_range", "vwap_gap",
    "market_return_1m", "market_return_5m", "market_breadth_1m",
    "relative_market_return_1m", "minute_sin", "minute_cos", "weekday_sin", "weekday_cos",
]


def sha256(path: Path) -> str:
    """Hash a source file so another person can reproduce this exact run."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_pearson(actual: pd.Series, prediction: np.ndarray, weight: pd.Series) -> float:
    """Competition-style weighted Pearson correlation, without silent filtering."""
    values = np.column_stack((actual.to_numpy(float), np.asarray(prediction, dtype=float), weight.to_numpy(float)))
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Every scored target, prediction, and weight must be finite.")
    if (values[:, 2] <= 0).any():
        raise ValueError("Every competition weight must be positive.")
    y, p, w = values.T
    y = y - np.average(y, weights=w)
    p = p - np.average(p, weights=w)
    denominator = np.sqrt(np.sum(w * y**2) * np.sum(w * p**2))
    if denominator == 0:
        raise ValueError("Pearson correlation is undefined for a constant series.")
    return float(np.sum(w * y * p) / denominator)


def exact_return(data: pd.DataFrame, minutes: int) -> pd.Series:
    """Use a lag only when the previous observation is exactly N minutes ago."""
    grouped = data.groupby("asset_id", group_keys=False)
    old_close = grouped["close"].shift(minutes)
    old_time = grouped["timestamp"].shift(minutes)
    is_exact = data["timestamp"] - old_time == pd.Timedelta(minutes=minutes)
    return (data["close"] / old_close - 1.0).where(is_exact)


def load_and_feature_engineer() -> pd.DataFrame:
    """Read the prepared official window and make only information-known-now features."""
    if not DATA_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError("Run `.jay-venv/bin/python prepare_official_data.py` before this script.")
    data = pd.read_parquet(DATA_PATH)
    required = {"timestamp", "asset_id", "weight", "open", "high", "low", "close", "volume", "vwap", "target"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Official window is missing columns: {sorted(missing)}")
    if data.duplicated(["timestamp", "asset_id"]).any():
        raise ValueError("The official source has duplicate asset/timestamp rows.")
    if (data["weight"] <= 0).any() or data["weight"].isna().any():
        raise ValueError("Official asset weights must be present and positive.")

    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    grouped = data.groupby("asset_id", group_keys=False)
    for minutes in (1, 5, 15, 60):
        data[f"return_{minutes}m"] = exact_return(data, minutes)
    data["volatility_15_rows"] = grouped["return_1m"].transform(lambda series: series.rolling(15).std())
    data["volatility_60_rows"] = grouped["return_1m"].transform(lambda series: series.rolling(60).std())
    for rows in (15, 60):
        average_volume = grouped["volume"].transform(lambda series: series.rolling(rows).mean())
        data[f"relative_volume_{rows}_rows"] = data["volume"] / average_volume - 1.0

    data["open_to_close"] = data["close"] / data["open"] - 1.0
    data["high_low_range"] = (data["high"] - data["low"]) / data["close"]
    data["vwap_gap"] = data["close"] / data["vwap"] - 1.0

    # A weighted contemporaneous market move is known after every minute candle closes.
    market_weight = data.groupby("timestamp")["weight"].transform("sum")
    data["market_return_1m"] = (data["return_1m"] * data["weight"]).groupby(data["timestamp"]).transform("sum") / market_weight
    data["market_return_5m"] = (data["return_5m"] * data["weight"]).groupby(data["timestamp"]).transform("sum") / market_weight
    data["market_breadth_1m"] = (data["return_1m"] > 0).groupby(data["timestamp"]).transform("mean")
    data["relative_market_return_1m"] = data["return_1m"] - data["market_return_1m"]

    minute = data["timestamp"].dt.hour * 60 + data["timestamp"].dt.minute
    data["minute_sin"] = np.sin(2 * np.pi * minute / (24 * 60))
    data["minute_cos"] = np.cos(2 * np.pi * minute / (24 * 60))
    data["weekday_sin"] = np.sin(2 * np.pi * data["timestamp"].dt.dayofweek / 7)
    data["weekday_cos"] = np.cos(2 * np.pi * data["timestamp"].dt.dayofweek / 7)
    return data.replace([np.inf, -np.inf], np.nan).dropna(subset=["target"]).reset_index(drop=True)


def build_splits(data: pd.DataFrame) -> tuple[list[tuple[pd.DataFrame, pd.DataFrame, str]], pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Create two expanding folds and one untouched later holdout by timestamp."""
    time_index = np.sort(data["timestamp"].unique())
    indexes = [int(len(time_index) * fraction) for fraction in (0.48, 0.64, 1 - HOLDOUT_FRACTION)]
    first_boundary, second_boundary, holdout_boundary = (pd.Timestamp(time_index[index]) for index in indexes)
    folds = []
    for train_end, validation_end, name in ((first_boundary, second_boundary, "fold_1"), (second_boundary, holdout_boundary, "fold_2")):
        train = data.loc[data["timestamp"] <= train_end - EMBARGO]
        validation = data.loc[(data["timestamp"] >= train_end + EMBARGO) & (data["timestamp"] <= validation_end - EMBARGO)]
        if min(len(train), len(validation)) == 0:
            raise RuntimeError(f"{name} is empty after applying the embargo.")
        folds.append((train, validation, name))
    development = data.loc[data["timestamp"] <= holdout_boundary - EMBARGO]
    holdout = data.loc[data["timestamp"] >= holdout_boundary + EMBARGO]
    if min(len(development), len(holdout)) == 0:
        raise RuntimeError("The final development or holdout partition is empty.")
    metadata = {"embargo_minutes": str(int(EMBARGO.total_seconds() / 60)), "development_end": str(development["timestamp"].max()), "final_holdout_start": str(holdout["timestamp"].min())}
    return folds, development, holdout, metadata


def fit(config: ModelConfig, train: pd.DataFrame, validation: pd.DataFrame | None = None) -> lgb.LGBMRegressor:
    """Fit LightGBM; early stopping only uses a chronological validation period."""
    model = lgb.LGBMRegressor(objective="regression", learning_rate=config.learning_rate, num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples, colsample_bytree=config.feature_fraction, reg_lambda=config.reg_lambda,
        n_estimators=config.n_estimators, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1)
    # Asset_ID is an identity label, not a quantity with meaningful distance.
    arguments: dict[str, object] = {
        "sample_weight": train["weight"],
        "categorical_feature": ["asset_id"],
        "callbacks": [lgb.log_evaluation(period=0)],
    }
    if validation is not None:
        arguments["eval_set"] = [(validation[FEATURES], validation["target"])]
        arguments["eval_sample_weight"] = [validation["weight"]]
        arguments["callbacks"] = [lgb.early_stopping(40, verbose=False), lgb.log_evaluation(period=0)]
    model.fit(train[FEATURES], train["target"], **arguments)
    return model


def select_model(folds: list[tuple[pd.DataFrame, pd.DataFrame, str]]) -> pd.DataFrame:
    """Compare candidates exclusively on chronological validation periods."""
    rows: list[dict[str, object]] = []
    for config in CANDIDATES:
        scores = []
        for train, validation, fold_name in folds:
            model = fit(config, train, validation)
            prediction = model.predict(validation[FEATURES])
            score = weighted_pearson(validation["target"], prediction, validation["weight"])
            # A missing lag means this simple baseline has no directional view;
            # score that explicitly as zero instead of deleting the row.
            naive = weighted_pearson(validation["target"], validation["return_1m"].fillna(0.0).to_numpy(), validation["weight"])
            scores.append(score)
            rows.append({**asdict(config), "fold": fold_name, "weighted_pearson": score, "naive_return_1m": naive, "rows_scored": len(validation), "best_iteration": int(model.best_iteration_ or config.n_estimators)})
        rows.append({**asdict(config), "fold": "mean_validation", "weighted_pearson": float(np.mean(scores)), "naive_return_1m": np.nan, "rows_scored": int(sum(len(validation) for _, validation, _ in folds)), "best_iteration": np.nan})
    return pd.DataFrame(rows)


def write_report(selection: pd.DataFrame, holdout: pd.DataFrame, metrics: dict[str, object]) -> None:
    """Create an image based on actual validation and final-holdout outputs."""
    means = selection.loc[selection["fold"] == "mean_validation"].sort_values("weighted_pearson", ascending=False)
    labels = [f"leaves={row.num_leaves}\nlr={row.learning_rate:g}" for row in means.itertuples()]
    # A raw row-wise rolling window would interleave fourteen assets and, if
    # plotted in asset order, draw false connecting lines.  One strict weighted
    # score per calendar day retains the actual time axis and remains legible.
    daily = (
        holdout.assign(day=holdout["timestamp"].dt.floor("D"))
        .groupby("day", sort=True)
        .apply(lambda frame: weighted_pearson(frame["target"], frame["prediction"], frame["weight"]), include_groups=False)
        .rename("weighted_pearson")
        .reset_index()
    )
    plt.style.use("dark_background")
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#111827")
    axes[0].bar(labels, means["weighted_pearson"], color=["#22c55e", "#38bdf8"])
    axes[0].axhline(0, color="#94a3b8", linewidth=0.8)
    axes[0].set_title("Mean weighted Pearson on expanding folds")
    axes[0].set_ylabel("correlation")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].plot(daily["day"], daily["weighted_pearson"], color="#38bdf8", linewidth=1.5, marker="o", markersize=3)
    axes[1].axhline(0, color="#94a3b8", linewidth=0.8)
    axes[1].set_title(f"Untouched final holdout: {metrics['final_holdout_weighted_pearson']:.4f}")
    axes[1].set_ylabel("daily weighted Pearson correlation")
    axes[1].grid(alpha=0.2)
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=5))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    figure.suptitle("G-Research Crypto Forecasting: official multi-asset LightGBM", fontsize=16, fontweight="bold")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "official_research_report.png", dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = load_and_feature_engineer()
    folds, development, holdout, split_metadata = build_splits(data)
    selection = select_model(folds)
    selection.to_csv(OUTPUT_DIR / "official_model_selection.csv", index=False)
    best = selection.loc[selection["fold"] == "mean_validation"].sort_values("weighted_pearson", ascending=False).iloc[0]
    best_config = ModelConfig(learning_rate=float(best.learning_rate), num_leaves=int(best.num_leaves), min_child_samples=int(best.min_child_samples), feature_fraction=float(best.feature_fraction), reg_lambda=float(best.reg_lambda), n_estimators=int(best.n_estimators))

    # Final training cannot look at holdout; reuse the mean validation stopping length.
    matching = selection.loc[(selection["fold"] != "mean_validation") & (selection["learning_rate"] == best_config.learning_rate) & (selection["num_leaves"] == best_config.num_leaves), "best_iteration"]
    final_config = ModelConfig(**{**asdict(best_config), "n_estimators": max(int(matching.mean()), 40)})
    model = fit(final_config, development)
    holdout = holdout.copy()
    holdout["prediction"] = model.predict(holdout[FEATURES])
    holdout["naive_prediction"] = holdout["return_1m"].fillna(0.0)
    per_asset = [{"asset_name": name, "rows": len(asset), "weighted_pearson": weighted_pearson(asset["target"], asset["prediction"], asset["weight"])} for name, asset in holdout.groupby("asset_name")]
    pd.DataFrame(per_asset).sort_values("weighted_pearson", ascending=False).to_csv(OUTPUT_DIR / "official_holdout_by_asset.csv", index=False)

    manifest = json.loads(MANIFEST_PATH.read_text())
    metrics: dict[str, object] = {"model": "LightGBM regression", "best_config": asdict(final_config), "features": FEATURES,
        "feature_timing": "Features use the current completed minute candle and prior history only.", "development_rows": int(len(development)), "final_holdout_rows": int(len(holdout)), "final_holdout_scored_rows": int(len(holdout)),
        "final_holdout_weighted_pearson": weighted_pearson(holdout["target"], holdout["prediction"], holdout["weight"]), "final_holdout_naive_return_1m_weighted_pearson": weighted_pearson(holdout["target"], holdout["naive_prediction"], holdout["weight"]),
        "evaluation_note": "The final holdout was not used for candidate selection. It remains historical research, not a live trading claim.", "prepared_window": manifest, "prepared_window_sha256": sha256(DATA_PATH), **split_metadata,
        "python": sys.version.split()[0], "platform": platform.platform(), "lightgbm": lgb.__version__, "pandas": pd.__version__}
    (OUTPUT_DIR / "official_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    model.booster_.save_model(str(OUTPUT_DIR / "official_model.txt"))
    sample_columns = ["timestamp", "asset_id", "asset_name", "target", "prediction", "naive_prediction", *[feature for feature in FEATURES if feature != "asset_id"]]
    holdout[sample_columns].head(10_000).to_parquet(OUTPUT_DIR / "official_holdout_prediction_sample.parquet", index=False)
    write_report(selection, holdout, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
