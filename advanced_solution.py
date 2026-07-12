"""Reproducible gradient-boosting research baseline for public G-Research data."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor


DATA_PATH = Path("data/full_data__3__2018.csv")
OUTPUT_DIR = Path("outputs")
EMBARGO = pd.Timedelta(minutes=16)
FINAL_HOLDOUT_FRACTION = 0.15


@dataclass(frozen=True)
class ModelConfig:
    learning_rate: float
    max_leaf_nodes: int
    l2_regularization: float
    max_iter: int = 60


CANDIDATES = [
    ModelConfig(learning_rate=0.05, max_leaf_nodes=31, l2_regularization=5.0),
    ModelConfig(learning_rate=0.08, max_leaf_nodes=63, l2_regularization=10.0),
]

FEATURES = [
    "asset_id",
    "return_1m",
    "return_5m",
    "return_15m",
    "return_30m",
    "return_60m",
    "volatility_15_rows",
    "volatility_60_rows",
    "relative_volume_15_rows",
    "relative_volume_60_rows",
    "open_to_close",
    "high_low_range",
    "vwap_gap",
    "minute_sin",
    "minute_cos",
    "weekday_sin",
    "weekday_cos",
]

REQUIRED_COLUMNS = {
    "timestamp",
    "asset_id",
    "weight",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "target",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_pearson(actual: pd.Series, prediction: pd.Series, weight: pd.Series) -> float:
    """Return weighted Pearson correlation without silently excluding any row."""
    frame = pd.DataFrame({"actual": actual, "prediction": prediction, "weight": weight})
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Evaluation inputs must be finite on every scored row.")
    if (frame["weight"] <= 0).any():
        raise ValueError("Evaluation weights must be strictly positive.")

    w = frame["weight"].to_numpy(dtype=float)
    y = frame["actual"].to_numpy(dtype=float)
    p = frame["prediction"].to_numpy(dtype=float)
    y_centered = y - np.average(y, weights=w)
    p_centered = p - np.average(p, weights=w)
    denominator = np.sqrt(np.sum(w * y_centered**2) * np.sum(w * p_centered**2))
    if denominator == 0:
        raise ValueError("Pearson correlation is undefined for zero-variance inputs.")
    return float(np.sum(w * y_centered * p_centered) / denominator)


def load_source() -> tuple[pd.DataFrame, dict[str, object]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run `python download_data.py` before training the model.")

    data = pd.read_csv(DATA_PATH).rename(columns=str.lower)
    data = data.drop(columns=[column for column in data.columns if column.startswith("unnamed")])
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Source data is missing required columns: {sorted(missing)}")

    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="s", utc=True)
    if data.duplicated(["asset_id", "timestamp"]).any():
        raise ValueError("Source data contains duplicate asset/timestamp observations.")
    if (data[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be strictly positive.")
    if (data["volume"] < 0).any():
        raise ValueError("Volume must be non-negative.")

    source_has_weights = data["weight"].notna().any()
    if source_has_weights and not data["weight"].notna().all():
        raise ValueError("Mixed missing and non-missing weights require an explicit source-data join.")
    data["weight"] = data["weight"].fillna(1.0)
    if (data["weight"] <= 0).any():
        raise ValueError("Source weights must be strictly positive.")

    metadata = {
        "source_file": str(DATA_PATH),
        "source_sha256": sha256(DATA_PATH),
        "source_rows": int(len(data)),
        "source_assets": int(data["asset_id"].nunique()),
        "source_has_non_null_weights": bool(source_has_weights),
    }
    return data, metadata


def add_return_feature(data: pd.DataFrame, minutes: int) -> pd.Series:
    """Return only when the lagged observation is exactly `minutes` old."""
    groups = data.groupby("asset_id", group_keys=False)
    prior_close = groups["close"].shift(minutes)
    prior_time = groups["timestamp"].shift(minutes)
    exact_lag = data["timestamp"] - prior_time == pd.Timedelta(minutes=minutes)
    return (data["close"] / prior_close - 1).where(exact_lag)


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create point-in-time features after the current minute's candle has closed."""
    data = data.sort_values(["asset_id", "timestamp"]).copy()
    for minutes in (1, 5, 15, 30, 60):
        data[f"return_{minutes}m"] = add_return_feature(data, minutes)

    groups = data.groupby("asset_id", group_keys=False)
    data["volatility_15_rows"] = groups["return_1m"].transform(lambda value: value.rolling(15).std())
    data["volatility_60_rows"] = groups["return_1m"].transform(lambda value: value.rolling(60).std())
    for window in (15, 60):
        mean_volume = groups["volume"].transform(lambda value: value.rolling(window).mean())
        data[f"relative_volume_{window}_rows"] = data["volume"] / mean_volume - 1

    data["open_to_close"] = data["close"] / data["open"] - 1
    data["high_low_range"] = (data["high"] - data["low"]) / data["close"]
    data["vwap_gap"] = data["close"] / data["vwap"] - 1

    minute_of_day = data["timestamp"].dt.hour * 60 + data["timestamp"].dt.minute
    data["minute_sin"] = np.sin(2 * np.pi * minute_of_day / (24 * 60))
    data["minute_cos"] = np.cos(2 * np.pi * minute_of_day / (24 * 60))
    data["weekday_sin"] = np.sin(2 * np.pi * data["timestamp"].dt.dayofweek / 7)
    data["weekday_cos"] = np.cos(2 * np.pi * data["timestamp"].dt.dayofweek / 7)
    return data.replace([np.inf, -np.inf], np.nan).dropna(subset=["target"]).reset_index(drop=True)


def split_partitions(data: pd.DataFrame) -> tuple[list[tuple[pd.DataFrame, pd.DataFrame, str]], pd.DataFrame, dict[str, str]]:
    """Build three expanding validation folds plus a later embargoed holdout."""
    timestamps = np.sort(data["timestamp"].unique())
    boundary_indexes = [int(len(timestamps) * fraction) for fraction in (0.50, 0.60, 0.70, 1 - FINAL_HOLDOUT_FRACTION)]
    boundaries = [pd.Timestamp(timestamps[index]) for index in boundary_indexes]
    folds = []
    for number, (train_boundary, validation_boundary) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        train = data.loc[data["timestamp"] <= train_boundary - EMBARGO].copy()
        validation = data.loc[
            (data["timestamp"] >= train_boundary + EMBARGO)
            & (data["timestamp"] <= validation_boundary - EMBARGO)
        ].copy()
        if min(len(train), len(validation)) == 0:
            raise RuntimeError(f"Validation fold {number} is empty.")
        folds.append((train, validation, f"fold_{number}"))

    final_boundary = boundaries[-1]
    development = data.loc[data["timestamp"] <= final_boundary - EMBARGO].copy()
    final_test = data.loc[data["timestamp"] >= final_boundary + EMBARGO].copy()
    if min(len(development), len(final_test)) == 0:
        raise RuntimeError("Development or final holdout partition is empty.")
    metadata = {
        "embargo_minutes": str(int(EMBARGO.total_seconds() / 60)),
        "final_development_end": str(development["timestamp"].max()),
        "final_holdout_start": str(final_test["timestamp"].min()),
    }
    return folds, final_test, metadata


def fit(config: ModelConfig, data: pd.DataFrame) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        learning_rate=config.learning_rate,
        max_leaf_nodes=config.max_leaf_nodes,
        l2_regularization=config.l2_regularization,
        max_iter=config.max_iter,
        early_stopping=False,
        random_state=42,
    )
    model.fit(data[FEATURES], data["target"], sample_weight=data["weight"])
    return model


def compare_candidates(folds: list[tuple[pd.DataFrame, pd.DataFrame, str]]) -> pd.DataFrame:
    rows = []
    for config in CANDIDATES:
        fold_scores = []
        for train, validation, name in folds:
            model = fit(config, train)
            prediction = model.predict(validation[FEATURES])
            score = weighted_pearson(validation["target"], prediction, validation["weight"])
            fold_scores.append(score)
            rows.append({**asdict(config), "fold": name, "weighted_pearson": score, "rows_scored": len(validation)})
        rows.append(
            {
                **asdict(config),
                "fold": "mean_validation",
                "weighted_pearson": float(np.mean(fold_scores)),
                "rows_scored": int(sum(len(validation) for _, validation, _ in folds)),
            }
        )
    return pd.DataFrame(rows)


def write_report(selection: pd.DataFrame, metrics: dict[str, object], test: pd.DataFrame) -> None:
    """Generate a transparent visual summary from the actual fitted-model output."""
    grouped = selection.loc[selection["fold"] != "mean_validation"].groupby(
        ["learning_rate", "max_leaf_nodes", "l2_regularization"], as_index=False
    )["weighted_pearson"].mean()
    labels = [f"lr={row.learning_rate:g}\nleaves={int(row.max_leaf_nodes)}" for row in grouped.itertuples()]
    rolling = test.assign(rolling_correlation=test["target"].rolling(1_440).corr(test["prediction"]))

    plt.style.use("dark_background")
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#111827")
    axes[0].bar(labels, grouped["weighted_pearson"], color=["#38bdf8", "#22c55e"])
    axes[0].axhline(0, color="#94a3b8", linewidth=0.8)
    axes[0].set_title("Expanding validation: mean weighted Pearson")
    axes[0].set_ylabel("correlation")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].plot(test["timestamp"], rolling["rolling_correlation"], color="#38bdf8", linewidth=1)
    axes[1].axhline(0, color="#94a3b8", linewidth=0.8)
    axes[1].set_title(f"Final holdout: Pearson {metrics['final_holdout_weighted_pearson']:.4f}")
    axes[1].set_ylabel("1,440-row rolling Pearson correlation")
    axes[1].grid(alpha=0.2)
    figure.suptitle("G-Research Crypto Forecasting: gradient-boosted baseline", fontsize=16, fontweight="bold")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "research_report.png", dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw_data, source_metadata = load_source()
    data = make_features(raw_data)
    folds, final_test, split_metadata = split_partitions(data)
    selection = compare_candidates(folds)
    selection.to_csv(OUTPUT_DIR / "model_selection.csv", index=False)

    summary = selection.loc[selection["fold"] == "mean_validation"].sort_values("weighted_pearson", ascending=False).iloc[0]
    best_config = ModelConfig(
        learning_rate=float(summary["learning_rate"]),
        max_leaf_nodes=int(summary["max_leaf_nodes"]),
        l2_regularization=float(summary["l2_regularization"]),
        max_iter=int(summary["max_iter"]),
    )
    final_development = data.loc[data["timestamp"] < final_test["timestamp"].min() - EMBARGO].copy()
    final_model = fit(best_config, final_development)
    final_test["prediction"] = final_model.predict(final_test[FEATURES])
    final_test["naive_prediction"] = final_test["return_1m"].fillna(0.0)

    metrics = {
        "best_config": asdict(best_config),
        "features": FEATURES,
        "development_rows": int(len(final_development)),
        "final_holdout_rows": int(len(final_test)),
        "final_holdout_weighted_pearson": weighted_pearson(final_test["target"], final_test["prediction"], final_test["weight"]),
        "final_holdout_naive_weighted_pearson": weighted_pearson(
            final_test["target"], final_test["naive_prediction"], final_test["weight"]
        ),
        "final_holdout_scored_rows": int(len(final_test)),
        "evaluation_note": "The final holdout is retrospective because it was used in an earlier exploratory run; model selection in this version used only the three expanding validation folds.",
        **source_metadata,
        **split_metadata,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    joblib.dump(final_model, OUTPUT_DIR / "model.joblib")
    final_test[["timestamp", "asset_id", "target", "prediction", "naive_prediction", *FEATURES]].head(1_000).to_csv(
        OUTPUT_DIR / "test_prediction_sample.csv", index=False
    )
    write_report(selection, metrics, final_test)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
