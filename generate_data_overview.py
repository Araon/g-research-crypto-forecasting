"""Create an honest visual introduction to the minute-level source data."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SOURCE = Path("data/full_data__3__2018.csv")
DESTINATION = Path("outputs/data_overview.png")


def main() -> None:
    data = pd.read_csv(SOURCE).rename(columns=str.lower)
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="s", utc=True)

    # A single day is readable at screen size while still showing that this is
    # minute-level data, not a daily stock-price series.
    first_day = data.loc[data["timestamp"] < data["timestamp"].min() + pd.Timedelta(days=1)].copy()
    # Reindex to a real one-minute clock. Missing source rows remain NaN, so
    # Matplotlib breaks the line instead of drawing an invented bridge across a
    # gap in exchange data.
    first_day = first_day.set_index("timestamp").asfreq("min").reset_index()

    plt.style.use("dark_background")
    figure, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, facecolor="#111827")
    axes[0].plot(first_day["timestamp"], first_day["close"], color="#38bdf8", linewidth=1.2)
    axes[0].set_title("One day of Cardano minute bars from the public G-Research mirror")
    axes[0].set_ylabel("close price (USD)")

    axes[1].fill_between(first_day["timestamp"], first_day["volume"], color="#22c55e", alpha=0.65)
    axes[1].set_ylabel("volume")

    axes[2].plot(first_day["timestamp"], first_day["target"], color="#f59e0b", linewidth=0.9)
    axes[2].axhline(0, color="#94a3b8", linewidth=0.8)
    axes[2].set_ylabel("future-return target")
    axes[2].set_xlabel("UTC time")

    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    DESTINATION.parent.mkdir(exist_ok=True)
    figure.savefig(DESTINATION, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())


if __name__ == "__main__":
    main()
