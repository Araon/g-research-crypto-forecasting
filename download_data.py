"""Download the public G-Research Crypto Forecasting data used by the model."""

from pathlib import Path

import kagglehub


DATASET = "yamqwe/cryptocurrency-extra-data-cardano"
FILE_NAME = "full_data__3__2018.csv"


def main() -> None:
    destination = Path(__file__).resolve().parent / "data"
    path = kagglehub.dataset_download(DATASET, path=FILE_NAME, output_dir=str(destination))
    print(path)


if __name__ == "__main__":
    main()
