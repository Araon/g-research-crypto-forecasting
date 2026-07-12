"""Fetch the canonical G-Research training files from archive version 35."""

from pathlib import Path

import kagglehub


DATASET = "yamqwe/cryptocurrency-extra-data-cardano/versions/35"
FILES = [
    "orig_train.jay",
    "orig_asset_details.jay",
    "orig_example_test.jay",
    "orig_example_sample_submission.jay",
]


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "data" / "official"
    for file_name in FILES:
        print(f"Downloading {file_name}...")
        path = kagglehub.dataset_download(DATASET, path=file_name, output_dir=str(output_dir))
        print(path)


if __name__ == "__main__":
    main()
