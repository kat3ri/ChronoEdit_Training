# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gemma 3 embedding extraction script.

Drop-in replacement for ``scripts/extract_umt5.py``. Reads a metadata CSV
with ``video`` and ``prompt`` columns and saves Gemma 3 embeddings as
pickle files (shape ``[512, 4096]`` per prompt in bfloat16).

Usage::

    python scripts/extract_gemma3.py --csv_path <metadata.csv>

The script creates a ``gemma3/`` subdirectory next to the CSV and updates
the CSV with a ``gemma3`` column pointing to the saved files.  Downstream,
update the dataset config to load from ``gemma3`` instead of ``umt5``.

Since the output dimension (4096) matches UMT5-XXL, the same
``t5_text_embeddings`` input key can be reused without any network changes.
"""

import csv
import pickle
from pathlib import Path
import argparse

import torch
from chronoedit._src.modules.gemma3 import get_gemma3_embedding


def extract_and_save_gemma3_embeddings(csv_path: str) -> None:
    """Extract Gemma 3 embeddings from captions in the metadata CSV file.

    Args:
        csv_path: Path to the metadata CSV file.
    """
    csv_path = Path(csv_path)
    csv_dir = csv_path.parent

    gemma3_dir = csv_dir / "gemma3"
    gemma3_dir.mkdir(exist_ok=True)

    print(f"Reading metadata from: {csv_path}")
    print(f"Gemma 3 embeddings will be saved to: {gemma3_dir}")

    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    print(f"Found {len(rows)} entries in metadata.csv")

    if "gemma3" not in fieldnames:
        fieldnames = list(fieldnames) + ["gemma3"]

    for idx, row in enumerate(rows):
        video_name = row["video"]
        caption = row["prompt"]

        video_basename = Path(video_name).stem
        gemma3_filename = f"{video_basename}.pkl"
        gemma3_path = gemma3_dir / gemma3_filename

        print(f"[{idx + 1}/{len(rows)}] Processing: {video_name}")
        print(
            f"  Caption: {caption[:100]}..."
            if len(caption) > 100
            else f"  Caption: {caption}"
        )

        try:
            embeddings = get_gemma3_embedding(caption)[0].to(dtype=torch.bfloat16).cpu()

            with open(gemma3_path, "wb") as f:
                pickle.dump(embeddings, f)

            row["gemma3"] = f"gemma3/{gemma3_filename}"
            print(f"  Saved to: {row['gemma3']}")

        except Exception as e:
            print(f"  Error processing caption: {e}")
            row["gemma3"] = ""

    print(f"\nWriting updated metadata to: {csv_path}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Processed {len(rows)} captions.")
    print(f"Gemma 3 embeddings saved in: {gemma3_dir}")
    print(f"Updated metadata file: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Gemma 3 embeddings from captions in metadata CSV"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        help="Path to the metadata CSV file",
        required=True,
    )
    args = parser.parse_args()
    extract_and_save_gemma3_embeddings(args.csv_path)


if __name__ == "__main__":
    main()
