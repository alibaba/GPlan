"""
CSV Dataset Loader
Load training data from local CSV files.
"""

import torch
import pandas as pd
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    """
    Load training data from a local CSV file.
    Each row is transformed into model input format via collate_fn.
    """

    def __init__(self, csv_path, collate_fn=None):
        self.collate_fn = collate_fn

        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        print(f"Loaded {len(df)} rows from {csv_path}")

        self.data = []
        skipped = 0
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            if self.collate_fn is not None:
                processed = self.collate_fn(row_dict)
                if processed is not None:
                    self.data.append(processed)
                else:
                    skipped += 1
            else:
                self.data.append(row_dict)

        print(f"CSVDataset: {len(self.data)} valid samples, {skipped} skipped")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
