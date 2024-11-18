import pandas as pd
import argparse
import random
import os
# Example:
# python split_data.py --dataset_csv /mnt/shareddata/datasets/strata_datav2/mds/full.csv --train 0.4 --val 0.3 --test 0.3 --seed 0 --output_dir /mnt/shareddata/datasets/strata_datav2/mds/

parser = argparse.ArgumentParser(
    description="Read and split a CSV dataset into train, validation, and test sets"
)

parser.add_argument(
    "--dataset_csv", type=str, required=True, help="Path to the CSV file"
)
parser.add_argument(
    "--train", type=float, required=True, help="Proportion of data to use for training"
)
parser.add_argument(
    "--val", type=float, required=True, help="Proportion of data to use for validation"
)
parser.add_argument(
    "--test", type=float, required=True, help="Proportion of data to use for testing"
)
parser.add_argument(
    "--seed", type=int, default=0, help="Random seed for reproducibility"
)
parser.add_argument(
    "--output_dir", type=str, required=True, help="Path to the output dir"
)
parser.add_argument(
    "--columns",
    type=str,
    required=True,
    help="Comma separated string of columns. At least one of them needs to be non-N/A for a sample to be valid.",
)
parser.add_argument(
    "--overwrite", action="store_true", help="Allow overwriting of existing files"
)

args = parser.parse_args()

if args.train + args.val + args.test != 1.0:
    raise ValueError("Train, val, and test proportions must sum to 1")

df = pd.read_csv(args.dataset_csv)

random.seed(args.seed)

columns = args.columns.split(",")
columns = [col.strip() for col in columns]
df = df[df[columns].notna().any(axis=1)].reset_index(drop=True)
df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

train_end = int(args.train * len(df))
val_end = train_end + int(args.val * len(df))

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

suffix = f"_{args.train}_{args.val}_{args.test}_{args.seed}"

os.makedirs(args.output_dir, exist_ok=True)

train_path = os.path.join(args.output_dir, f"train{suffix}.csv")
if not args.overwrite:
    assert not os.path.exists(train_path), f"File already exists: {train_path}"
train_df.to_csv(train_path, index=False)

val_path = os.path.join(args.output_dir, f"val{suffix}.csv")
val_df.to_csv(val_path, index=False)

test_path = os.path.join(args.output_dir, f"test{suffix}.csv")
test_df.to_csv(test_path, index=False)
