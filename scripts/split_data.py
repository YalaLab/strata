import pandas as pd
import argparse
import random
import os
import json

# Example:
# python scripts/split_data.py --dataset example_data/random_medical_data.csv --columns "Accession Number" --train 0.8 --val 0.1 --test 0.1 --seed 0 --output_dir example_data/

parser = argparse.ArgumentParser(
    description="Read and split a CSV or Json dataset into train, validation, and test sets"
)

parser.add_argument(
    "--dataset", type=str, required=True, help="Path to the full dataset file"
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

filetype = args.dataset.split(".")[-1]

random.seed(args.seed)
os.makedirs(args.output_dir, exist_ok=True)
suffix = f"_{args.train}_{args.val}_{args.test}_{args.seed}"

train_path = os.path.join(args.output_dir, f"train{suffix}.{filetype}")
if not args.overwrite:
    assert not os.path.exists(train_path), f"File already exists: {train_path}"
val_path = os.path.join(args.output_dir, f"val{suffix}.{filetype}")
test_path = os.path.join(args.output_dir, f"test{suffix}.{filetype}")

if filetype == "csv":
    df = pd.read_csv(args.dataset)

    columns = args.columns.split(",")
    columns = [col.strip() for col in columns]
    df = df[df[columns].notna().any(axis=1)].reset_index(drop=True)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    train_end = int(args.train * len(df))
    val_end = train_end + int(args.val * len(df))

    train_dataset = df.iloc[:train_end]
    val_dataset = df.iloc[train_end:val_end]
    test_dataset = df.iloc[val_end:]

    train_dataset.to_csv(train_path, index=False)
    val_dataset.to_csv(val_path, index=False)
    test_dataset.to_csv(test_path, index=False)
elif filetype == "json":
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    random.shuffle(dataset)
    train_end = int(args.train * len(dataset))
    val_end = train_end + int(args.val * len(dataset))

    train_dataset = dataset[:train_end]
    val_dataset = dataset[train_end:val_end]
    test_dataset = dataset[val_end:]

    def save_json(data, filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    save_json(train_dataset, train_path)
    save_json(val_dataset, val_path)
    save_json(test_dataset, test_path)
else:
    raise Exception(f"File type {filetype} not supported")


print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")
