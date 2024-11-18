import datasets
from typing import Any
import random


def get_mixed_dataset(dataset: datasets.Dataset, data: Any):
    data_mixin = data.get("data_mixin", None)
    mixin_ratio = data.get("mixin_ratio", 1.0)
    mixin_seed = data.get("mixin_seed", None)
    mixin_split = data.get("mixin_split", "train")

    if data_mixin is not None and mixin_ratio > 0.0:
        mixin_num = int(len(dataset) * mixin_ratio)
        mixin_dataset = datasets.load_dataset(data_mixin, split=mixin_split)

        dataset_list = dataset.to_list()
        mixin_dataset_list = mixin_dataset.to_list()

        rng = random.Random(mixin_seed)

        mixed_dataset_list = dataset_list + rng.sample(mixin_dataset_list, mixin_num)
        rng.shuffle(mixed_dataset_list)

        print(f"Dataset length before mixing: {len(dataset)}")
        dataset = dataset.from_list(mixed_dataset_list)
        print(f"Dataset length after mixing: {len(dataset)}")

    return dataset
