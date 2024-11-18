import torch
import os
import json
from .utils.unsloth_patch import patch_unsloth

patch_unsloth()
from .preprocess import (  # noqa
    preprocess_data,
    preprocessed_samples_to_dataset,
    load_data_for_inference,
)
from .train import train  # noqa
from .inference import inference, evaluate  # noqa
from .utils.parsing import parse_args  # noqa
from .utils.logging import set_loglevel, logger  # noqa
from .utils.data_mixin import get_mixed_dataset  # noqa


def main(args):
    torch.set_float32_matmul_precision("high")

    if args.command in ["train", "preprocess", "train_with_preprocessed_json"]:
        if args.command in ["train", "preprocess"]:
            preprocessed_json_path = os.path.join(
                args.save_path, "preprocessed_train_data.json"
            )
            # Note that the preprocessed data can be directly loaded through huggingface's `datasets.load_dataset` to train with other libraries.
            preprocessed_samples = preprocess_data(
                data_path=args.data.train_set_data_path,
                templates_path=args.templates_path,
                save_json_path=preprocessed_json_path,
                **args.data,
            )
        elif args.command == "train_with_preprocessed_json":
            assert os.path.exists(
                preprocessed_json_path
            ), f"Preprocessed JSON {preprocessed_json_path} does not exist"
            with open(preprocessed_json_path, "r") as f:
                preprocessed_samples = json.load(f)
        if args.command in ["train", "train_with_preprocessed_json"]:
            dataset = preprocessed_samples_to_dataset(
                preprocessed_samples, columns=["messages"]
            )
            # This is used when data_mixin is used.
            dataset = get_mixed_dataset(dataset, args.data)
            train(args, dataset)
    else:
        if args.get("use_test_set", False):
            data_path = args.data.test_set_data_path
            logger.info(f"Using test set {data_path} for inference.")
            save_suffix = "_test_set"
        else:
            data_path = args.data.val_set_data_path
            logger.info(f"Using val set {data_path} for inference.")
            save_suffix = ""

        if args.command == "test":
            inference_df = load_data_for_inference(
                data_path=data_path,
                to_dataset=False,
                start_idx=args.data.start_idx,
                end_idx=args.data.end_idx,
            )
            inference(args, inference_df, save_suffix=save_suffix)
            evaluate(args, inference_df, save_suffix=save_suffix)
        elif args.command == "inference":
            inference_df = load_data_for_inference(
                data_path=data_path,
                to_dataset=False,
                start_idx=args.data.start_idx,
                end_idx=args.data.end_idx,
            )
            inference(args, inference_df, save_suffix=save_suffix)
        elif args.command == "evaluate":
            inference_df = load_data_for_inference(
                data_path=data_path,
                to_dataset=False,
                start_idx=args.data.start_idx,
                end_idx=args.data.end_idx,
            )
            evaluate(args, inference_df, save_suffix=save_suffix)
        else:
            raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    set_loglevel(debug=True)
    args = parse_args()
    main(args)
