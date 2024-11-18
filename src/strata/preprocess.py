import pandas as pd
import json
import os
from collections import OrderedDict
from datasets import Dataset
from .utils.parsing import load_config_simple, load_function_from_file
from .template import get_task_info
import numpy as np


def read_df(path):
    if path is None:
        return None
    suffix = path.split(".")[-1]
    if suffix == "csv":
        return pd.read_csv(path)
    elif suffix == "xlsx":
        return pd.read_excel(path)
    else:
        raise Exception(f"Unknown file format: {suffix}")


def preprocess_sample(sample, questions_dict, templates_dict, report_text_col):
    dataset = sample.get("dataset", None)
    report_text = sample[report_text_col]

    questions = questions_dict[dataset]
    templates = [templates_dict[question] for question in questions]
    if dataset is not None:
        assert all(
            [template["dataset"] == dataset for template in templates]
        ), f"question mismatches with dataset {[(template['dataset'], question, dataset) for (template, question) in zip(templates, questions)]}"

    preprocessed_samples = []
    for question, template in zip(questions, templates):
        task_info = get_task_info(template)
        question = task_info["question"]
        response_start = task_info["response_start"]
        prompt_framework = task_info["prompt_framework"]
        prompt = prompt_framework.format(report=report_text, question=question)

        gt = OrderedDict([(col, sample[col]) for col in template["gt_columns"]])
        if any([np.isnan(sample[key]) for key in gt]):
            continue

        gt_to_response = load_function_from_file(
            file_path=template["parse_response"], function_name="gt_to_response"
        )
        response = gt_to_response(gt)

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response_start + " " + response},
        ]

        preprocessed_sample = {**sample, "question": question, "messages": messages}

        preprocessed_samples.append(preprocessed_sample)

    return preprocessed_samples


def preprocess_data(
    data_path,
    templates_path,
    report_text_col="Report Text",
    save_json_path=None,
    start_idx=0,
    end_idx=None,
    *,
    questions,
    **extras,
):
    dataset_df = read_df(data_path)
    dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)
    templates_dict = load_config_simple(templates_path)

    if not isinstance(questions, dict):
        # Use `None` as dataset_name disables the dataset name check in the templates (since it's likely a setting with only one dataset)
        dataset_name = None
        questions_dict = {dataset_name: questions}
    else:
        questions_dict = questions

    preprocessed_samples = []
    for _, sample in dataset_df.iterrows():
        preprocessed_samples_for_current_sample = preprocess_sample(
            sample, questions_dict, templates_dict, report_text_col
        )
        preprocessed_samples.extend(preprocessed_samples_for_current_sample)

    # The reason for not using a csv is that json stores list and dict better.
    if save_json_path:
        parent_dir = os.path.dirname(save_json_path)
        os.makedirs(parent_dir, exist_ok=True)
        with open(save_json_path, "w") as f:
            json.dump(preprocessed_samples, f, indent=4)

    if end_idx is not None:
        preprocessed_samples = preprocessed_samples[start_idx:end_idx]
    elif start_idx != 0:
        preprocessed_samples = preprocessed_samples[start_idx:]

    return preprocessed_samples


def preprocessed_samples_to_dataset(preprocessed_samples, columns=None, split=None):
    # pandas takes lists as object dtype
    preprocessed_samples = pd.DataFrame(preprocessed_samples)
    if columns is not None:
        preprocessed_samples = preprocessed_samples[columns]
    dataset = Dataset.from_pandas(preprocessed_samples, split=split)
    return dataset


def load_data_for_inference(
    data_path, to_dataset=True, start_idx=0, end_idx=None, **extra
):
    dataset_df = read_df(data_path)

    if end_idx is not None:
        dataset_df = dataset_df.iloc[start_idx:end_idx]
    elif start_idx != 0:
        dataset_df = dataset_df.iloc[start_idx:]

    if to_dataset:
        dataset = Dataset.from_pandas(dataset_df)

        return dataset
    else:
        return dataset_df
