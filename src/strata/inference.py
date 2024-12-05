import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import os
import math
from collections import defaultdict
import json

from .llm import get_chat_template
from .template import get_model_inputs, get_task_info
from .utils.logging import logger
from .utils.parsing import load_function_from_file, load_config_simple
from scipy.stats import bootstrap


DISABLE_UNSLOTH_INFERENCE = (
    os.environ.get("DISABLE_UNSLOTH_INFERENCE", "False").lower() == "true"
)
if DISABLE_UNSLOTH_INFERENCE:
    print("Using transformers for inference (unsloth inference disabled)")
    from transformers import AutoModelForCausalLM, AutoTokenizer
else:
    from unsloth import FastLanguageModel


def calculate_ci(series):
    ci = bootstrap(
        (np.array(series),),
        np.mean,
        n_resamples=5000,
        confidence_level=0.95,
    ).confidence_interval
    return np.round(ci.low * 100, 1), np.round(ci.high * 100, 1)


def lazy_init(fn):
    def wrapper():
        if not hasattr(wrapper, "fn_output"):
            wrapper.fn_output = fn()
        return wrapper.fn_output

    return wrapper


def predict(
    report_text,
    args,
    template,
    parse_response,
    verbose=False,
    *,
    model,
    tokenizer,
    chat_template,
    **kwargs,
):
    model_inputs = get_model_inputs(
        report_text,
        tokenizer=tokenizer,
        template=template,
        chat_template=chat_template,
        max_length=args.test.get("max_length", 7000),
        verbose=verbose,
    )
    responses = get_responses(model_inputs, model, tokenizer, **kwargs)

    # Extract response
    responses = [
        response.split(chat_template["response_beginning"])[-1]
        .split(chat_template["response_end"])[0]
        .strip()
        for response in responses
    ]

    if verbose:
        print(responses)
    response = responses[0]
    answers = parse_response(
        response, response_start=template["response_start"], verbose=verbose
    )

    return answers, response


@torch.no_grad()
def get_responses(
    model_inputs,
    model,
    tokenizer,
    max_new_tokens=256,
    top_p=0.5,
    temperature=0.5,
    **extra,
):
    model_inputs, attention_mask = model_inputs
    np.random.seed(100)
    torch.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    generated_ids = model.generate(
        model_inputs,
        tokenizer=tokenizer,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        num_beams=1,
        num_return_sequences=1,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        stop_strings=["\n\n"],
    )
    responses = tokenizer.batch_decode(generated_ids)

    return responses


def save_outputs(outputs, outputs_path, verbose=True):
    outputs_df = pd.DataFrame(outputs)
    outputs_df.to_csv(outputs_path, index=False)

    if verbose:
        logger.info(f"Saved outputs to {outputs_path}")


def init_model(model_path, load_in_4bit, max_seq_length):
    if model_path == "gpt4":
        raise ValueError(
            "GPT-4 inference is not supported. Please generate the responses manually and run eval."
        )

    if not DISABLE_UNSLOTH_INFERENCE:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            load_in_4bit=load_in_4bit,
            max_seq_length=max_seq_length,
        )
        FastLanguageModel.for_inference(model)
    else:
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
            original_model_path = adapter_config["base_model_name_or_path"]
            model = AutoModelForCausalLM.from_pretrained(
                original_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            model.load_adapter(model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            original_model_path = model_path
        tokenizer = AutoTokenizer.from_pretrained(
            original_model_path,
        )

        print(f"Loaded model from {model_path} (base model: {original_model_path})")

    chat_template = get_chat_template(model.config._name_or_path, tokenizer)

    return {"model": model, "tokenizer": tokenizer, "chat_template": chat_template}


def is_valid(x):
    if x is None:
        return False
    if isinstance(x, str):
        return x != ""
    return not math.isnan(x)


def inference(args, inference_df: pd.DataFrame, save_every: int = 10, save_suffix=""):
    save_path = os.path.join(args.save_path, args.exp_name)

    templates = load_config_simple(args.templates_path)
    os.makedirs(save_path, exist_ok=True)
    outputs_path = f"{save_path}/inference{save_suffix}.csv"
    questions = (
        args.data.inference_questions
        if "inference_questions" in args.data
        else args.data.questions
    )

    if os.path.exists(outputs_path):
        logger.info(f"Loading existing outputs from {outputs_path}")
        outputs = pd.read_csv(outputs_path)
        assert (
            "Accession Number" in outputs.columns
        ), "Accession Number column not found in existing outputs"
        assert set(outputs["Accession Number"]) == set(
            inference_df["Accession Number"]
        ), "Accession Numbers do not match"
    else:
        outputs = inference_df.copy()

        if args.save_only_necessary_cols:
            columns = list(
                set(
                    sum(
                        [
                            templates[question_name]["gt_columns"]
                            for question_name in questions
                        ],
                        [],
                    )
                )
            )
            outputs = outputs[
                [
                    "Accession Number",
                    args.data.get("report_text_col", "Report Text"),
                    *columns,
                ]
            ]

    # Turn outputs into list of dict
    outputs = outputs.to_dict(orient="records")

    if args.load_fine_tuned_model is not None:
        assert (
            args.inference_mode != "zero-shot"
        ), "mode cannot be zero-shot if load_fine_tuned_model is set"
        model_name = args.load_fine_tuned_model
        logger.info(f"Using model {model_name} (load fine-tuned model)")
    else:
        if args.inference_mode == "fine-tuned":
            model_name = save_path
            logger.info(
                f"Using model {model_name} (loading finetuned model from the default path)"
            )
        elif args.inference_mode == "zero-shot":
            model_name = args.model.model_name
            logger.info(f"Using model {model_name} (zero-shot)")
        else:
            raise ValueError(
                "Either --load-fine-tuned-model [model path] or --inference-mode [zero-shot/fine-tuned] must be set."
            )

    get_model_dict = lazy_init(
        lambda: init_model(
            model_name, args.model.load_in_4bit, args.trainer.max_seq_length
        )
    )

    for question_name in questions:
        task_dict = templates[question_name]
        task_template = get_task_info(task_dict)

        parse_response = load_function_from_file(
            file_path=task_dict["parse_response"], function_name="parse_response"
        )

        # Note that it's possible that gt_columns do not exist (e.g. for inference)
        # We still save to pred_columns
        gt_columns = task_dict["gt_columns"]
        pred_columns = [col + "_pred" for col in gt_columns]
        gt_to_pred_column_map = {
            gt_col: pred_col for gt_col, pred_col in zip(gt_columns, pred_columns)
        }

        pbar = tqdm(outputs, total=len(outputs))
        for i, output in enumerate(pbar):
            # if we have response, and all pred_columns are already filled, skip this sample
            if output.get(f"{question_name}_response", "") and all(
                [is_valid(output.get(col)) for col in pred_columns]
            ):
                logger.debug(
                    f"Skipping sample with accession number {output['Accession Number']}"
                )
                continue
            report_text = output[args.data.get("report_text_col", "Report Text")]

            answers, response = predict(
                report_text,
                args,
                task_template,
                parse_response,
                verbose=args.verbose,
                **get_model_dict(),
                **args.test,
            )

            output[f"{question_name}_response"] = response

            if answers is None:
                continue

            try:
                if isinstance(answers, tuple) or isinstance(answers, list):
                    # answers should be in list or tuple format
                    assert (
                        len(answers) == len(gt_columns)
                    ), f"Expected {len(gt_columns)} answers, got {len(answers)}. GT columns: {gt_columns}, answers: {answers}"
                    for pred_column, answer in zip(pred_columns, answers):
                        output[pred_column] = answer
                else:
                    # answers should be in dict format and should match gt_columns
                    for gt_column, pred_column in gt_to_pred_column_map.items():
                        output[pred_column] = answers[gt_column]
            except (AssertionError, KeyError) as e:
                logger.error(
                    f"Error processing answers for sample with accession number {output['Accession Number']}: {type(e)}: {e}"
                )
                continue

            if i % save_every == 0:
                save_outputs(outputs, outputs_path, verbose=False)
        save_outputs(outputs, outputs_path)


def evaluate(
    args,
    inference_df: pd.DataFrame,
    save_every: int = 10,
    save_suffix="",
    calculate_confidence_intervals=False,
):
    save_path = os.path.join(args.save_path, args.exp_name)

    templates = load_config_simple(args.templates_path)
    inference_file_path = f"{save_path}/inference{save_suffix}.csv"
    outputs_path = f"{save_path}/eval{save_suffix}.csv"

    # Eval does not support resuming since it should be fast (no LLM involved)
    logger.info(f"Loading inference results from {inference_file_path}")
    inference_results_df = pd.read_csv(inference_file_path)
    assert (
        "Accession Number" in inference_results_df.columns
    ), "Accession Number column not found in existing outputs"
    assert set(inference_results_df["Accession Number"]) == set(
        inference_df["Accession Number"]
    ), "Accession Numbers do not match"

    # inference_results_df contains the old gt values
    # merge with the new inference_df, which has nans where appropriate

    all_gt_columns = ["Accession Number", "Report Text"]
    all_pred_columns = ["Accession Number", "Report Text"]

    questions = (
        args.data.inference_questions
        if "inference_questions" in args.data
        else args.data.questions
    )
    for question_name in questions:
        task_dict = templates[question_name]

        gt_columns = task_dict["gt_columns"]
        all_gt_columns += gt_columns
        all_pred_columns += [col + "_pred" for col in gt_columns]

    inference_results_df = inference_results_df[all_pred_columns].merge(
        inference_df[all_gt_columns], on=["Accession Number", "Report Text"]
    )
    assert (
        len(inference_results_df.index) == len(inference_df.index)
    ), f"Expected {len(inference_df.index)} but got {len(inference_results_df.index)} rows."

    # Turn inference_results into list of dict
    inference_results = inference_results_df.to_dict(orient="records")

    # By default, we only carry the accession number from inference results
    outputs = [
        {"Accession Number": inference_result["Accession Number"]}
        for inference_result in inference_results
    ]

    for question_name in questions:
        task_dict = templates[question_name]

        # Note that it's possible that gt_columns do not exist (e.g. for inference)
        # We still save to pred_columns
        gt_columns = task_dict["gt_columns"]
        pred_columns = [col + "_pred" for col in gt_columns]
        correct_columns = [col + "_correct" for col in gt_columns]

        pbar = tqdm(inference_results, total=len(inference_results))
        for i, inference_result in enumerate(pbar):
            # if we have response, and all pred_columns are already filled, skip this sample
            if all([is_valid(inference_result.get(col)) for col in correct_columns]):
                logger.info(
                    f"Skipping sample with accession number {inference_result['Accession Number']}"
                )
                continue

            if any([pd.isna(inference_result[col]) for col in gt_columns]):
                continue

            # load pred and gt columns from inference results
            for gt_column, pred_column, correct_column in zip(
                gt_columns, pred_columns, correct_columns
            ):
                outputs[i][gt_column] = inference_result[gt_column]
                outputs[i][pred_column] = inference_result[pred_column]
                outputs[i][correct_column] = int(
                    outputs[i][gt_column] == outputs[i][pred_column]
                )

            outputs[i][f"{question_name}_all_correct"] = int(
                all([outputs[i][correct_column] for correct_column in correct_columns])
            )
            outputs[i][f"{question_name}_avg_correct"] = np.mean(
                [outputs[i][correct_column] for correct_column in correct_columns]
            )

            if i % save_every == 0:
                save_outputs(outputs, outputs_path, verbose=False)
        save_outputs(outputs, outputs_path)

    for output in outputs:
        output["all_correct"] = int(
            all(
                [
                    output.get(f"{question_name}_all_correct", True)
                    for question_name in questions
                ]
            )
        )
    outputs_df = pd.DataFrame(outputs)

    metrics = args.get("metrics", ["F1", "Precision", "Recall"])

    scores_by_question = []
    scores_by_column = defaultdict(dict)
    for question_name in questions:
        question_scores = {
            "Question": question_name,
            "All Correct": outputs_df[f"{question_name}_all_correct"].mean() * 100.0,
        }

        for metric in metrics:
            calculate_metric = load_function_from_file(
                file_path="src/strata/utils/metrics.py", function_name=metric
            )
            col_scores = {}
            for col in templates[question_name]["gt_columns"]:
                metric_value = calculate_metric(
                    col, col + "_pred", inference_results_df
                )
                col_scores[col] = metric_value

                scores_by_column[col][metric] = metric_value

                scores_by_column[col]["All Correct"] = (
                    outputs_df[col + "_correct"].mean() * 100
                )
                scores_by_column[col]["Question"] = col
                scores_by_column[col]["Category"] = question_name

            question_scores[metric] = sum(col_scores.values()) / len(col_scores)
            col_scores["Category"] = question_name

        scores_by_question.append(question_scores)

        if calculate_confidence_intervals:
            scores_by_question[-1]["All Correct CI"] = calculate_ci(
                outputs_df[f"{question_name}_all_correct"]
            )

    overall_metrics = {
        "Question": "overall",
        "All Correct": outputs_df["all_correct"].mean() * 100.0,
    }

    for metric in metrics:
        overall_metrics[metric] = np.array(
            [overall_score[metric] for overall_score in scores_by_question]
        ).mean()

    scores_by_question.append(overall_metrics)

    if calculate_confidence_intervals:
        scores_by_question[-1]["All Correct CI"] = calculate_ci(
            outputs_df["all_correct"]
        )

    with pd.option_context("display.precision", 1):
        scores_df = pd.DataFrame(scores_by_question)
        print(scores_df[["Question", "All Correct"] + metrics])
        with open(f"{save_path}/scores{save_suffix}.csv", "w") as f:
            f.write(scores_df.to_csv(index=False))
    print(
        "Note: for 'All Correct', the 'overall' metric is exact match over all questions. For other columns, the metric is averaged over questions.\n"
    )

    scores_df = pd.DataFrame(
        scores_by_column.values(),
        columns=["Question", "Category", "All Correct"] + metrics,
    )
    with pd.option_context("display.precision", 1):
        print(scores_df)
        with open(f"{save_path}/scores_per_column{save_suffix}.csv", "w") as f:
            f.write(scores_df.to_csv(index=False))
