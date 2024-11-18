# This file is for prompt engineering


def get_task_info(task):
    preamble = task.get(
        "preamble",
        "You are an experienced pathologist. Answer the question using the pathology report below. Base the answer on the report only. Do not add any additional information.\nPathology report:",
    )

    prompt_framework = (
        preamble
        + """

{report}

{question}"""
    )

    question = task["question"]

    # It's very important not to leave a trailing space, as in some tokenizers they will be encoded as a special token.
    response_start = task.get("response_start", "My answer is:")

    return {
        "prompt_framework": prompt_framework,
        "question": question,
        "response_start": response_start,
    }


def get_model_inputs(
    report_text, tokenizer, template, chat_template, max_length, verbose=False
):
    prompt_framework = template["prompt_framework"]
    question = template["question"]
    response_start = template["response_start"]

    chat_template_inference = chat_template["chat_template"]
    eos_suffix = chat_template["eos_suffix"]

    prompt = prompt_framework.format(report=report_text, question=question)
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response_start},
    ]

    if verbose:
        print(
            tokenizer.apply_chat_template(
                messages, chat_template=chat_template_inference, tokenize=False
            ).removesuffix(eos_suffix)
        )
        print(
            tokenizer(
                tokenizer.apply_chat_template(
                    messages, chat_template=chat_template_inference, tokenize=False
                ).removesuffix(eos_suffix),
                add_special_tokens=False,
                return_tensors="pt",
            )
        )

    inputs = tokenizer.apply_chat_template(
        messages, chat_template=chat_template_inference, tokenize=False
    )

    assert (
        inputs[-len(eos_suffix) :] == eos_suffix
    ), f"Suffix not found: the template might mismatch with the defined suffix. inputs: {inputs} ({inputs[-len(eos_suffix):]}), suffix: {eos_suffix}"
    inputs = inputs.removesuffix(eos_suffix)

    # TODO do not hardcode max_length
    inputs = tokenizer(
        inputs,
        add_special_tokens=False,
        return_tensors="pt",
        truncation="longest_first",
        max_length=max_length,
    ).to("cuda")

    # Useful to debug the tokens (especially the special tokens)
    if verbose:
        print(inputs)

    model_inputs = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    return model_inputs, attention_mask
