import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

from .utils.logging import logger


def init_peft_model(
    model_name="mistralai/Mistral-7B-v0.1",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=False,
    lora_r=16,
    lora_scaling=1,
    lora_dropout=0,
    random_state=3407,
    use_gradient_checkpointing=True,
    **kwargs,
):
    """_summary_

    Args:
        max_seq_length (int, optional): Choose any! We auto support RoPE Scaling internally!. Defaults to 2048.
        dtype (_type_, optional): None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+. Defaults to None.
        load_in_4bit (bool, optional): Use 4bit quantization to reduce memory usage. Can be False.. Defaults to False.
        load_in_4bit (bool, optional): _description_. Defaults to False.
        lora_r (int, optional): Choose any number > 0 ! Suggested 8, 16, 32, 64, 128. Defaults to 16.
        lora_scaling (int, optional): Lora alpha coefficient is lora_scaling * lora_r. Defaults to 1.
        lora_dropout (int, optional): Supports any, but = 0 is optimized. Defaults to 0.
        random_state (int, optional): _description_. Defaults to 3407.
    """

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,  #
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_scaling * lora_r,
        lora_dropout=lora_dropout,  #
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
        **kwargs,
    )

    return model, tokenizer


def get_chat_template(model_name, tokenizer):
    if "llama-3" in model_name.lower():
        logger.info("Using Llama 3 chat template.")
        # This is the template for Llama 3. We include this because it's needed when non-instruction-tuned models are used.
        chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        eos_suffix = "<|eot_id|>"
        response_beginning = (
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        response_end = "<|eot_id|>"
    elif "gemma-2" in model_name.lower():
        logger.info("Using Gemma 2 chat template.")
        # Reference: https://huggingface.co/google/gemma-2-9b-it
        chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"

        # We need to remove eos_suffix for inference (we need to generate after the initial prompt).
        eos_suffix = "<end_of_turn>\n"
        response_beginning = "<start_of_turn>model\n"
        response_end = "<end_of_turn>\n"
    else:
        if not ("mistral" in model_name.lower() or "mixtral" in model_name.lower()):
            assert (
                tokenizer.chat_template is None
            ), "An instruction-tuned model with unsupported chat template is detected. You can remove this assertion if you want to use mistral template, which is likely not optimal if the instruction tuning is not with mistral template."
            logger.warning(
                "A non-instruction-tuned model is detected. Using mistral template."
            )

        logger.info("Using Mistral chat template.")
        # This is the template for mistral. We include this because it's needed when non-instruction-tuned models are used.
        chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

        # We need to remove eos_suffix for inference (we need to generate after the initial prompt).
        eos_suffix = tokenizer.eos_token
        response_beginning = "[/INST]"
        response_end = "</s>"

    return dict(
        chat_template=chat_template,
        response_beginning=response_beginning,
        response_end=response_end,
        eos_suffix=eos_suffix,
    )


def init_trainer(
    model,
    save_path,
    tokenizer,
    dataset,
    epochs,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    max_seq_length=4096,
    learning_rate=2e-4,
    seed=3047,
):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=5,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            save_total_limit=1,
            save_strategy="epoch",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=seed,
            output_dir=save_path,
        ),
        dataset_kwargs=dict(
            # Seems like the double bos problem was already handled in unsloth's patch, but still setting to False to be safe
            # https://github.com/unslothai/unsloth/blob/933d9fe2cb2459f949ee2250e90a5b610d277eab/unsloth/tokenizer_utils.py#L928
            add_special_tokens=False
        ),
    )

    return trainer


def formatting_prompts_func(examples, tokenizer, chat_template):
    texts = []

    # TODO: implement truncation if messages are longer than the context length.

    # Must add EOS_TOKEN, otherwise your generation will go on forever! This is included in the template.
    # EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

    for messages in examples["messages"]:
        # the chat_template should have EOS token (following the original notebook)
        text = tokenizer.apply_chat_template(
            messages,
            chat_template=chat_template,
            tokenize=False,
            add_special_tokens=True,
        )
        # For debugging:
        # print(f"**{text}**")
        texts.append(text)

    return {
        "text": texts,
    }
