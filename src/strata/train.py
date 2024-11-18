import torch
from .llm import (
    formatting_prompts_func,
    get_chat_template,
    init_peft_model,
    init_trainer,
)
from functools import partial
import os


def train(args, train_dataset):
    model_name = args.model["model_name"]
    max_seq_length = args.trainer.max_seq_length
    model, tokenizer = init_peft_model(max_seq_length=max_seq_length, **args.model)

    dataset = train_dataset

    chat_template = get_chat_template(model_name, tokenizer)

    mapping = partial(
        formatting_prompts_func,
        tokenizer=tokenizer,
        chat_template=chat_template["chat_template"],
    )
    dataset = dataset.map(mapping, batched=True)

    # gpu stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    save_path = os.path.join(args.save_path, args.exp_name)
    print(f"Model will be saved at {save_path}")

    # training
    trainer = init_trainer(model, save_path, tokenizer, dataset, **args.trainer)
    print("Trainer initialized. Start training.")
    trainer_stats = trainer.train()

    # peak gpu stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # %%
    model.save_pretrained(save_path)  # Local saving
