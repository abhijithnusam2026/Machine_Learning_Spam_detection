"""Fine-tune Qwen2.5-1.5B-Instruct with LoRA or QLoRA.

The script is intentionally explicit instead of hiding everything behind Trainer defaults.
That makes it easier to discuss the ML engineering decisions in a portfolio review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training.collator import CausalLMCollator


def load_lora_sweep_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def tokenize_record(
    record: dict[str, Any],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> dict[str, Any]:
    messages = record["messages"]
    prompt_messages = [message for message in messages if message["role"] != "assistant"]

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    full_ids = full_ids[:max_length]
    prompt_length = min(len(prompt_ids), len(full_ids))

    labels = [-100] * prompt_length + full_ids[prompt_length:]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def prepare_dataset(path: Path, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    dataset = load_dataset("json", data_files=str(path), split="train")
    tokenized = dataset.map(
        lambda record: tokenize_record(record, tokenizer, max_length),
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {path}",
    )
    return tokenized


def build_model(
    model_name: str,
    use_qlora: bool,
    gradient_checkpointing: bool,
) -> AutoModelForCausalLM:
    quantization_config = None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        attn_implementation="sdpa",
    )
    if use_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def train_one_run(
    args: argparse.Namespace,
    sweep_config: dict[str, Any],
    rank: int,
    alpha: int,
) -> None:
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(sweep_config["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = prepare_dataset(args.train_file, tokenizer, args.max_length)
    eval_dataset = prepare_dataset(args.validation_file, tokenizer, args.max_length)

    model = build_model(
        model_name=sweep_config["model_name"],
        use_qlora=args.use_qlora,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=sweep_config["dropout"],
        target_modules=sweep_config["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    run_name = f"qwen25-1p5b-r{rank}-a{alpha}-{'qlora' if args.use_qlora else 'lora'}"
    output_dir = args.output_dir / run_name

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        report_to=["wandb"] if args.wandb_project else [],
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        fp16=args.fp16,
        optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
        fsdp=args.fsdp if args.fsdp else "",
        fsdp_config={
            "activation_checkpointing": args.gradient_checkpointing,
            "use_orig_params": True,
        }
        if args.fsdp
        else None,
    )

    if args.wandb_project:
        training_args.report_to = ["wandb"]
        training_args.run_name = run_name

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalLMCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/phase1/train.jsonl"))
    parser.add_argument(
        "--validation-file", type=Path, default=Path("data/processed/phase1/validation.jsonl")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/phase1"))
    parser.add_argument("--sweep-config", type=Path, default=Path("configs/lora_sweep.json"))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--rank", type=int, default=None, help="Override sweep ranks with one rank.")
    parser.add_argument("--alpha", type=int, default=None, help="Override sweep alphas with one alpha.")
    parser.add_argument("--use-qlora", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fsdp", default="", help='Example: "full_shard auto_wrap"')
    parser.add_argument("--wandb-project", default="", help="Set to enable W&B reporting.")
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.wandb_project:
        import os

        os.environ["WANDB_PROJECT"] = args.wandb_project

    sweep_config = load_lora_sweep_config(args.sweep_config)
    ranks = [args.rank] if args.rank else sweep_config["ranks"]
    alphas = [args.alpha] if args.alpha else sweep_config["alphas"]

    for rank in ranks:
        for alpha in alphas:
            train_one_run(args=args, sweep_config=sweep_config, rank=rank, alpha=alpha)


if __name__ == "__main__":
    main()
