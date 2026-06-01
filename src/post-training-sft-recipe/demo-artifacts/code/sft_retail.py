"""SFT training script for retail multi-turn data.

Designed to run in the curated Azure ML HF NLP image
(`mcr.microsoft.com/azureml/curated/acft-hf-nlp-gpu:117`), which already
ships transformers + peft + accelerate + datasets + bitsandbytes. We add
`trl` and (if needed) update `transformers` at job start.

The input dataset is a JSONL file produced by `generate_sft_multiturn.py`,
where each row has:

    {
      "task_id": ..., "turn_index": ...,
      "input":  {"system": str, "tools": [openai schemas], "messages": [...]},
      "target": {"role": "assistant", "content": str|null, "tool_calls": ...},
      "meta":   {...}
    }

We flatten each row into a chat with the target appended as the last turn
and compute loss only over the target tokens (prompt-masked SFT).

Usage (set via job CLI args):

    python sft_retail.py \
        --train-jsonl /mnt/data/retail_train_sft.jsonl \
        --base-model Qwen/Qwen3-14B \
        --output-dir /mnt/outputs/checkpoints \
        --epochs 5 --batch-size 1 --grad-accum 16 \
        --max-seq-len 4096 --lora-r 64 --lora-alpha 128
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)


def _normalize_messages(msgs):
    """Some chat templates (GLM, Mistral) expect tool_call arguments as dicts
    rather than JSON-encoded strings (OpenAI convention). Convert in place
    so the template doesn't blow up when iterating with .items()."""
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            tc_list = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except Exception:
                        pass
                tc["function"] = fn
                tc_list.append(tc)
            m["tool_calls"] = tc_list
        out.append(m)
    return out


def chat_template_format(row: dict[str, Any], tokenizer) -> dict[str, Any]:
    """Render one SFT row using the model's chat template.

    Returns: {"input_ids": [...], "labels": [...]} with labels=-100 for the
    prompt portion (everything except the final assistant target turn).

    Accepts BOTH input shapes:
      * Legacy dict shape: row["input"] = {"system": str, "messages": [...], "tools": [...]}
      * Modern list shape:       row["input"] = [{role, content}, ...]   (with row["tools"] at top level)
    Roles ``developer`` are normalized to ``system`` for chat templates that
    don't know about the developer role (Qwen3 / GLM-4 / Llama-3).
    """
    inp = row["input"]
    if isinstance(inp, list):
        # Modern list-of-messages shape
        msgs_raw = list(inp)
        tools = row.get("tools")
    else:
        # Legacy dict shape
        sys_msg = inp.get("system")
        msgs_raw = []
        if sys_msg:
            msgs_raw.append({"role": "system", "content": sys_msg})
        msgs_raw.extend(inp.get("messages") or [])
        tools = inp.get("tools")

    # Normalize roles + drop None content on assistant tool-only turns
    fixed = []
    for m in msgs_raw:
        m = dict(m)
        if m.get("role") == "developer":
            m["role"] = "system"
        if m.get("role") == "assistant" and m.get("content") is None:
            m.pop("content", None)
        fixed.append(m)
    messages = _normalize_messages(fixed)

    target_msg = dict(row["target"])
    target_msg = {k: v for k, v in target_msg.items() if v is not None}
    target_msg = _normalize_messages([target_msg])[0]

    tools = tools or None

    prompt_str = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=False,
    )
    full_str = tokenizer.apply_chat_template(
        [*messages, target_msg], tools=tools, add_generation_prompt=False, tokenize=False,
    )

    if full_str.startswith(prompt_str):
        # Additive template — tokenize FULL once (no re-tokenization, no BPE boundary issues).
        full_ids = tokenizer.apply_chat_template(
            [*messages, target_msg], tools=tools, add_generation_prompt=False, tokenize=True,
        )
        prompt_ids = tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=True,
        )
        if len(full_ids) <= len(prompt_ids):
            return {"input_ids": list(full_ids), "labels": [-100] * len(full_ids)}
        labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids):])
        return {"input_ids": list(full_ids), "labels": labels}

    # Non-additive template — fall back to char-level prefix matching and
    # tokenize prefix + target separately.
    n = 0
    for a, b in zip(prompt_str, full_str):
        if a != b:
            break
        n += 1
    prefix_str = full_str[:n]
    target_str = full_str[n:]
    if not target_str:
        full_ids = tokenizer(full_str, add_special_tokens=False)["input_ids"]
        return {"input_ids": list(full_ids), "labels": [-100] * len(full_ids)}
    prefix_ids = tokenizer(prefix_str, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_str, add_special_tokens=False)["input_ids"]
    input_ids = list(prefix_ids) + list(target_ids)
    labels = [-100] * len(prefix_ids) + list(target_ids)
    return {"input_ids": input_ids, "labels": labels}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", default=None)
    p.add_argument("--base-model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-seq-len", type=int, default=6144)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-strategy", default="epoch")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="random hold-out fraction for eval (0 disables)")
    p.add_argument("--val-seed", type=int, default=42)
    p.add_argument("--target-modules", default="qwen",
                   choices=["qwen", "all-linear", "glm"],
                   help="LoRA target modules preset")
    p.add_argument("--limit", type=int, default=0, help="0 = all rows")
    args = p.parse_args()

    print(f"# torch  : {torch.__version__}  cuda={torch.cuda.is_available()}")
    print(f"# device : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    if args.load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_cfg,
        trust_remote_code=True,
        attn_implementation="eager",  # safer default for newer archs (GLM MoE etc.)
    )

    # LoRA adapters
    from peft import LoraConfig, get_peft_model

    target_modules_preset = {
        "qwen":       ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # GLM-4 MoE Lite uses MLA attention: q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj
        "glm":        ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj",
                       "o_proj", "gate_proj", "up_proj", "down_proj"],
        "all-linear": "all-linear",
    }[args.target_modules]

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=target_modules_preset,
    )
    model = get_peft_model(model, lora)
    # Required when combining LoRA (frozen base) with gradient_checkpointing —
    # ensures the embedding output carries requires_grad so the first
    # reentrant checkpoint can backprop into the LoRA adapters.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        # Fallback: register a forward hook on the input embeddings.
        def _make_inputs_require_grad(module, inp, out):
            out.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)
    model.print_trainable_parameters()

    raw = load_dataset("json", data_files=args.train_jsonl, split="train")
    if args.limit:
        raw = raw.select(range(min(args.limit, len(raw))))
    print(f"# train rows: {len(raw)}")

    def _format(example):
        out = chat_template_format(example, tokenizer)
        # Truncate from the LEFT (keep most recent context) so we never lose the target.
        ids = out["input_ids"]; lab = out["labels"]
        if len(ids) > args.max_seq_len:
            keep = args.max_seq_len
            ids = ids[-keep:]
            lab = lab[-keep:]
        out["input_ids"] = ids; out["labels"] = lab
        out["attention_mask"] = [1] * len(ids)
        return out

    cols_to_remove = raw.column_names
    train_ds = raw.map(_format, remove_columns=cols_to_remove, num_proc=4)

    # Filter out rows where the entire target got truncated away (all -100 labels)
    def _has_supervision(ex):
        return any(l != -100 for l in ex["labels"])
    train_ds = train_ds.filter(_has_supervision)
    print(f"# rows after supervision filter: {len(train_ds)}")

    # Random hold-out for evaluation.
    eval_ds = None
    if args.val_frac and args.val_frac > 0 and len(train_ds) >= 5:
        split = train_ds.train_test_split(test_size=args.val_frac, seed=args.val_seed, shuffle=True)
        train_ds = split["train"]
        eval_ds = split["test"]
        print(f"# split: train={len(train_ds)}  eval={len(eval_ds)}  (val_frac={args.val_frac})")

    from transformers import DataCollatorForLanguageModeling

    class PadCollator:
        def __init__(self, pad_id, label_pad=-100):
            self.pad_id = pad_id; self.label_pad = label_pad
        def __call__(self, batch):
            max_len = max(len(b["input_ids"]) for b in batch)
            input_ids = []; labels = []; attn = []
            for b in batch:
                pad = max_len - len(b["input_ids"])
                input_ids.append(b["input_ids"] + [self.pad_id] * pad)
                labels.append(b["labels"] + [self.label_pad] * pad)
                attn.append(b["attention_mask"] + [0] * pad)
            return {
                "input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn),
            }

    collator = PadCollator(tokenizer.pad_token_id)

    args_train = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=2,
        eval_strategy="epoch" if eval_ds is not None else "no",
        report_to="none",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
    )

    # ── AzureML run-context logger (no-op outside an AML job) ────────────────
    class AzureMLCallback(__import__("transformers").TrainerCallback):
        def __init__(self):
            self.run = None
            try:
                from azureml.core.run import Run  # type: ignore
                run = Run.get_context()
                if run is not None and "OfflineRun" not in getattr(run, "id", ""):
                    self.run = run
                    print(f"[azureml_callback] attached to run {run.id}")
            except Exception as exc:
                print(f"[azureml_callback] disabled ({exc})")

        def on_log(self, args_, state, control, logs=None, **kw):
            if not self.run or not logs:
                return
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    try:
                        self.run.log(k, float(v), step=int(state.global_step))
                    except Exception:
                        pass

    from transformers import Trainer
    # transformers >=4.46 uses `processing_class`; older releases use `tokenizer`.
    # Pass via kwargs that work on both 4.51 (Qwen) and 5.x (GLM).
    try:
        trainer = Trainer(
            model=model,
            args=args_train,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            processing_class=tokenizer,
            callbacks=[AzureMLCallback()],
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            args=args_train,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            tokenizer=tokenizer,
            callbacks=[AzureMLCallback()],
        )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("# done. saved to", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
