"""LoRA fine-tuning of Qwen 2.5 Instruct for QuizForge's MCQ-generation task (QLoRA with Unsloth, one GPU).

Run on a CUDA GPU (the Colab notebook does this):
    python -m finetune.train --base unsloth/Qwen2.5-3B-Instruct-bnb-4bit --name quizforge-3b --data data --out exports
    python -m finetune.train --base unsloth/Qwen2.5-7B-Instruct-bnb-4bit --name quizforge-7b --data data --out exports \
                             --batch 1 --grad-accum 8

What it does
  1. loads the 4-bit base model and adds LoRA adapters (rank 16) to every attention and MLP projection;
  2. trains for one epoch on train.jsonl, computing the loss on the assistant reply only (the system prompt, the
     instructions and the material are context, not something to imitate);
  3. reports the validation loss on val.jsonl and a format check: the fine-tuned model writes questions for N validation
     prompts *without* a JSON schema constraint, and each reply is run through the app's validator;
  4. saves the LoRA adapter and a merged q4_K_M GGUF file that Ollama can load (finetune/import_to_ollama.py).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time
from pathlib import Path

from finetune.prepare_data import read_jsonl

RESPONSE_PART = "<|im_start|>assistant\n"
INSTRUCTION_PART = "<|im_start|>user\n"


def format_check(model, tokenizer, examples: list[dict], max_new_tokens: int = 900) -> dict:
    """Generate for validation prompts and validate the replies exactly like the app (no schema constraint here)."""
    from services.question_validator import MalformedOutputError, QuestionValidationError, parse_model_json, validate_question

    ok_json = ok_all = questions = valid = 0
    for ex in examples:
        prompt = tokenizer.apply_chat_template(ex["messages"][:2], tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        reply = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        material = ex["messages"][1]["content"].split("<material>\n")[-1].split("\n</material>")[0]
        n_opts = len(json.loads(ex["messages"][2]["content"])["questions"][0]["options"])
        try:
            data = parse_model_json(reply)
        except MalformedOutputError:
            continue
        ok_json += 1
        items = data.get("questions") if isinstance(data.get("questions"), list) else []
        good = 0
        for raw in items:
            questions += 1
            try:
                validate_question(raw, num_options=n_opts, source_text=material, require_excerpt=True)
                good += 1
            except QuestionValidationError:
                pass
        valid += good
        ok_all += bool(items) and good == len(items)
    n = len(examples) or 1
    return {"prompts": len(examples), "valid_json_rate": ok_json / n, "all_questions_valid_rate": ok_all / n,
            "question_validity_rate": (valid / questions) if questions else None, "questions": questions}


def find_gguf(search_dirs: list[str], started: float) -> str | None:
    hits = [p for d in search_dirs for p in glob.glob(os.path.join(d, "**", "*.gguf"), recursive=True)
            if os.path.getmtime(p) >= started - 5]
    hits = [p for p in hits if "q4_k_m" in os.path.basename(p).lower()] or hits
    return max(hits, key=os.path.getmtime) if hits else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="e.g. unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    ap.add_argument("--name", required=True, help="output model name, e.g. quizforge-3b")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("exports"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--check", type=int, default=20, help="validation prompts for the format check (0 = skip)")
    ap.add_argument("--no-gguf", action="store_true")
    args = ap.parse_args(argv)

    from unsloth import FastLanguageModel, is_bfloat16_supported  # GPU-only imports
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    args.out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=args.base, max_seq_length=args.max_seq, load_in_4bit=True)
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)

    def to_text(rows: list[dict]) -> tuple[Dataset, int]:
        texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False) for r in rows]
        keep = [t for t in texts if len(tokenizer(t)["input_ids"]) <= args.max_seq]  # never train on truncated JSON
        return Dataset.from_dict({"text": keep}), len(texts) - len(keep)

    train_rows, val_rows = read_jsonl(args.data / "train.jsonl"), read_jsonl(args.data / "val.jsonl")
    train_ds, dropped_train = to_text(train_rows)
    val_ds, dropped_val = to_text(val_rows)
    print(f"train {len(train_ds)} examples ({dropped_train} too long, dropped), val {len(val_ds)} ({dropped_val} dropped)")

    config = SFTConfig(
        dataset_text_field="text", max_seq_length=args.max_seq, packing=False,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, learning_rate=args.lr, warmup_ratio=0.03, lr_scheduler_type="linear",
        optim="adamw_8bit", weight_decay=0.01, fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
        logging_steps=10, eval_strategy="steps", eval_steps=100, save_strategy="no", seed=3407,
        output_dir=str(args.out / f"{args.name}-checkpoints"), report_to="none")
    try:
        trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=train_ds, eval_dataset=val_ds, args=config)
    except TypeError:  # older TRL releases call it "tokenizer"
        trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=train_ds, eval_dataset=val_ds, args=config)
    trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)

    base_eval = trainer.evaluate()
    t0 = time.time()
    stats = trainer.train()
    final_eval = trainer.evaluate()
    report = {"name": args.name, "base": args.base, "train_examples": len(train_ds), "val_examples": len(val_ds),
              "epochs": args.epochs, "lr": args.lr, "rank": args.rank, "effective_batch": args.batch * args.grad_accum,
              "train_seconds": round(time.time() - t0), "train_loss": stats.training_loss,
              "val_loss_before": base_eval.get("eval_loss"), "val_loss_after": final_eval.get("eval_loss")}
    print(json.dumps(report, indent=2))

    adapter_dir = args.out / f"{args.name}-lora"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    if args.check:
        FastLanguageModel.for_inference(model)
        report["format_check"] = format_check(model, tokenizer, val_rows[: args.check])
        print("format check (no schema constraint):", json.dumps(report["format_check"], indent=2))

    if not args.no_gguf:
        started = time.time()
        gguf_dir = args.out / f"{args.name}-gguf"
        model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method="q4_k_m")
        found = find_gguf([str(gguf_dir), str(args.out), "."], started)
        if not found:
            raise SystemExit("GGUF export finished but no .gguf file was found - check the export log above.")
        final = args.out / f"{args.name}.Q4_K_M.gguf"
        shutil.move(found, final)
        report["gguf"] = str(final)
        report["gguf_gb"] = round(final.stat().st_size / 1e9, 2)
        print(f"GGUF: {final} ({report['gguf_gb']} GB)")

    (args.out / f"{args.name}-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
