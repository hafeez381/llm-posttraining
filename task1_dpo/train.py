from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.guards import assert_fixed_files, assert_train_split, sha256_file
from common.logging_utils import append_jsonl, set_seed
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from common.results import finalize, reconcile, start_run
from common.rl_utils import disable_dropout
from common.runmode import add_mode_arg, resolve_models
from task1_dpo.dpo import dpo_loss


def _truncated(tokenizer, prompt, response, max_length) -> bool:
    n_prompt = len(tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True))
    n_response = len(tokenizer(response + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"])
    return n_prompt + n_response > max_length


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected, meta = [], [], {"prompt_ids": [], "truncated_c": [], "truncated_r": []}
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
            meta["prompt_ids"].append(row["prompt_id"])
            meta["truncated_c"].append(_truncated(tokenizer, prompt, yc, max_length))
            meta["truncated_r"].append(_truncated(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected), meta
    return collate


def pair_logprobs(model, chosen, rejected):
    device = next(model.parameters()).device
    chosen = {k: v.to(device) for k, v in chosen.items()}
    rejected = {k: v.to(device) for k, v in rejected.items()}
    pc = response_sequence_logprobs(model, chosen)[0]
    pr = response_sequence_logprobs(model, rejected)[0]
    with torch.no_grad(), reference_mode(model):
        rc = response_sequence_logprobs(model, chosen)[0]
        rr = response_sequence_logprobs(model, rejected)[0]
    return pc, pr, rc, rr


def summarize_train(rows: list[dict]) -> dict:
    last = rows[-10:]
    return {
        "n_steps": len(rows),
        "loss_first_step": rows[0]["loss"],
        "loss_last10_mean": sum(r["loss"] for r in last) / len(last),
        "acc_last10_mean": sum(r["acc"] for r in last) / len(last),
        "margin_last10_mean": sum(r["margin_mean"] for r in last) / len(last),
        "n_truncated_pairs": sum(r["n_trunc_pairs"] for r in rows),
        "first_batches_prompt_ids": rows[0]["prompt_ids"][:10],
    }


def _save_snapshot(model, optimizer, state, snap_dir):
    tmp = snap_dir.with_name(snap_dir.name + "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    (tmp / "state.json").write_text(json.dumps(state))
    shutil.rmtree(snap_dir, ignore_errors=True)
    tmp.rename(snap_dir)


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None,
                 beta: float | None = None, max_examples: int | None = None, mode: str = "full", resume: bool = False):
    cfg = resolve_models(load_yaml(config_path), mode)
    smoke = mode == "smoke"
    if smoke:
        cfg.update(grad_accum_steps=2, snapshot_every_steps=1)
        max_examples = min(max_examples or 8, 8)
    assert_fixed_files(mode)
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    assert_train_split(opened_path=path)
    beta = float(cfg["beta"] if beta is None else beta)
    seed, accum = int(cfg["seed"]), int(cfg["grad_accum_steps"])

    set_seed(seed)
    rows = read_jsonl(path)
    rows = rows[: int(max_examples)] if max_examples else rows
    tokenizer = load_tokenizer(cfg["base_model"])
    model = disable_dropout(load_policy(cfg, trainable=True, fresh_lora=True))
    lora_dtypes = {p.dtype for n, p in model.named_parameters() if "lora_" in n}
    if lora_dtypes != {torch.float32}:
        raise RuntimeError(f"LoRA parameters must be fp32, got {lora_dtypes}")
    loader = DataLoader(rows, batch_size=int(cfg["batch_size"]), shuffle=True,
                        generator=torch.Generator().manual_seed(seed),
                        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])))
    optimizer = AdamW(trainable_parameters(model), lr=float(cfg["learning_rate"]),
                      weight_decay=float(cfg.get("weight_decay", 0.0)))

    prefix = "_smoke/" if smoke else ""
    output = repo_path(output_path or f"outputs/{prefix}task1_dpo/{run_name}")
    snap_dir = output.with_name(output.name + "_snapshot")
    n_micro = len(loader)
    n_steps = math.ceil(n_micro / accum)
    state = {"step": 0, "micro": 0, "train_seconds": 0.0, "peak_vram_gb": 0.0}
    if resume:
        set_peft_model_state_dict(model, load_file(snap_dir / "adapter_model.safetensors"))
        optimizer.load_state_dict(torch.load(snap_dir / "optimizer.pt"))
        state = json.loads((snap_dir / "state.json").read_text())
    run = start_run(f"results/{prefix}task1_dpo/{run_name}/train", mode=mode, cfg=cfg, seed=seed, resume=resume)
    log_path = run["outdir"] / "train_log.jsonl"
    if resume:
        kept = [r for r in read_jsonl(log_path) if r["step"] <= state["step"]]
        log_path.write_text("".join(json.dumps(r) + "\n" for r in kept))

    params = trainable_parameters(model)
    buffer, t0 = [], time.perf_counter()
    for i, (chosen, rejected, meta) in enumerate(loader):
        if i < state["micro"]:
            continue
        group_size = min(accum, n_micro - (i // accum) * accum)
        pc, pr, rc, rr = pair_logprobs(model, chosen, rejected)
        loss, stats = dpo_loss(pc, pr, rc, rr, beta)
        if i == 0 and abs(loss.item() - math.log(2)) > 1e-4:
            print(f"BREACH: first micro-batch loss {loss.item():.6f} != log 2; the reference is not the initial policy")
            sys.exit(1)
        (loss / group_size).backward()
        buffer.append({"loss": loss.item(), **{k: float(v) for k, v in stats.items()},
                       "prompt_ids": meta["prompt_ids"],
                       "n_trunc": sum(a or b for a, b in zip(meta["truncated_c"], meta["truncated_r"]))})
        if (i + 1) % accum and i + 1 != n_micro:
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        mean = lambda k: sum(b[k] for b in buffer) / len(buffer)
        state["step"] += 1
        state["micro"] = i + 1
        state["train_seconds"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            state["peak_vram_gb"] = max(state["peak_vram_gb"], torch.cuda.max_memory_allocated() / 2**30)
        append_jsonl(log_path, {
            "step": state["step"], "loss": mean("loss"), "margin_mean": mean("margin_mean"),
            "acc": mean("preference_accuracy"), "chosen_reward": mean("chosen_reward"),
            "rejected_reward": mean("rejected_reward"), "grad_norm": float(grad_norm),
            "lr": optimizer.param_groups[0]["lr"], "n_trunc_pairs": sum(b["n_trunc"] for b in buffer),
            "prompt_ids": [p for b in buffer for p in b["prompt_ids"]],
        })
        print(f"step {state['step']}/{n_steps} loss={mean('loss'):.4f} acc={mean('preference_accuracy'):.3f} "
              f"grad_norm={float(grad_norm):.3f}", flush=True)
        buffer = []
        if state["step"] % int(cfg["snapshot_every_steps"]) == 0 or state["step"] == n_steps:
            _save_snapshot(model, optimizer, state, snap_dir)

    model.save_pretrained(output)
    log_rows = read_jsonl(log_path)
    summary = summarize_train(log_rows)
    extra = {
        "run_name": run_name, "dataset": str(path), "beta": beta, "n_pairs": len(rows),
        "budget": f"{len(rows)} pairs, {n_steps} steps", "adapter": str(output.relative_to(repo_path("."))),
        "adapter_sha256": sha256_file(output / "adapter_model.safetensors"),
        "train_seconds": round(state["train_seconds"], 1), "peak_vram_gb_all_segments": round(state["peak_vram_gb"], 3),
    }
    finalize(run, log_path, summary, summarize_train, extra=extra)
    reconcile(run["outdir"], {"train_log.jsonl", "summary.json"})
    return {"adapter": output, "summary": summary, **extra}


def main():
    ap = argparse.ArgumentParser()
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out = run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples,
                       args.mode, args.resume)
    print(f"saved {out['adapter']} sha256={out['adapter_sha256']} budget: {out['budget']}")


if __name__ == "__main__":
    main()
