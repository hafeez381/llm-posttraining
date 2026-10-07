from __future__ import annotations

import argparse

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.eval_protocol import evaluate_policy, summarize
from common.guards import assert_fixed_files, sha256_file
from common.models import load_policy, load_reward_model, load_tokenizer
from common.results import finalize, reconcile, start_run, write_rows
from common.runmode import adapter_kwargs, add_mode_arg, resolve_models
from task1_dpo.evaluate import logprob_temperature


def evaluate(config_path: str, adapter: str, name: str, task: str = "task2_ppo", mode: str = "full") -> dict:
    cfg = resolve_models(load_yaml(config_path), mode)
    smoke = mode == "smoke"
    assert_fixed_files(mode)
    temperature = logprob_temperature(cfg, mode)
    prefix = "_smoke/" if smoke else ""
    run = start_run(f"results/{prefix}{task}/{name}/eval", mode=mode, cfg=cfg, seed=int(cfg["seed"]))

    tok = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, **adapter_kwargs(cfg, adapter))
    rm, rm_tok = load_reward_model(cfg)
    prompts = [{"prompt_id": r["prompt_id"], "messages": prompt_messages(r)}
               for r in read_jsonl(cfg["paths"]["rl_prompt_eval"])[: 4 if smoke else None]]
    rows = evaluate_policy(policy, tok, rm, rm_tok, prompts, max_prompt_length=int(cfg["max_prompt_length"]),
                           max_new_tokens=32 if smoke else int(cfg["eval_max_response_length"]),
                           batch_size=2 if smoke else int(cfg["eval_batch_size"]),
                           reward_max_length=int(cfg["reward_max_length"]), logprob_temperature=temperature,
                           generation=cfg["generation"], seed=int(cfg["seed"]))
    rows_path = run["outdir"] / "generations.jsonl"
    write_rows(rows_path, rows)
    summary = summarize(rows)
    adapter_file = repo_path(adapter) / "adapter_model.safetensors"
    extra = {"name": name, "adapter": adapter, "logprob_temperature": temperature,
             "adapter_sha256": sha256_file(adapter_file) if adapter_file.exists() else None}
    finalize(run, rows_path, summary, summarize, extra=extra)
    reconcile(run["outdir"], {"generations.jsonl", "summary.json"})
    print(f"{name}: rm={summary['rm_score']['mean']:.3f}±{summary['rm_score']['boot_se']:.3f} "
          f"kl_seq={summary['kl_seq_mean']:.3f}±{summary['kl_seq']['boot_se']:.3f} "
          f"len={summary['length']['mean']:.1f}±{summary['length']['boot_se']:.1f} "
          f"ent={summary['entropy_exact_tokmean']:.3f} eos={summary['eos_rate']:.3f} "
          f"trunc={summary['truncation_rate']:.3f}")
    return {"summary": summary, **extra}


def main():
    ap = argparse.ArgumentParser()
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    evaluate(args.config, args.adapter, args.name, mode=args.mode)


if __name__ == "__main__":
    main()
