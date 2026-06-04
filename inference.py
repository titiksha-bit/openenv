"""
inference.py
============
Meta PyTorch OpenEnv Hackathon — LegalReviewEnv inference script.

Runs all three tasks (easy / medium / hard) with a rule-based agent
(falls back from LLM on any API error).

MANDATORY log format:
  [START] Task: <task_name>
  [STEP]  step=<n>, reward=<r>
  [END]   Task: <task_name>, Score: <score>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from openai import OpenAI

from env.legal_review_env import LegalReviewEnv, Action, N_ACTIONS, ACTION_LABELS
from grader import grade_episode, grade_all_tasks
from tasks import make_env


# ── Environment variables (NEVER hardcoded) ───────────────────────────────────

API_BASE_URL     = os.getenv("API_BASE_URL",  "http://localhost:8000/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",    "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "EMPTY")


# ── LLM agent ─────────────────────────────────────────────────────────────────

_SYS = (
    "You are a senior contract lawyer. Given clause info, pick ONE action:\n"
    "0=APPROVE 1=FLAG_CLAUSE 2=REDLINE 3=REQUEST_CLARIFY 4=ESCALATE_COUNSEL\n"
    "Reply with ONLY a single digit 0-4."
)

def _llm_action(ctx: dict) -> int:
    prompt = (
        f"Contract : {ctx.get('contract_type','')}\n"
        f"Type     : {ctx.get('clause_type','')}\n"
        f"Jur      : {ctx.get('jurisdiction','')}\n"
        f"Nested   : {ctx.get('has_nested_ref',False)}\n"
        f"Redlines : {ctx.get('prior_redlines',0)}\n"
        f"TimeLeft : {ctx.get('time_remaining',999)}\n"
        "Action (0-4):"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"system","content":_SYS},{"role":"user","content":prompt}],
            max_tokens=3, temperature=0.0,
        )
        a = int(resp.choices[0].message.content.strip()[0])
        return a if 0 <= a < N_ACTIONS else _rule_action(ctx)
    except Exception:
        return _rule_action(ctx)


# ── Rule-based agent ──────────────────────────────────────────────────────────

_HIGH = {"indemnity","limitation_of_liability","termination","ip_ownership"}

def _rule_action(ctx: dict) -> int:
    if ctx.get("time_remaining", 999) < 10:
        return int(Action.ESCALATE_COUNSEL)
    if ctx.get("clause_type","") in _HIGH:
        return int(Action.FLAG_CLAUSE)
    if ctx.get("has_nested_ref", False):
        return int(Action.REQUEST_CLARIFY)
    if ctx.get("clause_type","") in {"payment_terms","data_protection"} and ctx.get("prior_redlines",0) > 1:
        return int(Action.REDLINE)
    return int(Action.APPROVE)


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(task_name: str, use_llm: bool = True) -> dict:
    env  = make_env(task_name)
    obs  = env.reset()

    print(f"[START] Task: {task_name}")

    step_num = 0
    done     = False
    while not done:
        s   = env.state()
        ctx = s.get("current_clause") or {}
        ctx["contract_type"]  = s.get("contract_type","")
        ctx["time_remaining"] = s.get("time_remaining",999)

        action           = _llm_action(ctx) if use_llm else _rule_action(ctx)
        obs, reward, done, info = env.step(action)
        step_num        += 1

        print(f"[STEP]  step={step_num}, reward={reward:.4f}")

    score = grade_episode(env)
    print(f"[END]   Task: {task_name}, Score: {score:.4f}")

    return {"score": score, "metrics": env.episode_metrics()}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    use_llm = True
    results: dict[str, dict] = {}
    for task in ("easy", "medium", "hard"):
        results[task] = run_task(task, use_llm=use_llm)

    agg = grade_all_tasks(results)
    with open("results.json", "w") as f:
        json.dump({
            "per_task":    agg["per_task"],
            "final_score": agg["final_score"],
            "passed":      agg["passed"],
            "details":     {t: r["metrics"] for t, r in results.items()},
        }, f, indent=2)


if __name__ == "__main__":
    main()
