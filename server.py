"""
server.py — OpenEnv REST API Server
=====================================
The hackathon evaluator sends HTTP POST requests to validate the environment.
This FastAPI server exposes the required endpoints:

  POST /reset          → returns initial observation
  POST /step           → returns (obs, reward, done, info)
  GET  /state          → returns full serialisable state
  GET  /health         → health check
  POST /validate       → runs full validation and returns results
"""

from __future__ import annotations

import random
import sys
import types

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Mock heavy deps if not installed ─────────────────────────────────────────
def _mock_if_missing():
    if "gymnasium" not in sys.modules:
        gym_mod = types.ModuleType("gymnasium")
        sp_mod  = types.ModuleType("gymnasium.spaces")
        class _D:
            def __init__(self, n): self.n = n
            def contains(self, x): return 0 <= int(x) < self.n
            def sample(self): return random.randint(0, self.n - 1)
        class _B:
            def __init__(self, lo, hi, shape, dtype): self.shape = shape; self.dtype = dtype
        class _E:
            metadata = {}; action_space = None; observation_space = None
            def reset(self, **kw): pass
            def step(self, a): pass
            def close(self): pass
        sp_mod.Discrete = _D; sp_mod.Box = _B
        gym_mod.Env = _E; gym_mod.spaces = sp_mod
        sys.modules["gymnasium"] = gym_mod
        sys.modules["gymnasium.spaces"] = sp_mod

    if "faker" not in sys.modules:
        fk_mod = types.ModuleType("faker")
        class _F:
            @staticmethod
            def seed(s): random.seed(s)
            def bs(self): return random.choice([
                "synergize scalable markets", "leverage agile frameworks",
                "matrix B2B deliverables", "orchestrate cross-platform content",
            ])
        fk_mod.Faker = _F
        sys.modules["faker"] = fk_mod

_mock_if_missing()

from env.legal_review_env import LegalReviewEnv, DIFFICULTY_PRESETS
from grader import grade_episode

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LegalReviewEnv — OpenEnv API",
    description="OpenEnv-compliant REST API for the Legal Contract Review environment.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory environment store (one per difficulty) ──────────────────────────
_envs: dict[str, LegalReviewEnv] = {}


def _get_or_create(task: str) -> LegalReviewEnv:
    if task not in _envs:
        assert task in DIFFICULTY_PRESETS, f"Unknown task: {task}"
        _envs[task] = LegalReviewEnv(difficulty=task)
    return _envs[task]


# ── Request / Response models ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "easy"          # "easy" | "medium" | "hard"
    seed: int | None = None


class StepRequest(BaseModel):
    task:   str = "easy"
    action: int = 0             # 0-4


class ValidateRequest(BaseModel):
    task: str = "easy"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — evaluator pings this first."""
    return {"status": "ok", "env": "LegalReviewEnv", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest):
    """
    Reset the environment and return the initial observation.
    Required by OpenEnv evaluator (POST /reset).
    """
    try:
        env = _get_or_create(req.task)
        obs = env.reset(seed=req.seed)
        return {
            "observation": obs.tolist(),
            "task":        req.task,
            "n_clauses":   DIFFICULTY_PRESETS[req.task]["n_clauses"],
            "status":      "reset_ok",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    """
    Take one step in the environment.
    Returns (observation, reward, done, info).
    """
    env = _get_or_create(req.task)
    if env._done:
        raise HTTPException(status_code=400, detail="Episode done. Call /reset first.")
    try:
        obs, reward, done, info = env.step(req.action)
        return {
            "observation": obs.tolist(),
            "reward":      reward,
            "done":        done,
            "info":        info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state(task: str = "easy"):
    """Return full serialisable environment state."""
    env = _get_or_create(task)
    return env.state()


@app.post("/validate")
def validate(req: ValidateRequest):
    """
    Run a full episode with a rule-based agent and return metrics.
    Used by the OpenEnv evaluator for automated checks.
    """
    task = req.task
    env  = LegalReviewEnv(difficulty=task)
    env.reset()

    _HIGH = {"indemnity", "limitation_of_liability", "termination", "ip_ownership"}

    done = False
    while not done:
        s   = env.state()
        ctx = s.get("current_clause") or {}
        ctx["time_remaining"] = s.get("time_remaining", 999)

        # Rule-based agent
        if ctx.get("time_remaining", 999) < 10:
            action = 4  # ESCALATE
        elif ctx.get("clause_type", "") in _HIGH:
            action = 1  # FLAG
        elif ctx.get("has_nested_ref", False):
            action = 3  # CLARIFY
        else:
            action = 0  # APPROVE

        _, _, done, _ = env.step(action)

    score   = grade_episode(env)
    metrics = env.episode_metrics()

    return {
        "task":    task,
        "score":   score,
        "metrics": metrics,
        "status":  "validation_ok",
    }


@app.get("/")
def root():
    return {
        "name":        "LegalReviewEnv",
        "description": "OpenEnv-compliant legal contract review environment",
        "version":     "1.0.0",
        "endpoints": {
            "POST /reset":    "Reset environment, get initial observation",
            "POST /step":     "Take action, get (obs, reward, done, info)",
            "GET  /state":    "Get full environment state",
            "POST /validate": "Run full episode, get score",
            "GET  /health":   "Health check",
        },
        "tasks": ["easy", "medium", "hard"],
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
