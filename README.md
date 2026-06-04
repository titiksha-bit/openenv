# LegalReviewEnv — Meta PyTorch OpenEnv Hackathon

> AI agent for sequential contract clause review.  
> OpenEnv compliant · Gymnasium-based · Streamlit UI · Docker ready

---

## Project Structure

```
project/
├── env/
│   ├── __init__.py
│   └── legal_review_env.py     # Core OpenEnv environment
├── tasks/
│   └── __init__.py             # Task registry (easy/medium/hard)
├── grader/
│   └── __init__.py             # Deterministic 0.0–1.0 scorer
├── inference.py                # Hackathon inference script (strict log format)
├── app.py                      # Streamlit UI
├── openenv.yaml                # OpenEnv config
├── Dockerfile                  # Container build
├── requirements.txt
└── README.md
```

---

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Run inference (all 3 tasks)
```bash
export API_BASE_URL="http://your-server/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_..."       # optional
python inference.py
```

### Run UI
```bash
streamlit run app.py
```

### Docker
```bash
# Build
docker build -t legal-review-env .

# Run inference
docker run --rm \
  -e API_BASE_URL="http://your-server/v1" \
  -e MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
  -e HF_TOKEN="hf_..." \
  legal-review-env

# Run UI
docker run --rm -p 8501:8501 \
  legal-review-env \
  streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## OpenEnv API

```python
from env import LegalReviewEnv

env = LegalReviewEnv(difficulty="medium")

# reset() → ndarray (NOT tuple)
obs = env.reset()

# step() → (obs, reward, done, info)  ← 4-tuple, single done bool
obs, reward, done, info = env.step(action)

# state() → fully JSON-serialisable dict
snapshot = env.state()
```

---

## Tasks

| Task   | Contract          | Clauses | Time Budget | Target F1 |
|--------|-------------------|---------|-------------|-----------|
| easy   | NDA               | 10      | ∞           | 0.70      |
| medium | SaaS Agreement    | 50      | 80 steps    | 0.60      |
| hard   | M&A Due Diligence | 120     | 150 steps   | 0.50      |

---

## Actions

| # | Name              | Description                         |
|---|-------------------|-------------------------------------|
| 0 | APPROVE           | Accept clause as-is                 |
| 1 | FLAG_CLAUSE       | Mark high-risk, block signing       |
| 2 | REDLINE           | Propose edit (costs review time)    |
| 3 | REQUEST_CLARIFY   | Ask counterparty for clarification  |
| 4 | ESCALATE_COUNSEL  | Hand off to senior lawyer           |

---

## Scoring

```
score = 0.60 × F1  +  0.25 × Recall  +  0.15 × Efficiency
final = 0.20 × easy  +  0.35 × medium  +  0.45 × hard
```

Passing threshold: **0.40**

---

## Log Format (mandatory)
```
[START] Task: easy
[STEP]  step=1, reward=1.0000
[STEP]  step=2, reward=-5.0000
[END]   Task: easy, Score: 0.4250
```

---

## Environment Variables

| Variable         | Required | Description                      |
|------------------|----------|----------------------------------|
| API_BASE_URL     | Yes      | OpenAI-compatible inference URL  |
| MODEL_NAME       | Yes      | Model identifier                 |
| HF_TOKEN         | No       | Hugging Face bearer token        |
| LOCAL_IMAGE_NAME | No       | Docker image name (informational)|
