# Dockerfile — LegalReviewEnv OpenEnv Submission
# Hugging Face Spaces requires port 7860
# FastAPI handles POST /reset, POST /step, GET /state, POST /validate

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY env/        ./env/
COPY tasks/      ./tasks/
COPY grader/     ./grader/
COPY server.py      .
COPY inference.py   .
COPY app.py         .
COPY openenv.yaml   .

ENV API_BASE_URL="http://localhost:8000/v1"
ENV MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

RUN python -c "from env.legal_review_env import LegalReviewEnv; from grader import grade_episode; print('Build OK')"

EXPOSE 7860

CMD ["python", "server.py"]
