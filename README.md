# LLM Eval Demo

Small script that asks a question using either the OpenAI or Anthropic API, then runs a second evaluation step to score `correctness`, `hallucination_risk`, and `confidence`.

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` with one or both keys:
```env
OPENAI_API_KEY="..."
ANTHROPIC_API_KEY="..."
```

## Run

```bash
python3 main.py --provider anthropic --model claude-3-5-sonnet-latest -q "What is the capital of France?"
```

Each run appends a JSONL line to `logs.jsonl` containing the question, answer, and evaluation.
