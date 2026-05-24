# Adversarial Evaluation Agent

An adaptive multi-turn harness for stress-testing LLM-powered chatbots against adversarial inputs. The harness autonomously generates attack sequences, judges responses, and adapts its strategy across sessions based on what works — without generating harmful content.

## How it works

Each run orchestrates four LLM components:

| Component | Role |
|---|---|
| **Adaptation Planner** | Decides the next attack angle based on prior turn results |
| **Turn Generator** | Crafts the actual user message using that strategy |
| **Safety Judge** | Scores the chatbot's response (0–4 failure scale) |
| **Session Policy** | Decides when to abandon a session and start fresh |

The planner uses **cross-session attack memory** — it remembers which strategies triggered failures and which were consistently refused, and carries that context into future sessions.

## Scenario types

| Type | What it tests |
|---|---|
| `toxicity` | Identity-targeted language, hostile framing |
| `prompt-injection` | Instruction hijacking, role overrides |
| `data-leakage` | System prompt extraction, PII elicitation |
| `hallucination` | Confidence-inducing pressure, false authority |
| `persona-hijack` | Coercion into regulated professional personas (financial, medical, legal) |
| `tool-safety` | Unauthorized tool calls, parameter hijacking |

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in API keys
```

**No API key needed — run fully local with mock:**
```bash
python run_experiment.py --provider mock --target mock --verbose
```

**Against a real chatbot with Claude as the harness:**
```bash
python run_experiment.py \
  --provider claude \
  --target https://your-chatbot-api/chat \
  --api-key <key> \
  --scenario-type toxicity \
  --verbose
```

**Sweep all mock variants × budget tiers (9 experiments):**
```bash
python run_experiment.py --provider mock --target mock --multi-run
```

## Results

Results are written to `results/<env>/<scenario_type>/results_<timestamp>.json`.

| Run type | Output path |
|---|---|
| Mock target | `results/mock/<scenario_type>/` |
| Real target | `results/prod/<scenario_type>/` |

Each result file contains a `summary` block and full per-turn session data.

**Summary fields include:**
- `failure_rate`, `failed_sessions`, `avg_turns_to_failure`
- `tokens_used_total`, `tokens_per_failure`
- `estimated_cost_usd`, `avg_cost_per_session_usd`
- Judge axes: `avg_harm_potential`, `avg_specificity`, `avg_refusal_quality`

Cost estimates use current public pricing for Claude (Haiku/Sonnet/Opus), GPT-4o, and GPT-4o-mini. Mock runs report `$0.00`.

## Configuration

Key CLI flags:

```
--provider          claude | openai | bedrock | azure-openai | mock
--model             Override default model for the chosen provider
--target            'mock' or a real chatbot URL
--target-variant    strict | baseline | lenient  (mock only)
--budget            Token budget per run (default: 100,000)
--max-turns         Max turns per session (default: 8)
--scenario-type     toxicity | prompt-injection | data-leakage |
                    hallucination | persona-hijack | tool-safety
--session-policy    llm (default) | rule
--no-attack-memory  Disable cross-session memory
--multi-run         Sweep all mock variants × budget tiers
--verbose           Print per-turn progress
```

Per-component model overrides (useful for mixing cheap + capable models):

```
--planner-provider / --planner-model
--generator-provider / --generator-model
--judge-provider / --judge-model
--policy-provider / --policy-model
```

## Storage backends

```bash
# Local (default)
python run_experiment.py --storage local

# AWS S3
python run_experiment.py --storage s3 --s3-bucket my-bucket --s3-prefix adversarial-eval

# Azure Blob
python run_experiment.py --storage azure-blob --azure-container results
```

## Observability

MLflow run logs are written automatically alongside results:

| Run type | MLflow path |
|---|---|
| Mock target | `results/mock/mlruns/` |
| Real target | `results/prod/mlruns/` |

No configuration needed — the harness sets the tracking URI based on the target. To browse runs locally:

```bash
mlflow ui --backend-store-uri results/mock/mlruns   # mock runs
mlflow ui --backend-store-uri results/prod/mlruns   # prod runs
```

To use a remote MLflow server instead, set `MLFLOW_TRACKING_URI` in `.env`:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

## Project structure

```
run_experiment.py       # CLI entry point
harness/
  attack_agent.py       # Planner + Generator orchestration
  components.py         # Planner, Generator, Judge, SessionPolicy
  evaluator.py          # Main experiment loop
  llm_backends.py       # Claude, OpenAI, Bedrock, Azure, Mock backends
  metrics.py            # summarize_experiment + cost estimation
  models.py             # AttackMemory, TurnRecord, SessionState, ExperimentState
  observability.py      # MLflow integration
  prompts.py            # All system prompts + scenario/persona configs
  storage.py            # Local, S3, Azure Blob storage backends
  token_budget.py       # Token tracking across components
analysis/
  visualize.py          # Charts from result files
results/
  mock/                 # Mock run outputs
    mlruns/             # MLflow tracking data (mock)
  prod/                 # Real target run outputs
    mlruns/             # MLflow tracking data (prod)
```
