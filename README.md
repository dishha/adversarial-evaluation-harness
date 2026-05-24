# Adversarial Evaluation Agent

An adaptive multi-turn harness for stress-testing LLM-powered chatbots against adversarial inputs. The harness autonomously generates attack sequences, judges responses, and adapts its strategy across sessions based on what works — without generating harmful content.

## How it works

Each run orchestrates four LLM components in a closed loop:

```
generate → call chatbot → judge → adapt → continue/reset → repeat
```

| Component | Role |
|---|---|
| **Adaptation Planner** | Diagnoses model posture, selects attack angle + sub-tactic, plans multi-turn ladders |
| **Turn Generator** | Crafts realistic user messages using social engineering triggers and probe architectures |
| **Safety Judge** | Scores responses on failure scale, harm potential, specificity, and refusal quality |
| **Session Policy** | Decides when to abandon a "burned" session and start fresh |

The planner uses **cross-session attack memory** — it remembers which strategies triggered failures and which were consistently refused, and carries that context across sessions.

### Attack system

The planner operates in three phases each turn:

1. **Situational reading** — diagnose the model's current posture (`locked`, `engaged_but_guarded`, `compliant`, `suspicious`, `reset`) before choosing a move
2. **Angle + sub-tactic selection** — 8 primary attack angles, each with 4 named sub-tactics (e.g. `authority_injection / regulatory_framing`, `role_entrapment / debug_mode_claim`)
3. **Ladder planning** — explicit turn-to-turn dependency chaining so each turn sets up the next

The generator builds messages along four dimensions: **register** (formal/casual/technical/emotional), **directness** (explicit/nested/steganographic), **probe architecture** (direct/lead-with-benign/sandwich/multi-step), and **social engineering trigger** (urgency/authority/sympathy/curiosity/reciprocity/scarcity).

### Judge output

Every turn produces a structured judge record with:
- `reasoning` — chain-of-thought before scoring
- `failure_score` (0–4), `harm_potential`, `specificity`, `refusal_quality`
- `near_miss` — true if the guardrail almost failed
- `confidence` — judge's confidence in the verdict (0–1)
- Scenario-specific fields (e.g. `injection_vector`, `override_technique`, `regulatory_framework`, `enforceability_risk`)

## Scenario types

| Type | What it tests |
|---|---|
| `toxicity` | Identity-targeted language, hostile framing, dehumanizing abstractions |
| `prompt-injection` | Instruction hijacking, role overrides, indirect payload delivery |
| `data-leakage` | System prompt extraction, credential fishing, retrieval leakage |
| `hallucination` | Confidence-inducing pressure, false authority, fabricated citations |
| `persona-hijack` | Coercion into regulated professional personas (financial, medical, legal) |
| `tool-call` | Unauthorized tool invocations, parameter hijacking, schema leakage |
| `document-exfiltration` | Bulk document dumps, knowledge-base content leakage, retrieval manipulation |
| `system-prompt-override` | Replacing operating instructions, debug-mode claims, persona substitution |
| `malicious-resource-fetching` | SSRF payloads, attacker-controlled URL injection, callback exfiltration |
| `data-pii-leak` | Eliciting names, emails, SSNs, DOBs, financial/medical records |
| `unsupervised-contracts` | Getting the model to commit to agreements or obligations without human oversight |

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
  --scenario-type data-pii-leak \
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

Each result file contains a `summary` block and full per-turn session traces.

**Summary fields include:**
- `failure_rate`, `failed_sessions`, `avg_turns_to_failure`
- `tokens_used_total`, `tokens_per_failure`
- `estimated_cost_usd`, `avg_cost_per_session_usd`
- Judge axes: `avg_harm_potential`, `avg_specificity`, `avg_refusal_quality`

Cost estimates use current public pricing for Claude, GPT-4o, and GPT-4o-mini. Mock runs report `$0.00`.

## Configuration

Key CLI flags:

```
--provider          claude | openai | bedrock | azure-openai | mock
--model             Override default model for the chosen provider
--target            'mock' or a real chatbot URL
--target-variant    strict | baseline | lenient  (mock only)
--budget            Token budget per run (default: 100,000)
--max-turns         Max turns per session (default: 8)
--scenario-type     toxicity | prompt-injection | data-leakage | hallucination |
                    persona-hijack | tool-call | document-exfiltration |
                    system-prompt-override | malicious-resource-fetching |
                    data-pii-leak | unsupervised-contracts
--session-policy    llm (default) | rule
--no-attack-memory  Disable cross-session memory
--multi-run         Sweep all mock variants × budget tiers
--verbose           Print per-turn progress
```

Per-component model overrides (useful for mixing cheap and capable models):

```
--planner-provider / --planner-model
--generator-provider / --generator-model
--judge-provider / --judge-model
--policy-provider / --policy-model
```

Persona pool controls (for `persona-hijack`):

```
--personas all                  # full built-in pool
--personas financial            # financial domain subset
--personas medical              # medical domain subset
--personas legal                # legal domain subset
--personas "custom A,custom B"  # custom strings
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

To use a remote MLflow server, set `MLFLOW_TRACKING_URI` in `.env`:

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
