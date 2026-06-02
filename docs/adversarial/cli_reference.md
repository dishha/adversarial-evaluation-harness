# Adversarial Response Engine — CLI Reference

Run `are --help` for the canonical flag list. This document groups and explains all flags.

## Quickstart

```bash
# No API key needed
are --dry-run --verbose

# Multi-run: all variants × all budget tiers
are --provider mock --target mock --multi-run

# Claude attacker, mock target
are --provider claude --target mock --budget 100000

# Real chatbot endpoint
are --target https://your-chatbot-api/chat --api-key <key>
```

---

## Core

| Flag | Default | Description |
|---|---|---|
| `--config` | — | Path to a YAML config file; CLI flags override it |
| `--dry-run` | off | Shorthand for `--provider mock --target mock` |
| `--verbose` | off | Print turn-by-turn progress |
| `--realtime` | off | Rich live display as experiment runs |
| `--chat` | off | Watch attacker vs. bot stream live (use with `--target simulate`) |

## LLM Provider (Attacker)

| Flag | Default | Description |
|---|---|---|
| `--provider` | `claude` | `claude`, `openai`, `mock`, `bedrock`, `azure-openai` |
| `--model` | provider default | Override attacker model name |

## Target (Chatbot Under Test)

| Flag | Default | Description |
|---|---|---|
| `--target` | `mock` | `mock`, `simulate`, or a real chatbot URL |
| `--target-variant` | `baseline` | Mock strictness: `strict`, `baseline`, `lenient` |
| `--simulate-model` | cheapest for provider | Model for `--target simulate` |
| `--api-key` | env `TARGET_CHATBOT_API_KEY` | API key for real target |

## Experiment Control

| Flag | Default | Description |
|---|---|---|
| `--budget` | `100000` | Token budget per experiment |
| `--max-turns` | `8` | Max turns per session |
| `--failure-threshold` | `3` | Score ≥ this = failure (0–4 scale) |
| `--reserve-tokens` | `1500` | Min tokens needed to attempt another turn |
| `--multi-run` | off | Run all variants × all budget tiers |
| `--session-policy` | `llm` | `llm` (adaptive) or `rule` (deterministic) |
| `--no-attack-memory` | off | Disable cross-session attack memory |

## Scenario

| Flag | Default | Description |
|---|---|---|
| `--scenario` | built-in text | Evaluation scenario description |
| `--scenario-type` | `toxicity` | See types below |
| `--personas` | — | For `persona-hijack`: `all`, `financial`, `medical`, `legal`, or comma-separated custom strings |

**Scenario types:** `toxicity`, `prompt-injection`, `data-leakage`, `hallucination`, `persona-hijack`, `tool-call`, `document-exfiltration`, `system-prompt-override`, `malicious-resource-fetching`, `data-pii-leak`, `unsupervised-contracts`

## Per-Component Model Overrides

Each attacker component can use a different provider and/or model. Unset components inherit `--provider` / `--model`.

| Flags | Component |
|---|---|
| `--planner-provider` / `--planner-model` | Adaptation planner |
| `--generator-provider` / `--generator-model` | Turn generator |
| `--judge-provider` / `--judge-model` | Safety judge |
| `--policy-provider` / `--policy-model` | Session policy controller |

Example — use a stronger model only for judging:
```bash
are --provider mock --target mock --judge-provider claude --judge-model claude-sonnet-4-6
```

## Output & Storage

| Flag | Default | Description |
|---|---|---|
| `--output-dir` | `results` | Directory for result files |
| `--output` | timestamped | Override output filename |
| `--storage` | `local` | `local`, `s3`, `azure-blob` |
| `--s3-bucket` | — | S3 bucket name (required for `--storage s3`) |
| `--s3-prefix` | `adversarial-eval` | S3 key prefix |
| `--s3-region` | `us-east-1` | AWS region |
| `--azure-container` | — | Azure Blob container (required for `--storage azure-blob`) |
| `--azure-prefix` | `adversarial-eval` | Azure Blob path prefix |

## Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Required when `--provider claude` |
| `OPENAI_API_KEY` | Required when `--provider openai` |
| `LLM_PROVIDER` | Default provider (overridden by `--provider`) |
| `LLM_MODEL` | Default model (overridden by `--model`) |
| `TARGET_CHATBOT_URL` | Default target URL (overridden by `--target`) |
| `TARGET_CHATBOT_API_KEY` | API key for real target (overridden by `--api-key`) |
| `EVAL_BUDGET` | Default token budget (overridden by `--budget`) |
| `EVAL_SCENARIO_TYPE` | Default scenario type (overridden by `--scenario-type`) |

Can also be set via a `.env` file in the project root.
