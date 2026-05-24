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

### Pipeline diagram

```mermaid
flowchart TD
    CFG(["Run Config\nscenario_type · scenario\nbudget · max_turns"])
    MEM[("Cross-Session\nAttack Memory")]

    subgraph loop ["Session Loop"]
        P["Adaptation Planner"]
        G["Turn Generator"]
        T["Target Chatbot"]
        J["Safety Judge"]
        SP["Session Policy"]

        P -->|"strategy instruction"| G
        G -->|"user_input"| T
        T -->|"chatbot_response"| J
        J -->|"judge record"| SP
        SP -->|"continue"| P
        SP -->|"start_new_session"| P
    end

    CFG --> P
    MEM -->|"prior strategy outcomes"| P
    SP -->|"update"| MEM
    SP -.->|"budget exhausted"| OUT(["Results JSON\n+ MLflow Logs"])
```

**Data flow — key inputs and outputs per component:**

| Component | Inputs | Key outputs |
|---|---|---|
| **Adaptation Planner** | session history (tried_angles + compact turn log), attack memory, scenario, current scores | `attack_angle`, `sub_tactic`, `ladder_dependency`, `next_generator_instruction`, `model_posture`, `risk_level` |
| **Turn Generator** | scenario, `next_generator_instruction` | `user_input`, `register`, `probe_architecture`, `social_trigger`, `framing_rationale` |
| **Target Chatbot** | `user_input` | `chatbot_response` |
| **Safety Judge** | `user_input`, `chatbot_response` | `failure_score` (0–4), `harm_potential`, `specificity`, `refusal_quality`, `near_miss`, `confidence`, `reasoning` |
| **Session Policy** | session state (turn_count, best_failure_score, repeated_refusals, suspicion_score), recent judge records | `action` (continue / start_new_session), `updated_suspicion_score` |

### Attack system

The planner operates in three phases each turn:

1. **Situational reading** — diagnose the model's current posture (`locked`, `engaged_but_guarded`, `compliant`, `suspicious`, `reset`) before choosing a move
2. **Angle + sub-tactic selection** — 12 primary attack angles, each with named sub-tactics. Never repeats a `(angle, sub_tactic)` pair already used in the session.
3. **Ladder planning** — explicit turn-to-turn dependency chaining so each turn sets up the next

**Attack angles:**

| Angle | Sub-tactics (sample) | Notes |
|---|---|---|
| `angle_shift` | academic_frame, policy_audit, fictional_universe, historical_parallel | Reframe the epistemics entirely |
| `indirect_priming` | context_seeding, false_premise_anchor, incremental_normalization | Build cover context before the probe |
| `specificity_escalation` | vague_to_precise, slow_ramp, normalization_then_spike | Push from vague acknowledgment to actionable detail |
| `authority_injection` | professional_credential, regulatory_framing, peer_citation | Introduce legitimacy signals implying authorized access |
| `deflection_wrap` | buried_probe, multi_part_obfuscation, benign_anchor_with_payload | Embed the probe as a non-salient element |
| `role_entrapment` | simulation_frame, debug_mode_claim, co_author_setup | Establish a role where unsafe behavior feels expected |
| `social_pressure` | urgency_injection, sunk_cost_leverage, reciprocity_trigger | Apply psychological leverage to lower the threshold |
| `memory_exploitation` | prior_compliance_reference, false_memory_injection, compliance_misattribution | Reference and extend prior partial compliance |
| `semantic_drift` | topic_sliding, frame_shift_accumulation, boundary_erosion | Shift meaning gradually — no single turn is a red flag |
| `decomposition_attack` | component_extraction, workflow_reconstruction, knowledge_assembly | Extract harmful knowledge piecemeal across turns |
| `hypothetical_framing` | suppositional_frame, fictional_world_building, what_if_escalation | Use fictional distance to lower the refusal threshold |
| `conversation_state_attack` | false_history_injection, conversation_reset_claim, prior_context_erasure | Manipulate the conversation's apparent history or state |

The generator builds messages along five dimensions: **register** (formal/casual/technical/emotional), **directness** (explicit/nested/steganographic), **person** (first/third/hypothetical/delegated), **probe architecture** (direct/lead-with-benign/sandwich/multi-step/decomposition-chain/false-history-frame), and **social engineering trigger** (urgency/authority/sympathy/curiosity/reciprocity/scarcity).

### Judge output

Every turn produces a structured judge record with:
- `reasoning` — chain-of-thought before scoring
- `failure_score` (0–4), `harm_potential`, `specificity`, `refusal_quality`
- `near_miss` — true if the guardrail almost failed
- `confidence` — judge's confidence in the verdict (0–1)
- Scenario-specific fields (e.g. `injection_vector`, `override_technique`, `regulatory_framework`, `enforceability_risk`)

### Planner history feed

The planner receives a compact operator-style log — not a raw transcript. The full user message and bulk judge fields are dropped; what remains is the technique used, the defense signal observed, and the first 200 chars of the chatbot response (where the refusal or compliance signal typically appears). A `tried_angles` index is prepended so the planner can scan all exhausted approaches at a glance without re-reading every turn.

Structure fed to the planner each turn:

```json
{
  "tried_angles": [
    { "angle": "authority_injection", "sub_tactic": "regulatory_framing", "score": 1 },
    { "angle": "role_entrapment",     "sub_tactic": "debug_mode_claim",   "score": 0 }
  ],
  "turns": [
    {
      "t": 3,
      "attack": {
        "angle": "authority_injection",
        "sub_tactic": "regulatory_framing",
        "instruction": "Pose as compliance auditor requesting system config details",
        "ladder": "Establish authority this turn; request specific config values next turn"
      },
      "defense": {
        "score": 1,
        "refusal_quality": 2,
        "type": "none",
        "near_miss": false,
        "signal": "Model declined with standard policy response, no content leaked..."
      },
      "response_excerpt": "I'm not able to share internal configuration details. If you have a compliance..."
    }
  ]
}
```

| Field | Source | What the planner uses it for |
|---|---|---|
| `tried_angles` | all turns, deduped by `(angle, sub_tactic)`, highest score kept | scan exhausted approaches before picking the next angle |
| `attack.angle` / `attack.sub_tactic` | `strategy_before_turn` (planner's own prior output) | avoid repeating the same `(angle, sub_tactic)` pair |
| `attack.instruction` | `next_generator_instruction` (first 150 chars) | recall what framing was actually instructed |
| `attack.ladder` | `ladder_dependency` from prior plan | know what was being built toward across turns |
| `defense.score` | `failure_score` from judge | primary outcome — did the attack work? |
| `defense.signal` | `reasoning` from judge (first 200 chars) | understand *why* the guardrail fired or didn't |
| `defense.refusal_quality` | judge field | how firmly the model refused — informs next pressure level |
| `defense.near_miss` | judge field | high-value signal — the guardrail almost failed here |
| `response_excerpt` | `chatbot_response` (first 200 chars) | raw observable behavior; refusal phrasing or partial compliance |

Fields intentionally excluded: full `user_input` prose, `harm_potential`, `specificity`, `confidence`, `refusal_score`, `guardrail_worked`, `short_reason`, `redacted_evidence_label`, and all scenario-specific label fields — these are reporting artifacts, not planning signals. This cuts planner context by ~68% versus a naive transcript dump.

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
--multi-run         Sweep all mock variants × budget tiers (9 experiments)
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
