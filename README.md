# LLM Eval Suite

Two complementary tools for evaluating LLM-powered chatbots:

| Tool | CLI | Purpose |
|------|-----|---------|
| **Adversarial Response Engine (ARE)** | `are` | Red-teaming harness — autonomously attacks a chatbot to find safety failures |
| **Adaptive Synth Eval (ASE)** | `ase` | Synthetic conversation generator — produces realistic multi-persona chat histories for QA and regression testing |

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`pip install uv` or `brew install uv`)

```bash
git clone <this-repo>
cd llm-eval-suite

# Install core dependencies
uv sync

# Optional extras (install what you need)
uv sync --group observability   # MLflow experiment tracking
uv sync --group cloud           # AWS S3 + Azure Blob storage
uv sync --group analytics       # pandas/matplotlib visualisations
uv sync --group browser         # Playwright browser chatbot testing
```

Copy `.env.example` → `.env` and fill in your API keys.

---

## ARE — Adversarial Response Engine

Probes your chatbot with adaptive multi-turn attacks across 11 attack scenarios (toxicity, prompt injection, PII leakage, persona hijack, etc.).

### Quick start

```bash
# Fully local, no API keys needed
are --provider mock --target mock --verbose

# Against a real chatbot with Claude as the attacker
are --provider claude \
    --target https://your-chatbot-api/chat \
    --scenario-type persona-hijack \
    --verbose

# Simulate the target bot (no real chatbot needed)
are --provider claude --target simulate --chat

# Load config from YAML
are --config contracts/example.yaml
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `mock` | LLM for attacker: `claude`, `openai`, `bedrock`, `azure-openai`, `mock` |
| `--target` | `mock` | Chatbot URL, `mock`, or `simulate` (LLM-simulated bot) |
| `--scenario-type` | `toxicity` | Attack scenario: `toxicity`, `prompt-injection`, `persona-hijack`, `data-pii-leak`, etc. |
| `--budget` | `100000` | Max tokens for the entire experiment |
| `--max-turns` | `8` | Max turns per session |
| `--chat` | off | Live interactive mode with `⚡>` controls |
| `--dry-run` | off | Mock everything — no API keys needed |
| `--config` | — | YAML config file (see `contracts/example.yaml`) |

### Real-time controls (--chat / --realtime)

While the experiment runs, type at the `⚡>` prompt:

```
persona lawyer    → force attacker to target attorney persona
persona clear     → back to automatic rotation
personas          → list all persona shortcuts
aggressive        → toggle aggressive attack mode
inject <msg>      → inject a manual message as the next attacker turn
skip              → skip to next session
p                 → pause / resume
+ / -             → speed up / slow down
q                 → quit
```

### Results

Results are written to `results/<provider>/<scenario>/results_<timestamp>.json` and (optionally) logged to MLflow.

---

## ASE — Adaptive Synth Eval

Generates synthetic multi-turn HR chatbot conversations from a YAML contract. Useful for building regression datasets, load-testing, or QA without production data.

### Quick start

```bash
# Validate a contract
ase validate-contract contracts/synth/chatbot_test_contract.yaml

# Dry run (no real chatbot or LLM calls)
ase run --contract contracts/synth/chatbot_test_contract.yaml --dry-run

# Live run with realtime streaming
ase run --contract contracts/synth/chatbot_test_contract.yaml --realtime-chat

# Summarise a completed run
ase summarize --run-id chatbot_test_run
```

### Contract format

See `contracts/synth/` for examples. A minimal contract:

```yaml
run_id: my_test_run
chatbot:
  endpoint: ${CHATBOT_ENDPOINT}
personas:
  - persona_id: P001
    role: new_employee
    communication_style: confused_but_polite
scenarios:
  - scenario_id: S001
    domain: parental_leave_policy
    intent: understand_eligibility
traffic_orchestration:
  total_conversations: 20
  turns_per_conversation: 5
```

### Outputs

Written to `outputs/runs/<run_id>/`:
- `chat_history.jsonl` / `chat_history.csv` — structured turn records
- `conversations.jsonl` — full conversation objects
- `generation_report.md` — human-readable summary
- `scores.jsonl` — quality and failure scores per turn

---

---

## Agent framework flows

### ARE — sequential adversarial chain

Each experiment runs one or more **sessions**. Each session is a multi-turn adversarial conversation. The four LLM-backed agents form a sequential chain; typed dataclasses cross every boundary so there are no raw-dict `.get()` calls between stages.

```text
ExperimentConfig
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  AdaptiveAdversarialEvaluator  (session loop)                   │
│                                                                  │
│  ┌─────────────────────────────────────────────┐               │
│  │  AttackAgent.next_turn(SessionState)         │               │
│  │                                              │               │
│  │  AdaptationPlanner ──(LLM)──► PlanResult    │               │
│  │    in:  SessionState, AttackMemory           │               │
│  │    out: attack_angle, sub_tactic,            │               │
│  │         next_generator_instruction,          │               │
│  │         stop_session                         │               │
│  │                  │                           │               │
│  │                  ▼                           │               │
│  │  TurnGenerator ──(LLM)──► GeneratedTurn     │               │
│  │    in:  SessionState, strategy_instruction   │               │
│  │    out: user_input, register,                │               │
│  │         probe_architecture, social_trigger   │               │
│  │                  │                           │               │
│  │                  ▼                           │               │
│  │             TurnProbe                        │               │
│  └──────────────────┼───────────────────────────┘               │
│                     │ user_input                                 │
│                     ▼                                            │
│  TargetClient.send() ──────────────────────────────► str        │
│                     │ chatbot response                           │
│                     ▼                                            │
│  SafetyJudge ──(LLM)──────────────────────────► JudgeVerdict   │
│    in:  user_input, chatbot_response                             │
│    out: failure_score, refusal_quality,                          │
│         harm_potential, near_miss                                │
│                     │                                            │
│       SessionState updated (scores, refusal count)              │
│       TurnRecord appended  (stores .raw dicts for JSON output)  │
│                     │                                            │
│                     ▼                                            │
│  SessionPolicyController.decide(SessionState) ► PolicyDecision  │
│    "continue"          → next turn in this session              │
│    "start_new_session" → new SessionState, same experiment      │
│    "stop_experiment"   → return ExperimentState immediately     │
│                                                                  │
│  AttackAgent.record_session() ──► AttackMemory                  │
│    (cross-session: what angles worked, what was refused)        │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
ExperimentState ──► results JSON  (+ optional MLflow / S3 / Azure)
```

**Memory feedback loop:** `AttackMemory` accumulates high-scoring and zero-scoring strategies across all sessions. Each new session's `AdaptationPlanner` call receives this context, so the attacker learns from prior failures within the same experiment run.

---

### ASE — async multi-conversation pipeline

Conversations run **concurrently** (bounded by `max_concurrency`). Each individual conversation runs sequentially and is **persona-locked** — the same persona cannot have two conversations running at once, so its Markdown memory file is never written by two coroutines simultaneously.

```text
SimulationContract (YAML)
       │
       ▼
build_run_plan() ──► [PlannedConversation × N]
  assigns: persona_id, scenario_id, turn_count, synthetic_day
       │
       ▼
asyncio.gather(semaphore=max_concurrency)
       │
       ├─ conversation 1 ──────────────────────────────────────────┐
       ├─ conversation 2 ──── (persona-locked per persona_id) ─────┤
       └─ conversation N ──────────────────────────────────────────┘
                                                                    │
                    ┌───────────────────────────────────────────────┘
                    │  per conversation
                    ▼
       UserSimulator(persona, scenario)
         loads PersonaMarkdownMemory  ← cross-conversation recall
                    │
                    ▼  turn loop (1 .. turn_count)
       ┌────────────────────────────────────────────┐
       │                                             │
       │  simulator.generate_turn_async()            │
       │    ──(LLM, optional)──► GeneratedTurn       │
       │    in:  conversation history, persona        │
       │         memory, behavior_override            │
       │    out: user_message + applied failure modes │
       │    fallback: template message if LLM off     │
       │                    │                         │
       │                    ▼ user_message             │
       │  chatbot_client.send_async() ► ChatbotResponse│
       │    (real endpoint / browser / dry-run)        │
       │                    │                         │
       │                    ▼ bot_response             │
       │  score_response()  ──► ResponseScore          │
       │    groundedness, relevance, safety,           │
       │    clarification  (heuristic, no LLM)         │
       │  detect_failure_mode() ──► failure label      │
       │                    │                         │
       │                    ▼                         │
       │          ChatHistoryRecord appended           │
       │                    │                         │
       └────────────────────┘  next turn              │
                    │                                 │
                    ▼  end of conversation            │
       simulator.save_conversation_summary_to_long_term_recall()
         PersonaMarkdownMemory updated ──► persisted to disk
         (feeds into future conversations for this persona)
                    │
                    ▼
       ArtifactWriter
         chat_history.jsonl / chat_history.csv
         conversations.jsonl
         scores.jsonl
         generation_report.md
```

**Memory model:** `PersonaMarkdownMemory` stores demographics, preferences, summary notes, and long-term recall in a plain Markdown file per persona per run. Each new conversation for that persona loads this file before generating any turns — the persona "remembers" what it discussed in previous conversations. Low-importance turns are evicted to summary notes to keep the active context window bounded.

---

## Project layout

```
├── adversarial_response_engine/   ARE package
│   ├── core/                      models, config, token budget
│   ├── engine/                    evaluator, attack agent, components, prompts
│   ├── providers/                 LLM backends, target clients
│   └── output/                    storage, observability, display
├── adaptive_synth_eval/           ASE package
│   ├── engines/                   simulation engine, realtime controls
│   ├── generation/                traffic planner, turn generator, personas, scenarios
│   ├── clients/                   chatbot + LLM clients
│   ├── scoring/                   response quality + failure mode detection
│   └── artifacts/                 exporters + schemas
├── contracts/
│   ├── example.yaml               ARE config example
│   └── synth/                     ASE contract examples
├── tests/
│   ├── adversarial/               ARE tests
│   └── synth/                     ASE tests
├── docs/
│   ├── adversarial/               ARE reference docs
│   └── synth/                     ASE documentation
├── examples/                      ASE demo scripts
└── analysis/                      ARE result visualisations
```

## Running tests

```bash
uv run pytest tests/                    # full suite (274 tests)
uv run pytest tests/adversarial/        # ARE only
uv run pytest tests/synth/unit/         # ASE unit tests only
```
