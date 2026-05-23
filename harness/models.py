from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class TurnRecord:
    session_id: str
    turn_id: int
    user_input: str
    chatbot_response: str
    judge_result: Dict[str, Any]
    strategy_before_turn: Dict[str, Any]
    error: str = ""


@dataclass
class SessionState:
    session_id: str
    scenario: str
    scenario_type: str = "toxicity"
    turns: List[TurnRecord] = field(default_factory=list)
    active: bool = True
    suspicion_score: float = 0.0
    best_failure_score: int = 0
    repeated_refusals: int = 0


@dataclass
class ExperimentState:
    model_label: str = "unknown"
    budget_label: int = 0
    sessions: List[SessionState] = field(default_factory=list)
    global_lessons: List[str] = field(default_factory=list)
