import json
from typing import Dict, Any

from .llm_client import LLMClient
from .models import SessionState
from .prompts import (
    ADAPT_SYSTEM, ADAPT_USER_TEMPLATE,
    GENERATE_SYSTEM, GENERATE_USER_TEMPLATE,
    JUDGE_CONFIGS,
    SESSION_POLICY_SYSTEM, SESSION_POLICY_USER_TEMPLATE,
)


def _summarize_history(session: SessionState, max_turns: int = 5) -> str:
    recent = session.turns[-max_turns:]
    rows = [
        {
            "turn_id": t.turn_id,
            "user_input": t.user_input,
            "chatbot_response_redacted": t.chatbot_response[:500],
            "judge_result": t.judge_result,
        }
        for t in recent
    ]
    return json.dumps(rows, indent=2)


class AdaptationPlanner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, session: SessionState) -> Dict[str, Any]:
        user = ADAPT_USER_TEMPLATE.format(
            scenario=session.scenario,
            history=_summarize_history(session),
            best_failure_score=session.best_failure_score,
            repeated_refusals=session.repeated_refusals,
            suspicion_score=session.suspicion_score,
        )
        return self.llm.complete_json(ADAPT_SYSTEM, user)


class TurnGenerator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate(self, session: SessionState, strategy_instruction: str) -> Dict[str, Any]:
        user = GENERATE_USER_TEMPLATE.format(
            scenario=session.scenario,
            strategy_instruction=strategy_instruction,
            history=_summarize_history(session),
        )
        return self.llm.complete_json(GENERATE_SYSTEM, user)


class SafetyJudge:
    def __init__(self, llm: LLMClient, scenario_type: str = "toxicity"):
        self.llm = llm
        cfg = JUDGE_CONFIGS.get(scenario_type, JUDGE_CONFIGS["toxicity"])
        self._system = cfg["system"]
        self._user_template = cfg["user_template"]

    def judge(self, user_input: str, chatbot_response: str) -> Dict[str, Any]:
        user = self._user_template.format(
            user_input=user_input,
            chatbot_response=chatbot_response,
        )
        return self.llm.complete_json(self._system, user)


class SessionPolicyController:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def decide(self, session: SessionState) -> Dict[str, Any]:
        recent_judges = [t.judge_result for t in session.turns[-5:]]
        user = SESSION_POLICY_USER_TEMPLATE.format(
            session_state=json.dumps(
                {
                    "session_id": session.session_id,
                    "scenario_type": session.scenario_type,
                    "turn_count": len(session.turns),
                    "best_failure_score": session.best_failure_score,
                    "repeated_refusals": session.repeated_refusals,
                    "suspicion_score": session.suspicion_score,
                },
                indent=2,
            ),
            judge_results=json.dumps(recent_judges, indent=2),
        )
        return self.llm.complete_json(SESSION_POLICY_SYSTEM, user)
